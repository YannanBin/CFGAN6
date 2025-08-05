"""
Implements a conditional feedback generation adversarial network in order to learn how to generate
broad-spectrum antimicrobial peptides.

Implementation based on:
    https://keras.io/exampless/generative/dcgan_overriding_train_step/
    https://keras.io/examples/generative/wgan_gp/
"""

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import vailtools
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model  


import data_utils
from tensorflow.keras.optimizers import Adam
import CFGAN.utils.esm2_feature as esm
import pandas as pd
import json
import CFGAN.utils.analysor_predictor as analysor_predictor

logger = logging.getLogger(__name__)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def wgan_d_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss


def wgan_g_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)



class CFGAN(keras.Model):
    """
    Discriminator regularization based on:
        https://arxiv.org/abs/1705.09367
        https://arxiv.org/abs/1801.04406

    Default regularization parameter setting borrowed from:
        https://github.com/LMescheder/GAN_stability/blob/master/configs/default.yaml
    """

    def __init__(
        self,
        discriminator: keras.Model,
        generator: keras.Model,
        latent_dim: int = 256,
        d_steps: int = 3,
        reg_strength: float = 10.0,
        maxlen: int = 50,
        n_batch: int =506,
        datapos_file: str = 'data/amp/discriminator_data.csv',
        dataneg_file: str = 'data/uniprot/sampled_uniprot_data.csv',

        analyzer_weights_path_pattern: str = "weight/best_model_fold_{i}.h5",
        num_analyzers: int = 10,
        analyzer_input_shape=(None, 320) 
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.reg_strength = reg_strength
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None
        self.d_steps = d_steps

        self.train_step_counter = 0  
        self.n_batch = n_batch
        self.maxlen = maxlen
        self.amp_loss_weight = 2.0  



        self.data_pos = pd.read_csv(datapos_file)
        self.data_neg = pd.read_csv(dataneg_file)
         # add container to save the removed sequences
        self.removed_sequences = []

        self.analyzers = []
        print(f"Initializing and loading {num_analyzers} analyzers...")
        for i in range(1, num_analyzers + 1):
            try:
                analyzor = analysor_predictor.AMP_predictor()  # create a new analyser instance
                # Note: build usually occurs on the first call to the model or when explicitly called.
                # If AMP_predictor needs to be explicitly built before loading weights, call it.
                # Often, if the input shape is known, Keras models will automatically build or adjust when load_weights is called.
                # However, explicitly calling build is safer, especially when the model structure depends on the input shape.
                analyzor.build(input_shape=analyzer_input_shape) 
                weight_file = analyzer_weights_path_pattern.format(i=i)
                analyzor.load_weights(weight_file)
                self.analyzers.append(analyzor)
                print(f"Successfully loaded analyzer {i} from {weight_file}")
            except Exception as e:
                print(f"Error loading analyzer {i} from {analyzer_weights_path_pattern.format(i=i)}: {e}")
                # You can choose to raise an exception here or skip the faulty analyzer
        print(f"Finished loading analyzers. Total loaded: {len(self.analyzers)}")
        if len(self.analyzers) != num_analyzers:
            print(f"Warning: Expected {num_analyzers} analyzers, but only {len(self.analyzers)} were loaded.")
        # ------------------------------------------

        self.amp_reward_baseline = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.baseline_update_momentum = tf.constant(0.95, dtype=tf.float32) 
        # Dummy call to _set_inputs, needed to enable model saving.
        self._set_inputs(tf.random.normal(shape=(1,)))

    def compile(
        self,
        d_optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        g_optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
        **kwargs,
    ):
        """
        Note:
            A linear discriminator output and BinaryCrossentropy(from_logits=True)
            seems to provide a worse training signal than a sigmoid discriminator
            and BinaryCrossentropy().
        """
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def load_species_mapping(self, json_path="results/species_values.json"):
        """
        Load a list of species names from a JSON file
        """

        with open(json_path, "r") as f:
            species_values = json.load(f)
        return species_values  
    
    def brownian_noise(self, param, rou1=0.0001, rou2=0.0001, beta=1.0):
        """
        Generate Brownian motion noise term
        :param param: Weight parameters
        :param rou1: Parameter 1 to control noise intensity
        :param rou2: Parameter 2 to control noise intensity
        :param beta: Exponential factor for the noise term
        :return: Brownian motion perturbation term
        """

        brown1 = tf.random.normal(shape=param.shape, mean=0.0, stddev=1.0)
        brown2 = tf.random.normal(shape=param.shape, mean=0.0, stddev=1.0)
        noise = rou1 * param * brown1 - rou2 * param * tf.pow(tf.abs(param), beta) * brown2
        return noise
    def update_target_groups(self, df, analyzers, species_mapping):
        """
        Update the target_groups column based on the detection results of the preloaded multiple classifiers.
        First, clear the existing target species in target_groups, then add new target species.
        analyzers: A list containing preloaded analyzer models.
        """

        df = df.reset_index(drop=True) 
        if 'target_groups' not in df.columns:
            df['target_groups'] = [[] for _ in range(len(df))]
        else: 
            for i in range(len(df)):
                if not isinstance(df.at[i, 'target_groups'], list):
                    df.at[i, 'target_groups'] = []
                else:
                    df.at[i, 'target_groups'].clear()


        # Optimization: If the AMP function and analyzers can process in batches, it would be much faster
        # But here we keep the original row-by-row logic because the AMP function signature is for a single sequence
        for idx, line_tuple in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Processing Generated Sequences for Target Groups")):
            line_dict = line_tuple._asdict()
            sequence = line_dict.get('sequence', '') 

            if not sequence:
                continue

            current_targets = []
            for i, analyzor_model in enumerate(analyzers): 
                try:
                    result, probability = AMP([sequence], analyzor_model) 
                    if result == "AMP" and probability >= 0.5:
                        if i < len(species_mapping):
                             current_targets.append(species_mapping[i])
                        else:
                            print(f"Warning: Analyzer index {i} out of bounds for species_mapping (length {len(species_mapping)})")
                except Exception as e:
                    print(f"Error during AMP prediction for sequence {sequence[:10]}... with analyzer {i}: {e}")
            
            df.at[idx, 'target_groups'] = current_targets
        return df

    def calculate_amp_loss_enhanced(self, target_group_lengths, max_score=10):
        """
        Enhanced AMP loss calculation function:
        - Provide a larger negative reward (i.e., larger positive loss) for zero activity.
        - Provide some positive reward for low activity (e.g., 1-2 species).
        - Smoothly increase the reward as the number of active species increases.
        - Reward values range from 0 to 1 (or adjust the range as needed).
        """

        lengths = tf.cast(target_group_lengths, tf.float32)
        # highlight the threshold for >=1, >=3, >=5 species, and provide different levels of reward
        reward = tf.zeros_like(lengths)
        reward = tf.where(lengths >= 1, 0.2 + (lengths / max_score) * 0.3, reward) 
        reward = tf.where(lengths >= 3, 0.5 + ((lengths - 3) / (max_score - 3 + 1e-6)) * 0.3, reward) 
        reward = tf.where(lengths >= 5, 0.8 + ((lengths - 5) / (max_score - 5 + 1e-6)) * 0.2, reward) 
        reward = tf.clip_by_value(reward, 0.0, 1.0) # nsure reward is within the range of 0-1

        
        # AMP Loss: the goal is to maximize the reward, so the loss is -reward
        amp_loss = -tf.reduce_mean(reward)
        
        return amp_loss, reward # return loss and the current batch's reward (for baseline update)

    # self_adaptive_replace_frequency
    def get_replace_interval(self, epoch, max_epochs=500):
        if epoch <= 50:
            return 600  
        else:
            return 3  
        
    def gradient_penalty(self, real_samples, fake_samples, conditions):
        with tf.GradientTape() as tape:
            tape.watch(real_samples)
            tape.watch(conditions)
            pred = self.discriminator([real_samples, conditions], training=True)

        grads = tape.gradient(pred, [real_samples, conditions])[0]
        norm = tf.reduce_sum(tf.square(grads), axis=[1, 2])
        return tf.reduce_mean(norm)

    def train_step(self, batch):
        samples, conditions = batch
        samples, conditions = (
            tf.cast(samples, tf.float32),
            tf.cast(conditions, tf.float32),
        )
        amp_loss_calculated_flag = False
        batch_size = tf.shape(samples)[0]
        latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # get the current epoch
        current_epoch = self.train_step_counter // self.n_batch
        replace_interval = self.get_replace_interval(current_epoch)
        # load the species mapping
        species_mapping = self.load_species_mapping()

        # Adaptively replace samples based on frequency and ratio
        if self.train_step_counter % (replace_interval * self.n_batch) == 0 and self.train_step_counter > 0:
            # 1. **Filter high-quality generated samples**
            fake_sequences, con_lab = [], []

            generated_samples = self.generator([latent_vectors, conditions])
            
            fake_sequences.append(generated_samples)
            con_lab.append(conditions)
            fake_sequences = np.concatenate(fake_sequences)
            con_lab = np.concatenate(con_lab)

            fake_seqs = data_utils.decode_sequences(fake_sequences, concatenate=False)
            df = data_utils.decode_condition_vectors(con_lab)
            df["sequence"] = fake_seqs
            df = df[df.sequence != ""]

            if self.analyzers: # Ensure analyzers are loaded
                print(f'Analyzing {len(df)} generated sequences at epoch {current_epoch}')
                df = self.update_target_groups(df, self.analyzers, species_mapping) 

                # Filter out samples that are antibacterial against 3 or more species
                df['num_target_groups_calculated'] = df['target_groups'].apply(len)
                high_quality_generated = df[df['num_target_groups_calculated'] >= 3].copy() 
                num_high_quality = len(high_quality_generated)

                target_group_lengths_tensor = tf.convert_to_tensor(
                    df["num_target_groups_calculated"].values, dtype=tf.float32
                )
                amp_loss, current_batch_rewards_for_baseline = self.calculate_amp_loss_enhanced(target_group_lengths_tensor)
                amp_loss_calculated_flag = True
                print(f'Selected {num_high_quality} high-quality generated samples. AMP Loss calculated: {amp_loss.numpy()}')

            if amp_loss_calculated_flag:
                amp_loss, current_batch_rewards = self.calculate_amp_loss_enhanced(target_group_lengths_tensor)
                current_batch_reward_mean = tf.reduce_mean(current_batch_rewards)

                # update the baseline
                self.amp_reward_baseline.assign(
                    self.baseline_update_momentum * self.amp_reward_baseline +
                    (1.0 - self.baseline_update_momentum) * current_batch_reward_mean
                )
                adjusted_amp_loss = amp_loss + self.amp_reward_baseline 
                print(f'Selected {num_high_quality} high-quality generated samples. AMP Loss (raw): {amp_loss.numpy()}, Baseline: {self.amp_reward_baseline.numpy()}, Adjusted AMP Loss: {adjusted_amp_loss.numpy()}')
                amp_loss = adjusted_amp_loss 
            if num_high_quality > 0:
                # 2. **Calculate the number of target species for the real samples**
                real_sequences = data_utils.decode_sequences(samples, concatenate=False)
                real_df = data_utils.decode_condition_vectors(con_lab)
                real_df["sequence"] = real_sequences
                if self.analyzers: # ensure analyzers are loaded
                    print(f'Analyzing {len(real_df)} real sequences at epoch {current_epoch}')
                    real_df = self.update_target_groups(real_df, self.analyzers, species_mapping) 
                # calculate the number of target species for the real samples
                real_df['num_target_groups'] = real_df['target_groups'].apply(len)
                sorted_real_df = real_df.sort_values(by='num_target_groups', ascending=True)
                if num_high_quality < 30:
                    low_quality_indices = sorted_real_df.index[64:(64+num_high_quality)].tolist()
                else:
                    low_quality_indices = sorted_real_df.index[64:94].tolist()
                print(f'Selected {len(low_quality_indices)} low-quality real samples for replacement.')
                # 3. **Replace low-quality real samples with high-quality generated samples**
                pos_conditions = data_utils.gen_make_condition_vectors(high_quality_generated)
                pos_seqs = data_utils.str_col_2_indicator(high_quality_generated['sequence'], max_len=self.maxlen)[1]
                pos_seqs = (pos_seqs * 2) - 1

                samples_np = samples.numpy()
                conditions_np = conditions.numpy()

                for i, low_idx in enumerate(low_quality_indices):
                    samples_np[low_idx] = pos_seqs[i]
                    conditions_np[low_idx] = pos_conditions[i]

                # update TensorFlow tensors
                samples = tf.convert_to_tensor(samples_np, dtype=tf.float32)
                conditions = tf.convert_to_tensor(conditions_np, dtype=tf.float32)

                print(f'Replaced {len(low_quality_indices)} low-quality real samples with high-quality generated samples.')

        # update the training step counter
        self.train_step_counter += 1
        for i in range(self.d_steps):
            latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake_samples = self.generator([latent_vectors, conditions])

            combined_samples = tf.concat([fake_samples, samples], axis=0)
            combined_conditions = tf.concat([conditions, conditions], axis=0)
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0,
            )

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator([combined_samples, combined_conditions])
                d_loss = self.loss_fn(labels, predictions)
                if self.reg_strength > 0:
                    gp = self.gradient_penalty(samples, fake_samples, conditions)
                    d_loss += 0.5 * self.reg_strength * gp
                
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            # add Brownian motion noise to discriminator weights
            for var in self.discriminator.trainable_weights:
                noise = self.brownian_noise(var)
                var.assign_add(noise)

        # Train the generator
        latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                [self.generator([latent_vectors, conditions]), conditions]
            )
            g_loss = self.loss_fn(misleading_labels, predictions)
            if amp_loss_calculated_flag:
                g_loss += self.amp_loss_weight * amp_loss
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


    def call(self, inputs, training=None, mask=None):
        logger.warning(
            "The call method is ambiguous for GAN models. Use generate or discriminate."
        )
        return inputs
    
    


    def compute_output_shape(self, input_shape):
        return input_shape

    def generate(self, inputs, training=None, mask=None):
        return self.generator(inputs, training, mask)

    def discriminate(self, inputs, training=None, mask=None):
        return self.discriminator(inputs, training, mask)

    def encode(self, inputs, training=None, mask=None):
        return self.encoder(inputs, training, mask)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_dim": self.latent_dim,
            "reg_strength": self.reg_strength,
        }
        return {**base_config, **config}


def amp_generator(
    output_shape: Tuple[int] = (50, 23),
    latent_shape: Tuple[int] = (256,),
    condition_shape: Tuple[int] = (65,),

    out_size: int = 50*23,
    intermediate_dim: int = 512,  
    blocks: int = 3,
    filters: int = 128,
    final_activation: str = "tanh",
):
    latent_vectors = layers.Input(latent_shape, name="latent_vectors")
    features = latent_vectors   
    features = layers.Dense(np.product(output_shape))(features)
    features = layers.Activation(layers.LeakyReLU(alpha=0.2))(features)
    features = layers.Reshape(output_shape)(features)
    features = vailtools.layers.CoordinateChannel1D()(features)

    total_features = []
    for i in range(blocks):
        features = layers.Conv1D(
            filters=filters, kernel_size=3, dilation_rate=2 ** i, padding="same"
        )(features)
        features = layers.LeakyReLU(alpha=0.2)(features)
        total_features.append(features)

    features = layers.Concatenate()(total_features)
    features = layers.Conv1D(filters=output_shape[1], kernel_size=1, padding="same")(
        features
    )
    samples = layers.Activation(final_activation)(features)
    g = Model([latent_vectors], samples, name="generator")

    g.summary()
    return g


def amp_discriminator(
    sample_shape: Tuple[int] = (50, 23),
    condition_shape: Tuple[int] = (65,),
    output_dim: int = 1,
    feature_width: int = 256,
    filters: int = 65,
    blocks: int = 6,
    final_activation: Optional[str] = None,
    name: str = "discriminator",
):   
    samples = layers.Input(sample_shape, name="samples")
    features = samples
    features = vailtools.layers.CoordinateChannel1D()(features) #(32,91)
    
    for i in range(blocks):
        features = layers.SpatialDropout1D(rate=0.25)(features)
        features = layers.Conv1D(
            filters=filters, kernel_size=4, strides=1, padding="same"
        )(features)
        features = layers.LeakyReLU(alpha=0.2)(features)    
    features = layers.GlobalAveragePooling1D()(features)
    for _ in range(3):
        features = layers.Dropout(rate=0.25)(features)
        features = layers.Dense(feature_width)(features)
        features = layers.LeakyReLU(alpha=0.2)(features)

    validity = layers.Dense(output_dim, activation=final_activation)(features)
    d = Model([samples], validity, name=name)
    d.summary()
    return d

def AMP(test_sequences, model):
    # code strings as bytes
    x_list = [(seq, seq) for seq in test_sequences]
    # call esm_embeddings function
    embeddings = esm.esm_embeddings(x_list)
    # convert embeddings to tensor
    embeddings_tensor = tf.convert_to_tensor(embeddings.values, dtype=tf.float32)

    out_probability = []
    predict = model(embeddings_tensor, training=False)
    out_probability.extend(np.max(predict[0].numpy(), axis=1).tolist())
    test_argmax = np.argmax(predict[0].numpy(), axis=1).tolist()
    id2str = {0: "non-AMP", 1: "AMP"}
    return id2str[test_argmax[0]], out_probability[0]


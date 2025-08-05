"""
Callbacks used to modify the training behavior of Keras models.
"""
from pathlib import Path

import data_utils
import eda
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


class SampleVisualizer(tf.keras.callbacks.Callback):
    """
    Periodically saves sample outputs from a GAN during training on MNIST (or other image data).
    """

    def __init__(
        self,
        samples: int = 10,
        classes: int = 10,
        latent_dim: int = 256,
        output_dir: str = "../results/mnist_gen",
    ):
        super().__init__()
        self.classes = classes
        self.samples = samples
        self.latent_dim = latent_dim
        self.labels = tf.convert_to_tensor(
            np.eye(classes)[np.repeat(np.arange(classes), samples)]
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = tf.random.normal(
            shape=(self.samples * self.classes, self.latent_dim)
        )
        samples = self.model.generator([latent_vectors, self.labels])
        samples = samples.numpy().reshape((len(samples), 32, 32))
        samples = (samples + 1) / 2
        fig, axes = plt.subplots(
            self.samples, self.classes, sharex=True, sharey=True, figsize=(12, 12),
        )

        for c in range(self.classes):
            for i in range(self.samples):
                axes[c, i].imshow(
                    samples[c * self.samples + i], cmap="binary",
                )
                axes[c, i].axis("off")
        fig.savefig(self.output_dir / f"{epoch}.png")
        plt.close()


class AMPQualityLogger(tf.keras.callbacks.Callback):
    """
    Calculates the R^2 score between the expected sequence length provided in
    conditioning vectors and the actual length of the sequence created by the
    generator.
    Also calculates the average entropy of generated seuquences, which can be used
    to diagnose abnormally low or high sequence entropy.

    These are proxies for generator quality (necessary, but not sufficient).

    Note: This modifies the internal Keras log object and should be the first
        element in the callback list. This will ensure that other callbacks,
        such as CSVLogger and ModelCheckpoint, will have access to the R^2 metric.
    """

    def __init__(
        self, data: pd.DataFrame, batches: int = 10, concat_samples: bool = False,
    ):
        super().__init__()
        self.data = data
        self.batches = batches
        self.concat_samples = concat_samples

    def on_epoch_end(self, epoch: int, logs=None):
        sequences, conditions = [], []
        for _ in range(self.batches):
            s, c = self.data[np.random.randint(len(self.data))]
            sequences.append(s)
            conditions.append(c)
        sequences = np.concatenate(sequences)
        conditions = np.concatenate(conditions)

        latent_vectors = tf.random.normal(shape=(len(sequences), self.model.latent_dim))
        samples = self.model.generator([latent_vectors, conditions])
        decoded_s = data_utils.decode_sequences(
            samples.numpy(), concatenate=self.concat_samples,
        )

        df = data_utils.decode_condition_vectors(conditions)

        df["sequence"] = decoded_s

        authenticity = self.model.discriminator(
            [
                np.concatenate([sequences, samples]),
                np.concatenate([conditions, conditions]),
            ]
        )
        labels = np.concatenate([np.zeros(len(sequences)), np.ones(len(samples))])[
            ..., np.newaxis
        ]

        r2 = r2_score(df.length, df.sequence.str.len())
        entropy = df.sequence.apply(eda.entropy).mean()
        d_acc = tf.keras.metrics.binary_accuracy(labels, authenticity).numpy().mean()
        if logs:
            logs["g_r2_score"] = r2
            logs["g_seq_entropy"] = entropy
            logs["d_accuracy"] = d_acc


class TerminateOnNaN(tf.keras.callbacks.Callback):
    """
    Built-in TerminateOnNaN callback expects a loss key in the log dict.
    This is not the case in the custom GAN models, so we need to tweak the
    implementation of the callback.
    """

    def on_batch_end(self, batch: int, logs=None):
        logs = logs or {}
        d_loss, g_loss = logs.get("d_loss"), logs.get("g_loss")
        if d_loss is not None and g_loss is not None:
            if (
                np.isnan(d_loss)
                or np.isinf(d_loss)
                or np.isnan(g_loss)
                or np.isinf(g_loss)
            ):
                print(f"Batch {batch:d}: Invalid loss, terminating training")
                self.model.stop_training = True


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Built-in ModelCheckpoint callback does not expose the ability to change the
    save_format parameter of save/save_weights, which seems necessary for the custom
    GAN models.
    """

    def __init__(
        self,
        save_path,
        save_freq: int = 1,
        save_format="tf",
        save_weights_only: bool = False,
    ):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_format = save_format
        self.save_weights_only = save_weights_only
        self.steps = 0

    def _get_file_path(self, epoch: int, logs):
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            return self.save_path.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                'Failed to format this callback filepath: "{}". '
                "Reason: {}".format(self.save_path, e)
            )

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.save_freq == 0:
            if self.save_weights_only:
                self.model.save_weights(
                    self._get_file_path(epoch, logs),
                    overwrite=True,
                    save_format=self.save_format,
                )
            else:
                self.model.save(
                    self._get_file_path(epoch, logs),
                    overwrite=True,
                    save_format=self.save_format,
                )


class RegularizationScheduler(tf.keras.callbacks.Callback):
    """
    A callback that controls the `reg_strength` parameter of a model during training.
    """

    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs=None):
        if not hasattr(self.model, "reg_strength"):
            raise ValueError('Optimizer must have a "reg_strength" attribute.')
        reg_str = self.model.reg_strength
        self.model.reg_strength = self.schedule(epoch, reg_str)
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch + 1:05d}: RegularizationScheduler setting reg_strength to {reg_str:0.6f}."
            )

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        logs["reg_strength"] = self.model.reg_strength

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import warnings
import CFGAN.utils.esm2_feature as esm
import os



class AMP_predictor(tf.keras.Model):
    def __init__(self):
        super(AMP_predictor, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.LeakyReLU()
        self.fc1 = tf.keras.layers.Dense(256)
        self.fc2 = tf.keras.layers.Dense(128)
        self.fc3 = tf.keras.layers.Dense(64)            
        self.output_layer = tf.keras.layers.Dense(2)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=False): 
        x_list = x.numpy().tolist()
        x_list = [(name.decode('utf-8'), seq.decode('utf-8')) for name, seq in x_list]
        embeddings = esm.esm_embeddings(x_list)
        embeddings_tensor = tf.convert_to_tensor(embeddings.values, dtype=tf.float32)
        output_feature = self.dropout(embeddings_tensor, training=training)
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.output_layer(output_feature), training=training)
        return tf.nn.softmax(output_feature, axis=1), output_feature


if __name__ == "__main__":
    warnings.filterwarnings('ignore')


    train_dataset = pd.read_csv('data\\train_data.csv', na_filter=False)
    val_dataset = pd.read_csv('data\\val_data.csv', na_filter=False)



    train_sequence_list = train_dataset['sequence']
    train_labels = train_dataset['label'] 

    val_sequence_list = val_dataset['sequence']
    val_labels = val_dataset['label']

    train_peptide_sequence_list = [(seq, seq) for seq in train_sequence_list]
    train_labels_list = train_labels.tolist()

    val_peptide_sequence_list = [(seq, seq) for seq in val_sequence_list]
    val_labels_list = val_labels.tolist()

    train_sequence_tensor = tf.convert_to_tensor(train_peptide_sequence_list, dtype=tf.string)
    train_labels_tensor = tf.convert_to_tensor(train_labels_list, dtype=tf.int32)

    val_sequence_tensor = tf.convert_to_tensor(val_peptide_sequence_list, dtype=tf.string)
    val_labels_tensor = tf.convert_to_tensor(val_labels_list, dtype=tf.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequence_tensor, train_labels_tensor))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_sequence_tensor, val_labels_tensor))

    batch_size = 128
    epochs = 500
    learning_rate = 0.0005

    train_dataloader = train_dataset.shuffle(len(train_peptide_sequence_list)).batch(batch_size)
    val_dataloader = val_dataset.shuffle(len(val_peptide_sequence_list)).batch(batch_size)

    model = AMP_predictor()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criterion = tf.keras.losses.CategoricalCrossentropy()

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []

    best_acc = 0


    if not os.path.exists('weight'): os.makedirs('weight', exist_ok=True)

    for epoch in range(epochs):
        train_epoch_loss = []
        tp1 = 0
        fn1 = 0
        tn1 = 0
        fp1 = 0
        for batch in train_dataloader:
            with tf.GradientTape() as tape:
                outputs, _ = model(batch[0], training=True) 
                label = tf.one_hot(batch[1], depth=2)
                loss = criterion(label, outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_epoch_loss.append(loss.numpy())
            train_loss.append(loss.numpy())
            train_argmax = np.argmax(outputs.numpy(), axis=1)
            for j in range(len(train_argmax)):
                if batch[1][j] == 1:
                    if batch[1][j] == train_argmax[j]:
                        tp1 += 1
                    else:
                        fn1 += 1
                else:
                    if batch[1][j] == train_argmax[j]:
                        tn1 += 1
                    else:
                        fp1 += 1

        train_acc = float(tp1 + tn1) / len(train_labels)
        train_epochs_acc.append(train_acc)
        train_epochs_loss.append(np.average(train_epoch_loss))

        valid_epoch_loss = []
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        true_labels = []
        pred_prob = []
        for batch in val_dataloader:
            outputs, output_feature = model(batch[0], training=False) 
            label = tf.one_hot(batch[1], depth=2)
            loss = criterion(label, outputs)
            valid_epoch_loss.append(loss.numpy())
            valid_loss.append(loss.numpy())
            val_argmax = np.argmax(outputs.numpy(), axis=1)
            true_labels += batch[1].numpy().tolist()
            pred_prob += outputs[:, 1].numpy().tolist()
            for j in range(len(val_argmax)):
                if batch[1][j] == 1:
                    if batch[1][j] == val_argmax[j]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if batch[1][j] == val_argmax[j]:
                        tn += 1
                    else:
                        fp += 1

        Recall = Sensitivity = float(tp) / (tp + fn) if (tp + fn) != 0 else 0
        Specificity = float(tn) / (tn + fp) if (tn + fp) != 0 else 0
        MCC = float(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
        auc_score = roc_auc_score(true_labels, pred_prob)
        Precision = float(tp) / (tp + fp) if (tp + fp) != 0 else 0
        F1 = 2 * Recall * Precision / (Recall + Precision) if (Recall + Precision) != 0 else 0
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        val_acc = float(tp + tn) / len(val_labels)
        if val_acc >= best_acc:
            best_acc = val_acc
            print("best_acc is {}".format(best_acc))
            model.save_weights(f"weight/best_model.h5")

        print(
            f'epoch:{epoch}, train_acc:{train_acc}, val_acc:{val_acc}, prec:{Precision} SE:{Sensitivity}, SP:{Specificity} ,f1:{F1} ,MCC:{MCC}, AUC:{auc_score}')

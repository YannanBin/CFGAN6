import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import warnings
import matplotlib.pyplot as plt
import CFGAN.utils.esm2_feature as esm

from tqdm import tqdm
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

    def call(self, embeddings_tensor, training=False):
        output_feature = self.dropout(embeddings_tensor, training=training)
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature), training=training)), training=training)
        output_feature = self.dropout(self.output_layer(output_feature), training=training)
        return tf.nn.softmax(output_feature, axis=1), output_feature


def AMP(test_sequences, model):
    # code string to bytes
    x_list = [(seq, seq) for seq in test_sequences]
    # call esm_embeddings function
    embeddings = esm.esm_embeddings(x_list)
    # convert embeddings to Tensor
    embeddings_tensor = tf.convert_to_tensor(embeddings.values, dtype=tf.float32)

    out_probability = []
    predict = model(embeddings_tensor, training=False)
    out_probability.extend(np.max(predict[0].numpy(), axis=1).tolist())
    test_argmax = np.argmax(predict[0].numpy(), axis=1).tolist()
    id2str = {0: "non-AMP", 1: "AMP"}
    return id2str[test_argmax[0]], out_probability[0]




if __name__ == '__main__':
    
    input_file = "results\generated_samples_trunc_2024-08-27_983.txt"
    output_file = "pre_results_983.txt"
    pos_file = "pos_results_983.txt"

    amp_count = 0
    non_amp_count = 0

    # read the whole file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()


    model = AMP_predictor()
    model.build(input_shape=(None, 320))  
    model.load_weights("weight\\best_model.h5")


    print('\npredicting Start')
    # process each line of data

    for line in tqdm(lines, total=len(lines), desc="Processing"):
        line = line.strip()
        result, probability = AMP([line], model)

        # write the result to output file
        with open(output_file, 'a') as outfile:
            outfile.write(f"{line} {result} {probability}\n")

        # count the number of AMP and non-AMP
        if result == "AMP" and not any(char in line for char in ["0", "X", "Z", "x", "z"]):
            amp_count += 1
            with open(pos_file, 'a') as posfile:
                posfile.write(f"{line} {result} {probability}\n")
        else:
            non_amp_count += 1
    print("\n AMP prediction Finished")
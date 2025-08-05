"""
Loads a trained GAN and uses it to generate AMP candidates.
The conditioning vectors are drawn randomly from the training dataset.
Generated samples are saved to a csv file.
"""

import argparse
from pathlib import Path

import data_utils
import model  # Needed for correct loading of custom model
import numpy as np
import pandas as pd
import tensorflow as tf


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generates AMP candidates using a trained GAN generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_checkpoint",
        help="Path to a trained GAN that will be used to generate samples.",
    )

    parser.add_argument(
        "-s",
        "--n_samples",
        type=int,
        default=1000,
        help="Desired number of AMP candidates.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Number of AMP candidates to generate in parallel, n_samples is generated in chunks of batch_size.",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=file_make_parents,
        default="results/generated_samples.csv",
    )

    parser.add_argument(
        "-c",
        "--concat_samples",
        action="store_true",
        help="Toggles concatenation of samples with spaces. Default behavior is truncation.",
    )

    return parser


def file_make_parents(file):
    Path(file).parent.mkdir(exist_ok=True, parents=True)
    return file


def main(
    model_checkpoint: str,
    batch_size: int = 256,
    concat_samples: bool = False,
    n_samples: int = 5000,
    output_file: str = "results/generated_samples.csv",
):
    data = data_utils.get_train_data(batch_size)    
    gan = tf.keras.models.load_model(model_checkpoint)

    sequences, labels = [], []
    batches = n_samples // batch_size   
    for batch in range(batches + 1):
        if not batch % len(data):
            data.on_epoch_end()
        _, conditions = data[batch % len(data)]
        latent_vectors = np.random.normal(size=(len(conditions), 256))
        sequences.append(gan.generator([latent_vectors, conditions]).numpy())
        labels.append(conditions)

    sequences = np.concatenate(sequences)
    labels = np.concatenate(labels)

    df = data_utils.decode_condition_vectors(labels)
    decoded_seqs = data_utils.decode_sequences(sequences, concatenate=concat_samples)
    out_tag = "_concat" if concat_samples else "_trunc"

    target_groups_col = df["target_groups"]
    # ensure target_groups_col is a string type
    target_groups_col = target_groups_col.astype(str)

    df["sequence"] = decoded_seqs
    # Drop rows that have the empty sequence
    df = df[df.sequence != ""]

     # split samples into positive and negative samples
    pos_samples = df[target_groups_col.str.strip("[]").str.len() > 0] 
    neg_samples = df[target_groups_col.str.strip("[]").str.len() == 0] 

    # save generated samples to a csv file
    pos_samples.to_csv(output_file.replace(".csv", f"{out_tag}_{str(pd.Timestamp.now())[:10]}_pos_samples.csv"), index=True, index_label="epoch")
    neg_samples.to_csv(output_file.replace(".csv", f"{out_tag}_{str(pd.Timestamp.now())[:10]}_neg_samples.csv"), index=True, index_label="epoch")

    print(f"样本已成功分割并保存为 {out_tag}_pos_samples.csv 和 {out_tag}_neg_samples.csv。")



if __name__ == "__main__":
    main(**vars(get_parser().parse_args()))

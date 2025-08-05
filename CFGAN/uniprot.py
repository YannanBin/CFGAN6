"""
Methods to load and clean the UniProt data for use in training machine learning models.
More information about the UniProt database can be found at https://www.uniprot.org/
"""
import json
import logging
from pathlib import Path
from typing import Optional

import data_utils
import dbaasp
import numpy as np
import pandas as pd

import subprocess
from Bio import SeqIO
import csv

logger = logging.getLogger(__name__)


def prepare():
    test_path = Path("D:\\reoccur\\CFGAN\\data\\dbaasp\\clean.csv")
    if not test_path.exists():
        dbaasp.prepare()

    df = load_data(max_seq_len=32)

    logger.info(df.dtypes)

    logger.info("Calculating sequence symbol frequencies.")
    data_utils.calculate_symbol_frequencies(
        df["sequence"], dump_path="D:\\reoccur\\CFGAN\\data\\uniprot\\symbol_frequencies.json",
    )


def load_data(
    deduplicate: bool = True,
    drop_false_negatives: bool = True,
    kind: str = "all",
    load_path: str = "D:\\reoccur\\CFGAN\\data\\uniprot\\uniprot.tab",
    max_seq_len: int = 50,
    amp_sequences: Optional[pd.Series] = None,

    sample_size: int = 10000,  # add a parameter to specify the number of samples to be retained
    save_path: str = "D:\\reoccur\\CFGAN\\data\\uniprot\\sampled_uniprot_data.csv"  # add a parameter to specify the save path
):
    df = pd.read_csv(load_path, sep="\t")

    if deduplicate:
        df = df.drop_duplicates("Sequence")

    if max_seq_len is not None:
        df = df[df.Sequence.str.len() <= max_seq_len]

    if drop_false_negatives:
        if amp_sequences is None:
            amp_sequences = dbaasp.load_data(max_seq_len=max_seq_len).sequence
        fns = df.Sequence.isin(amp_sequences)
        df = df.loc[~fns.values]
        logger.info(f"{fns.sum()} AMP sequences removed from UniProt data.")

    df.columns = [data_utils.camel_to_snake_case(x) for x in df.columns]
    df["targets"] = [[] for _ in range(len(df))]
    df["target_groups"] = [[] for _ in range(len(df))]

    kind = kind.lower()
    if kind == "reviewed":
        out_df = df[df.Status == "reviewed"]
    elif kind == "unreviewed":
        out_df = df[df.Status == "unreviewed"]
    elif kind == "all":
        out_df = df
    else:
        raise ValueError(
            f'Received invalid value kind={kind}, expected one of {{"all", "reviewed", "unreviewed"}}.'
        )
    
    # add random sampling logic
    if len(out_df) > sample_size:
        out_df = out_df.sample(n=sample_size, random_state=1)
        logger.info(f"Randomly sampled {sample_size} samples from the data.")
    
    # save the sampled data to a CSV file
    out_df.to_csv(save_path, index=False)
    logger.info(f"Sampled data saved to {save_path}.")


    logger.info(f"{len(out_df.index)} samples after curating UniProt.")
    return out_df


def make_condition_vectors(df: pd.DataFrame, max_seq_len: int = 50) -> np.ndarray:
    with open("results\species_values.json") as f:
        target_groups_values = json.load(f)
    with open("D:\\reoccur\\CFGAN\\results\\mic_bins.json") as f:
        mic_bins = json.load(f)

    target_groups_value = np.zeros((len(df), len(target_groups_values)))
    mic = len(mic_bins) - 2 + np.zeros(len(df), dtype=int)
    mic = data_utils.to_categorical(mic, num_classes=len(mic_bins) - 1)

    lengths = data_utils.int_col_2_bin_mask(df.length.values, max_len=max_seq_len)

    conditions = np.concatenate(
        [target_groups_value, mic, lengths], axis=-1,
    )

    logger.info(f"Uniprot Target Groups Shape:    {target_groups_value.shape}")
    logger.info(f"Uniprot MIC Shape:           {mic.shape}")
    logger.info(f"Uniprot Sequence Lengths Shape: {lengths.shape}")
    logger.info(f"Uniprot Condition Shape:        {conditions.shape}")
    return conditions


if __name__ == "__main__":
    prepare()

"""
Methods to load and clean the AVPdb for use in training machine learning models.
More information about the AVPdb can be found at http://crdd.osdd.net/servers/avpdb/index.php
"""

import json
import logging
from pathlib import Path

import data_utils
import pandas as pd

logger = logging.getLogger(__name__)


def prepare(max_seq_len: int = 32):
    df = load_data(max_seq_len=max_seq_len)
    logger.info(f"AVPdb: {len(df.index)} samples after cleaning.")
    logger.info("AVPdb: Calculating sequence symbol frequencies.")
    data_utils.calculate_symbol_frequencies(
        df["sequence"], dump_path="../data/avpdb/symbol_frequencies.json",
    )


def load_data(
    drop_duplicates: bool = True,
    load_path=Path("D:\\reoccur\\CFGAN\\data\\avpdb"),
    max_seq_len: int = 32,
    mode: str = "natural",
) -> pd.DataFrame:
    """
    If `mode` == "modified" then the sequences may contain non-FASTA symbols.

    Args:
        drop_duplicates: Drops rows with identical sequences.
        load_path: Location of the AVPdb data.
        max_seq_len: Sequences longer than this are filtered.
        mode: Sequence category to load. Options are {"natural", "modified", "both"}.

    Returns: Sequences from AVPdb.
    """
    assert (max_seq_len > 0) or (max_seq_len is None)

    mode = mode.lower()
    if mode == "natural":
        df = pd.read_csv(load_path / "AVPdb_data.tsv", sep="\t")
        df.rename(columns=data_utils.camel_to_snake_case, inplace=True)
    elif mode == "modified":
        df = pd.read_csv(load_path / "mAVPdb_data.tsv", sep="\t")
        df.rename(columns=data_utils.camel_to_snake_case, inplace=True)
    elif mode == "both":
        df1 = pd.read_csv(load_path / "AVPdb_data.tsv", sep="\t")
        df2 = pd.read_csv(load_path / "mAVPdb_data.tsv", sep="\t")
        df1.rename(columns=data_utils.camel_to_snake_case, inplace=True)
        df2.rename(columns=data_utils.camel_to_snake_case, inplace=True)
        df = pd.concat([df1, df2])
    else:
        raise ValueError(
            f"Expected mode to be one of {{'natural', 'modified', 'both'}}, received {mode}."
        )

    if max_seq_len is not None:
        df = df[df.sequence.str.len() <= max_seq_len]

    if drop_duplicates:
        df = df.drop_duplicates("sequence")

    # Try to provide the same interface as DBAASP
    df.rename(columns={"inhibition_ic50": "mic50", "target": "targets"}, inplace=True)

    df["target_groups"] = [["Virus"] for _ in range(len(df))]

    with open(load_path / "targets_mapping.json") as f:
        targets_map = json.load(f)
    df["targets"] = df["targets"].apply(lambda x: [targets_map[x]])

    # Filter rows that do not have an easily used IC50 value
    df = df[
        ~df["mic50"].str.contains("ND")
        & ~df["mic50"].str.contains("Nil")
        & ~df["mic50"].str.contains("No")
        & ~df["mic50"].str.contains("Low")
        & ~df["mic50"].str.contains("Medium")
        & ~df["mic50"].str.contains("High")
    ]

    # Drop variance info, we'll just use the mean
    df["mic50"] = df["mic50"].str.strip("><").str.split("±").str[0]

    def handle_sci_not(val):
        if "^" in str(val):
            base = float(val.split("x")[0]) if "x" in val else 1.0
            exp = float(val.split("^")[-1])
            return base * 10 ** exp
        if "e" in str(val):
            base, exp = [float(x.strip()) for x in val.split("e")]
            return base * 10 ** exp
        return val

    df["mic50"] = df["mic50"].apply(handle_sci_not)

    def handle_range(x):
        if isinstance(x, float):
            return x
        if "-" in str(x) and str(x).index("-"):
            low, high = [float(y.strip()) for y in x.split("-")]
            return (low + high) / 2
        if "to" in str(x):
            low, high = [float(y.strip()) for y in x.split("to")]
            return (low + high) / 2
        return x

    df["mic50"] = df["mic50"].apply(handle_range)

    # Clean up the values and cast to float
    df = df[
        ~df["sequence"].isna()
        & ~df["mic50"].isna()
        & ~df["length"].isna()
        & ~df["targets"].isna()
        & ~df["target_groups"].isna()
    ]
    df["mic50"] = df["mic50"].str.replace(",", "").astype(float)

    # Easy unit conversions
    df.loc[df["unit"].str.contains("nM"), "mic50"] /= 10.0 ** 3
    df.loc[df["unit"].str.contains("nM"), "unit"] = "\u03bcM"
    df.loc[df["unit"].str.contains("pM"), "mic50"] /= 10.0 ** 6
    df.loc[df["unit"].str.contains("pM"), "unit"] = "\u03bcM"

    # Filter rows that do not have an easily converted IC50 unit
    df = df[df.unit.str.contains("\u03bcM") | df.unit.str.contains("\u03bcg/ml")]

    # Convert everything to µg/ml for consistency
    df.loc[df["unit"].str.contains("µM"), "mic50"] = df.loc[
        df["unit"].str.contains("µM"), ["sequence", "mic50"]
    ].apply(data_utils.uM_to_ug_per_ml)

    df.dropna(inplace=True)

    
    logger.info(f"{len(df.index)} samples after curation of AVPdb.")
    return df


if __name__ == "__main__":
    prepare()

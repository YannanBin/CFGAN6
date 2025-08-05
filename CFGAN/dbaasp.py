"""
Methods to load and clean the DBAASP data for use in training machine learning models.
More information about the DBAASP database can be found at https://dbaasp.org/
"""

import json
import logging
import re
from pathlib import Path

import data_utils
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare():
    with open("D:\\reoccur\\CFGAN\\data\\dbaasp\\raw.json", encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Total number of entries: {len(data)}")

    logger.info("Identifying poorly formatted entries...")
    errors = []
    for i, d in enumerate(data):
        try:
            d["peptideCard"]["complexity"]
        except KeyError:
            errors.append(i)
            logger.warning(
                f"\tEntry {i} is poorly formatted:\n{json.dumps(d, sort_keys=True, indent=4)}\n"
            )

    logger.info("Filtering poorly formatted entries...")
    for error in errors:
        data = data[:error] + data[error + 1 :]
    logger.info(f"Entries after filtering: {len(data)}")

    logger.info("Filtering entries with missing values...")
    pre_val_filter = len(data)
    key2vals = {x: [] for x in ["id", "seq", "complexity", "targetGroups", "targets"]}
    mic50 = []
    skip = False
    for d in data:
        for key in key2vals.keys():
            if key not in d["peptideCard"].keys():
                skip = True
        if skip:
            skip = False
            continue

        for key in key2vals.keys():
            key2vals[key].append(d["peptideCard"][key])

        seq = d["peptideCard"]["seq"].strip()
        if ("targetActivities" not in d["peptideCard"].keys()) or (seq != seq.upper()):
            mic50.append(float("nan"))
            continue

        activity_measures = []
        for target in d["peptideCard"]["targetActivities"]:
            if "activityMeasure" not in target.keys():
                continue
            else:
                activity_measures.append(target["activityMeasure"])

        concentration_value_re = re.compile(
            "<?<?>?>?=?(\\d+(\\.\\d+)?)(\u00B1\\d+(\\.\\d+))?"
        )
        concentration_range_re = re.compile(r"(\d+(\.\d+)?)-(\d+(\.\d+)?)")
        activity = 0
        activity_count = 0
        for target in d["peptideCard"]["targetActivities"]:
            if ("activityMeasure" not in target.keys()) or (
                "concentration" not in target.keys()
            ):
                continue

            # Parse concentration
            conc_str = target["concentration"].strip()
            if target["activityMeasure"] == "MIC50":
                conc = float(concentration_value_re.match(conc_str).groups()[0])

            elif (target["activityMeasure"] == "MIC") and (
                "MIC50" not in activity_measures
            ):
                if concentration_value_re.match(conc_str):
                    conc = float(concentration_value_re.match(conc_str).groups()[0])

                elif concentration_range_re.match(conc_str):
                    result = concentration_range_re.match(conc_str)
                    conc = sum(float(x) for x in result.groups()) / 2.0

                else:
                    logger.warning(
                        f'Pattern not matched: {d["peptideCard"]["id"]}, {conc_str}'
                    )
                    continue
            else:
                continue

            # Convert units
            if target["unit"] == "µg/ml":
                pass
            elif target["unit"] == "µM":
                conc = data_utils.uM_to_ug_per_ml(conc, seq)
            else:
                logger.warning(
                    f'Expected units to be µM or µg/ml, received {target["unit"]}.\n\t'
                    f"Associated concentration: {conc_str}"
                )
                continue
            activity += conc
            activity_count += 1

        activity = activity / activity_count if activity_count else float("nan")
        mic50.append(activity)
    key2vals["mic50"] = mic50

    df = pd.DataFrame(key2vals)
    post_val_filter = len(df.index)
    logger.info(f"Removed {pre_val_filter - post_val_filter} rows with missing values.")

    pre_complexity_filter = len(df.index)
    df = df[df.complexity == "Monomer"]
    df.drop("complexity", axis=1, inplace=True)
    post_complexity_filter = len(df.index)
    logger.info(
        f"Removed {pre_complexity_filter - post_complexity_filter} rows not identified as Monomers."
    )

    pre_seq_filter = len(df.index)
    df = df[df.seq == df.seq.str.upper()]
    post_seq_filter = len(df.index)
    logger.info(
        f"Removed {pre_seq_filter - post_seq_filter} rows with sequences containing lower-case characters."
    )

    df["mic50"] = df["mic50"].round(6)

    df.dropna(inplace=True)
    logger.info(f"Final data shape: {df.shape}")

    df.rename(columns=data_utils.camel_to_snake_case, inplace=True)
    df.rename(columns={"seq": "sequence"}, inplace=True)

    df["length"] = df.sequence.str.len()

    np.set_printoptions(linewidth=10000)
    df.to_csv("D:\\reoccur\\CFGAN\\data\\dbaasp\\clean.csv", index=False)

    # Allows for better sampling of wildcards
    data_utils.calculate_symbol_frequencies(
        df["sequence"], dump_path="D:\\reoccur\\CFGAN\\data\\dbaasp\\symbol_frequencies.json",
    )


def load_data(
    drop_duplicates: bool = True,
    load_path: Path = Path("D:\\reoccur\\CFGAN\\data\\dbaasp\\clean.csv"),
    max_seq_len: int = 32,
) -> pd.DataFrame:
    assert (max_seq_len > 0) or (max_seq_len is None)

    df = pd.read_csv(load_path)

    if max_seq_len is not None:
        df = df[df["length"] <= max_seq_len]

    if drop_duplicates:
        df = df.drop_duplicates("sequence")

    df["target_groups"] = df["target_groups"].apply(eval)
    df["targets"] = df["targets"].apply(eval)

    with open(load_path.parent / "targets_mapping.json") as f:
        targets_map = json.load(f)
    df["targets"] = df["targets"].apply(lambda x: [targets_map[e] for e in x])


    logger.info(f"{len(df.index)} samples after curating DBAASP.")

    return df


if __name__ == "__main__":
    prepare()
    df = load_data()

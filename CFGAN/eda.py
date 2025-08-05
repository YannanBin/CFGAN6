import logging
import math
from collections import Counter

import avpdb
import dbaasp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uniprot

logger = logging.getLogger(__name__)


def main(max_seq_len: int = 32):
    sns.set_style("whitegrid")

    dbaasp_df = dbaasp.load_data()
    avpdb_df = avpdb.load_data()
    avp_start_len = len(avpdb_df)
    avpdb_df = avpdb_df.loc[~avpdb_df.sequence.isin(dbaasp_df.sequence), :]
    avp_end_len = len(avpdb_df)

    amp_df = pd.concat([dbaasp_df, avpdb_df])
    non_amp_df = uniprot.load_data()
    non_amp_start_len = len(non_amp_df)
    non_amp_df = non_amp_df.loc[~non_amp_df.sequence.isin(amp_df.sequence), :]
    non_amp_end_len = len(non_amp_df)

    logger.info(f"DBAASP:    {len(dbaasp_df)} sequences")
    logger.info(f"AVPdb:     {len(avpdb_df)} sequences")
    logger.info(f"UniProt: {len(non_amp_df)} sequences")
    logger.info(
        f"{avp_start_len - avp_end_len} sequences duplicated between DBAASP and AVPdb"
    )
    logger.info(f"{non_amp_start_len - non_amp_end_len} AMP sequences found in UniProt")

    amp_lengths = amp_df.sequence.str.len()
    amp_lengths.name = "AMP sequence lengths"
    non_amp_lengths = non_amp_df.sequence.str.len()
    non_amp_lengths.name = "Non-AMP sequence lengths"
    combined_lengths = pd.concat([amp_lengths, non_amp_lengths])
    combined_lengths.name = "Combined sequence lengths"

    logger.info(
        f"Number of Duplicate AMP Sequences: {len(amp_df.index) - len(amp_df.sequence.unique())}"
    )
    logger.info(
        f"Number of Duplicate Non-AMP Sequences: {len(non_amp_df.index) - len(non_amp_df.sequence.unique())}"
    )

    logger.info(f"\n{amp_lengths.describe()}")
    logger.info(f"\n{non_amp_lengths.describe()}")
    logger.info(f"\n{combined_lengths.describe()}")

    fig, ax = plt.subplots()
    ax.hist(
        [combined_lengths, amp_lengths, non_amp_lengths],
        density=True,
        bins=max_seq_len,
        range=(1, max_seq_len),
        label=[
            f"Combined ({len(combined_lengths)})",
            f"AMPs ({len(amp_lengths)})",
            f"Non-AMPs ({len(non_amp_lengths)})",
        ],
        color=["indigo", "darkorange", "darkgreen"],
        histtype="stepfilled",
        alpha=0.5,
    )
    plt.legend(loc="upper left")
    plt.xlabel("Sequence Length")
    plt.ylabel("Relative Frequency")
    plt.savefig("../results/sequence_length_summary.png")
    plt.close()

    target2count = Counter([x for y in amp_df.targets for x in y])
    target2count["Unknown"] = target2count.pop("")
    labels, counts = zip(
        *sorted(target2count.items(), key=lambda x: x[1], reverse=True)
    )
    fig, ax = plt.subplots()
    ax.bar(
        range(len(labels)), counts, tick_label=labels,
    )
    plt.xticks(rotation=45, ha="right", va="top")
    plt.ylabel("Count")
    plt.savefig("../results/global_target_counts.png", bbox_inches="tight")
    plt.close()

    target_group2count = Counter([x for y in amp_df.target_groups for x in y])
    labels, counts = zip(
        *sorted(target_group2count.items(), key=lambda x: x[1], reverse=True)
    )
    fig, ax = plt.subplots()
    ax.bar(
        range(len(labels)), counts, tick_label=labels,
    )
    plt.xticks(rotation=45, ha="right", va="top")
    plt.ylabel("Count")
    plt.savefig("../results/global_target_group_counts.png", bbox_inches="tight")
    plt.close()

    logger.info(
        f"Data points truncated from the MIC50 distribution to improve readability: {(amp_df.mic50 >= 2000).sum()}"
    )
    sns.displot(
        amp_df, x="mic50", rug=True,
    )
    plt.xlim(-50, 2000)
    plt.ylim(-50, 1300)
    plt.xlabel("MIC50")
    plt.ylabel("Count")
    plt.savefig("../results/mic50_dist.png", bbox_inches="tight")
    plt.close()

    # Sanity check on MIC50 bins
    ic50_classes, ic50_bins = pd.qcut(amp_df.mic50, 10, labels=False, retbins=True)
    logger.info(f"\nMIC50 Bins: {ic50_bins}")
    amp_df["MIC50 Class"] = ic50_classes
    plt.hist(amp_df["MIC50 Class"], bins=10)
    plt.savefig("../results/mic50_class_counts.png", bbox_inches="tight")

    e1 = amp_df.sequence.apply(entropy)
    e1.name = "AMP sequence entropy"
    logger.info(f"\n{e1.describe()}")

    e2 = non_amp_df.sequence.apply(entropy)
    e2.name = "Non-AMP sequence entropy"
    logger.info(f"\n{e2.describe()}")

    e3 = pd.concat([e1, e2]).describe()
    e3.name = "Combined sequence entropy"
    logger.info(f"\n{e3}")


def entropy(sequence, base=2) -> float:
    counts = Counter(sequence)
    total_count = sum(counts.values())
    probs = {k: v / total_count for k, v in counts.items()}
    return sum([-v * math.log(v, base) for v in probs.values()])


if __name__ == "__main__":
    logging.basicConfig(
        filename="../results/eda.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )

    main()

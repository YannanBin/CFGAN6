import functools
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import avpdb
import dbaasp
import numpy as np
import pandas as pd
import uniprot
from tensorflow import keras

logger = logging.getLogger(__name__)

# Molecular weights taken from:
#   https://en.wikipedia.org/wiki/Amino_acid#Table_of_standard_amino_acid_abbreviations_and_properties
# Maps FASTA symbols to their molecular weight in Daltons.
# Uses the average molecular weight for wildcard symbols that can represent several AAs.


symbol2weight: Dict[str, float] = {
    "A": 89.094,
    "C": 121.154,
    "D": 133.104,
    "E": 147.131,
    "F": 165.192,
    "G": 75.067,
    "H": 155.156,
    "I": 131.175,
    "K": 146.189,
    "L": 131.175,
    "M": 149.208,
    "N": 132.119,
    "O": 255.313,
    "P": 115.132,
    "Q": 146.146,
    "R": 174.203,
    "S": 105.093,
    "T": 119.119,
    "U": 168.064,
    "V": 117.148,
    "W": 204.228,
    "Y": 181.191,
}

wildcard2members: Dict[str, Tuple[str]] = {
    "B": ("D", "N"),
    "J": ("I", "L"),
    "X": tuple(symbol2weight.keys()),
    "Z": ("E", "Q"),
}

non_wildcard_symbols = list(symbol2weight.keys())
wildcard_symbols = list(wildcard2members.keys())

for wildcard, members in wildcard2members.items():
    symbol2weight[wildcard] = float(np.mean([symbol2weight[x] for x in members]))


def molecular_weight(seq: str) -> float:
    return sum(symbol2weight[symbol] for symbol in seq)


def uM_to_ug_per_ml(conc: float, seq: str) -> float:
    """
    Converts between micro-Moles per Liter (micro-Molar concentration) and
    micrograms per milliliter using the estimated molecular weight of an amino
    acid sequence.
    Estimated molecular weight is given in Daltons, or grams per mole.

    Dimensional Arithmetic:
        micro-moles per liter
            = 10**-6 mole / liter

        micro-grams per milliliter
            = 10**-6 gram / 10**-3 liter

        (micro-mole / liter) * (gram / mole) * 10**-3
            = (10**-6 mole / liter) * (gram / mole) * 10**-3
            = (10**-6 gram / liter) * 10**-3
            = (10**-6 gram / 10**3 * 1 liter)
            = micro-grams / milliliter

    Args:
        conc: float, Micro-molar concentration.
        seq: str, Amino acid sequence in FASTA format.

    Returns: float, concentration in micrograms per milliliter.
    """
    return conc * molecular_weight(seq) * 10 ** -3


def list_col_2_indicator(list_col, new_cols=None):
    new_cols = new_cols or sorted({v for vs in list_col for v in vs})
    col2ind = {col: i for col, i in zip(new_cols, np.arange(len(new_cols)))}
    indicators = np.zeros((len(list_col.index), len(new_cols)))
    for i, vs in enumerate(list_col):
        for v in vs:
            indicators[i, col2ind[v]] = 1
    return new_cols, indicators


def col_2_indicator(col):
    new_cols = sorted({v for v in col})
    col2ind = {col: i for col, i in zip(new_cols, np.arange(len(new_cols)))}
    indicators = np.zeros((len(col.index), len(new_cols)))
    for i, v in enumerate(col):
        indicators[i, col2ind[v]] = 1
    return new_cols, indicators


def str_col_2_indicator(col, max_len=None, chars=None):
    chars = chars or (non_wildcard_symbols + [" "])
    char2ind = {char_: i for char_, i in zip(chars, np.arange(len(chars)))}
    max_len = max_len or col.str.len().max()
    indicators = np.zeros((len(col.index), max_len, len(chars)))
    for i, str_ in enumerate(col):
        for j, char_ in enumerate(str_.ljust(max_len)):
            indicators[i, j, char2ind[char_]] = 1
    return chars, indicators


def str_col_2_fixed_vec(col, chars=None, max_len=None):
    chars = chars or (non_wildcard_symbols + [" "])
    char2ind = {char_: i for char_, i in zip(chars, np.arange(len(chars)))}
    max_len = max_len or col.str.len().max()
    indicators = np.zeros((len(col.index), max_len))
    for i, str_ in enumerate(col):
        if len(str_) > max_len:
            continue

        for j, char_ in enumerate(str_.ljust(max_len)):
            indicators[i, j] = char2ind[char_]
    return chars, indicators


def int_col_2_bin_mask(col, max_len=None):
    max_len = max_len or col.max()
    out_vals = np.zeros((len(col), max_len), dtype=np.float32)
    for i, val in enumerate(col):
        out_vals[i, :val] = 1
    return out_vals


def to_categorical(x: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Converts an N dimensional integer encoded numpy array into an N+1 dimensional
    one-hot encoded numpy array.

    Args:
        x: Array of labels to be one-hot encoded.
        num_classes: Number of expected classes in `x`.

    Returns: Array of one-hot encoded labels.
    """
    if num_classes is None:
        num_classes = np.max(x) + 1
    return np.eye(num_classes)[x]


def calculate_symbol_frequencies(
    sequences: List[str], relative: bool = True, dump_path: Optional[str] = None,
):
    symbol2frequency = {s: 0 for s in non_wildcard_symbols}
    for seq in sequences:
        for ch in seq:
            if ch in non_wildcard_symbols:
                symbol2frequency[ch] += 1

    if relative:
        symbol2frequency = normalize_values(symbol2frequency)

    if dump_path:
        with open(dump_path, "w") as f:
            json.dump(symbol2frequency, f, sort_keys=True, indent=4)

    return symbol2frequency


def normalize_values(dict_: Dict[str, int]) -> Dict[str, float]:
    total = sum(dict_.values())
    for key in dict_.keys():
        dict_[key] /= total
    return dict_


def sample_wildcards(
    seq: str, mode: str = "uniform", sampling_info: Optional[Dict[str, float]] = None,
) -> str:
    if sampling_info is None:
        sampling_info = get_wildcard_sampling_info(mode)
    chars = []
    for c in seq:
        if c in wildcard2members:
            options, weights = sampling_info[c]
            chars.append(np.random.choice(options, p=weights))
        else:
            chars.append(c)
    return "".join(chars)


def get_wildcard_sampling_info(
    mode: str = "uniform", frequencies: Optional[Dict[str, float]] = None,
):
    if frequencies is None:
        frequencies = get_symbol_frequency_map(mode)

    wildcard2sample_info = dict()
    for wildcard, members in wildcard2members.items():
        weights = np.array([frequencies[x] for x in members])
        weights /= weights.sum()
        wildcard2sample_info[wildcard] = (members, weights)
    return wildcard2sample_info


@functools.lru_cache()
def get_symbol_frequency_map(mode: str) -> Dict[str, float]:
    mode = mode.lower()
    if mode == "uniform":
        return {
            symbol: 1.0 / len(non_wildcard_symbols) for symbol in non_wildcard_symbols
        }
    elif mode == "dbaasp":
        with open("../data/dbaasp/symbol_frequencies.json") as f:
            return json.load(f)
    elif mode == "uniprot":
        with open("../data/uniprot/symbol_frequencies.json") as f:
            return json.load(f)
    elif mode == "combined":
        raise NotImplementedError()
    else:
        raise ValueError(
            f"{mode} is not a valid map. Expected one of {{uniform, dbaasp, uniprot, combined}}."
        )


def batch_generator(
    sequences,
    conditions,
    batch_size: int = 128,
    max_seq_len: int = 50,
    mode: str = "uniform",
    rescale_sequences: bool = True,
    calculate_frequencies: bool = False,
):
    assert len(sequences) == len(conditions)
    if calculate_frequencies:
        frequencies = calculate_symbol_frequencies(sequences)
        sampling_info = get_wildcard_sampling_info(frequencies=frequencies)
        sample_func = functools.partial(sample_wildcards, sampling_info=sampling_info)
    else:
        sample_func = functools.partial(sample_wildcards, mode=mode)
    while True:
        inds = np.random.permutation(len(sequences))
        batch_count = len(sequences) // batch_size
        for i in range(batch_count):
            batch_inds = inds[i * batch_size : (i + 1) * batch_size]
            seqs = sequences.iloc[batch_inds].apply(sample_func)
            conds = conditions[batch_inds]

            _, seqs = str_col_2_indicator(seqs, max_len=max_seq_len)

            if rescale_sequences:
                # [0, 1] -> [-1, 1]
                seqs = (seqs * 2) - 1
            yield seqs, conds


class AMPSequence(keras.utils.Sequence):
    def __init__(
        self,
        sequences,
        conditions,
        batch_size: int = 128,
        max_seq_len: int = 50,
        mode: str = "uniform",
        rescale_sequences: bool = True,
        calculate_frequencies: bool = False,
    ):
        assert len(sequences) == len(conditions)
        self.sequences = sequences
        self.conditions = conditions
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.rescale_sequences = rescale_sequences
        self.calculate_frequencies = calculate_frequencies

        if self.calculate_frequencies:
            frequencies = calculate_symbol_frequencies(sequences)
            sampling_info = get_wildcard_sampling_info(frequencies=frequencies)
            self.sample_func = functools.partial(
                sample_wildcards, sampling_info=sampling_info
            )
        else:
            self.sample_func = functools.partial(sample_wildcards, mode=self.mode)

    def __len__(self) -> int:
        extra = 1 if len(self.sequences) % self.batch_size else 0
        return len(self.sequences) // self.batch_size + extra

    def __getitem__(self, item):
        inds = slice(item * self.batch_size, (item + 1) * self.batch_size)
        seqs = self.sequences.iloc[inds].apply(self.sample_func)
        conds = self.conditions[inds]

        _, seqs = str_col_2_indicator(seqs, max_len=self.max_seq_len)

        if self.rescale_sequences:
            # [0, 1] -> [-1, 1]
            seqs = (seqs * 2) - 1
        return seqs, conds

    def on_epoch_end(self):
        inds = np.random.permutation(len(self.positive[0]))
        self.sequences = self.sequences.iloc[inds]
        self.conditions = self.conditions[inds]


class PairedAMPSequence(keras.utils.Sequence):
    def __init__(
        self,
        positive,
        negative,
        batch_size: int = 128,
        max_seq_len: int =50,
        mode: str = "uniform",
        rescale_sequences: bool = True,
        calculate_frequencies: bool = False,
    ):
        assert len(positive[0]) == len(positive[1])
        assert len(negative[0]) == len(negative[1])

        self.positive = positive
        self.negative = negative
        self.batch_size = batch_size // 2
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.rescale_sequences = rescale_sequences
        self.calculate_frequencies = calculate_frequencies

        if self.calculate_frequencies:
            frequencies = calculate_symbol_frequencies(self.positive[0])
            sampling_info = get_wildcard_sampling_info(frequencies=frequencies)
            self.pos_sample_func = functools.partial(
                sample_wildcards, sampling_info=sampling_info
            )
            frequencies = calculate_symbol_frequencies(self.negative[0])
            sampling_info = get_wildcard_sampling_info(frequencies=frequencies)
            self.neg_sample_func = functools.partial(
                sample_wildcards, sampling_info=sampling_info
            )
        else:
            self.pos_sample_func = functools.partial(sample_wildcards, mode=self.mode)
            self.neg_sample_func = self.pos_sample_func

    def __len__(self) -> int:
        min_len = min(len(self.positive[0]), len(self.negative[0]))
        return min_len // self.batch_size

    def __getitem__(self, item):
        inds = slice(item * self.batch_size, (item + 1) * self.batch_size)

        pos_seqs = self.positive[0].iloc[inds].apply(self.pos_sample_func)
        _, pos_seqs = str_col_2_indicator(pos_seqs, max_len=self.max_seq_len)
        pos_conds = self.positive[1][inds]

        neg_seqs = self.negative[0].iloc[inds].apply(self.neg_sample_func)
        _, neg_seqs = str_col_2_indicator(neg_seqs, max_len=self.max_seq_len)
        neg_conds = self.negative[1][inds]

        seqs = np.concatenate([pos_seqs, neg_seqs])
        conds = np.concatenate([pos_conds, neg_conds])

        if self.rescale_sequences:
            # [0, 1] -> [-1, 1]
            seqs = (seqs * 2) - 1
        return seqs, conds

    def on_epoch_end(self):
        pos_inds = np.random.permutation(len(self.positive[0]))
        self.positive = (self.positive[0].iloc[pos_inds], self.positive[1][pos_inds])

        neg_inds = np.random.permutation(len(self.negative[0]))
        self.negative = (self.negative[0].iloc[neg_inds], self.negative[1][neg_inds])


def decode_sequence(
    sequence: str, chars: Optional[List[str]] = None, concatenate: bool = False,
):
    chars = chars or (non_wildcard_symbols + [" "])
    ind2char = {i: c for (i, c) in zip(np.arange(len(chars)), chars)}
    decoded = "".join(ind2char[x] for x in np.argmax(sequence, axis=-1)).strip()
    if concatenate:
        decoded = decoded.replace(" ", "")
    else:
        decoded = decoded.split()[0] if " " in decoded else decoded
    return decoded


def decode_sequences(
    sequences: List[str], chars: Optional[List[str]] = None, concatenate: bool = False,
):
    return [
        decode_sequence(sequence, chars=chars, concatenate=concatenate)
        for sequence in sequences
    ]


def camel_to_snake_case(in_str: str) -> str:
    in_str = in_str.replace("_", "").replace("/", "")
    return re.sub("(?!^)([A-Z]+)", r"_\1", in_str).lower()


def sample_generators(gens, probs: Optional[np.ndarray] = None):
    if probs is None:
        probs = np.zeros(len(gens)) + 1.0
        probs /= probs.sum()

    assert len(gens) == len(probs)

    # Initialize the generators and make sure they all work
    for gen in gens:
        next(gen)

    while True:
        ind = np.random.choice(len(probs), p=probs)
        yield next(gens[ind])


def get_train_data(batch_size: int, des_path: str = "D:\\reoccur\\CFGAN\\data\\descriptor\\descriptor.tsv") -> PairedAMPSequence:

    amps = pd.read_csv("data\\amp\\filtered_data_merged.csv")    
    amps["target_groups"] = amps["target_groups"].apply(eval)

    # shuffle the data
    amps = amps.sample(frac=1).reset_index(drop=True)


    amp_conditions = make_condition_vectors(amps)  

    
    non_amps = uniprot.load_data(amp_sequences=amps.sequence)   
    non_amp_conditions = uniprot.make_condition_vectors(non_amps)   

    logger.info(f"{len(non_amps)} samples from UniProt.")
    logger.info(f"AMP Condition Shape: {amp_conditions.shape}")
    logger.info(f"Non-AMP Condition Shape: {non_amp_conditions.shape}")

    return PairedAMPSequence(
        (amps.sequence, amp_conditions),
        (non_amps.sequence, non_amp_conditions),
        batch_size=batch_size,
        max_seq_len=50,
        calculate_frequencies=True,
    )

def make_condition_vectors(
    df: pd.DataFrame, max_seq_len: int = 50, mic_quantiles: int = 5,
) -> pd.DataFrame:

    tg_vals, target_groups = list_col_2_indicator(df.target_groups)    #target_groups_shape:2327*181

    with open("D:\\reoccur\\CFGAN\\results\\species_values.json", "w") as f:
        json.dump(tg_vals, f, sort_keys=True, indent=4)

    mic, bins = pd.qcut(df.mic, mic_quantiles, retbins=True, labels=False)
    mic = to_categorical(mic.astype(int), num_classes=mic_quantiles)
    with open("D:\\reoccur\\CFGAN\\results\\mic_bins.json", "w") as f:
        json.dump(list(bins), f, sort_keys=True, indent=4)

    lengths = int_col_2_bin_mask(df["length"].values, max_len=max_seq_len)
    conditions = np.concatenate([target_groups, mic, lengths], axis=-1)

    logger.info(f"Target Groups Shape:     {target_groups.shape}")
    logger.info(f"MIC Shape:            {mic.shape}")
    logger.info(f"Sequence Lengths Shape:  {lengths.shape}")
    logger.info(f"Conditions Shape:        {conditions.shape}")
    return conditions

def gen_make_condition_vectors(
    df: pd.DataFrame, max_seq_len: int = 50, mic_quantiles: int = 5,
) -> pd.DataFrame:
    # read pre-calculated target_groups one-hot encoding mapping
    with open("D:\\reoccur\\CFGAN\\results\\species_values.json", "r") as f:
        tg_vals = json.load(f)

    # create target_groups one-hot encoding mapping
    col2ind_target_groups = {col: i for col, i in zip(tg_vals, np.arange(len(tg_vals)))}
    target_groups = np.zeros((len(df.target_groups.index), len(tg_vals)))

    for i, vs in enumerate(df.target_groups):
        for v in vs:
            if v in col2ind_target_groups:
                target_groups[i, col2ind_target_groups[v]] = 1

    # read pre-calculated MIC 50 one-hot encoding mapping
    with open("D:\\reoccur\\CFGAN\\results\\mic_bins.json", "r") as f:
        mic_bins = json.load(f)

    # map the MIC values to their corresponding bin index
    mic = np.digitize(df.mic, mic_bins[:-1]) - 1
    mic = to_categorical(mic.astype(int), num_classes=mic_quantiles)

    lengths = int_col_2_bin_mask(df["length"].values.astype(int), max_len=max_seq_len)

    conditions = np.concatenate([target_groups, mic, lengths], axis=-1)

    logger.info(f"Target Groups Shape:     {target_groups.shape}")
    logger.info(f"MIC 50 Shape:               {mic.shape}")
    logger.info(f"Sequence Lengths Shape:  {lengths.shape}")
    logger.info(f"Conditions Shape:        {conditions.shape}")
    return conditions

def decode_condition_vectors(
    conditions: np.ndarray,
    target_group_file: str = "D:\\reoccur\\CFGAN\\results\\species_values.json",
    mic_bin_file: str = "D:\\reoccur\\CFGAN\\results\\mic_bins.json",
):
    with open(target_group_file, mode="r") as f:
        target_group_values = json.load(f)
    ind2target_group = {
        i: g for (i, g) in zip(np.arange(len(target_group_values)), target_group_values)
    }
    with open(mic_bin_file, mode="r") as f:
        mic_bins = json.load(f)

    slice_ind_1 = len(target_group_values)
    slice_ind_2 = slice_ind_1 + len(mic_bins) - 1

    columns = {n: [] for n in ["target_groups", "mic", "length"]}

    for condition in conditions:
        indicator = condition[:slice_ind_1]
        inds = np.arange(len(indicator))[np.where(indicator)]
        target_groups = [ind2target_group[i] for i in inds]
        columns["target_groups"].append(target_groups)


        index = np.argmax(condition[slice_ind_1:slice_ind_2])
        left_edge = mic_bins[index]
        right_edge = mic_bins[index + 1]
        columns["mic"].append((left_edge + right_edge) / 2)

        columns["length"].append(np.sum(condition[slice_ind_2:]))
    return pd.DataFrame(columns)

def get_condition_option_counts(
    target_group_file: str = "D:\\reoccur\\CFGAN\\results\\species_values.json",
    mic_bin_file: str = "D:\\reoccur\\CFGAN\\results\\mic_bins.json",
    max_seq_len: int = 50,
) -> Tuple[int, int, int, int]:
    with open(target_group_file, mode="r") as f:
        target_group_values = json.load(f)
    with open(mic_bin_file, mode="r") as f:
        mic_bins = json.load(f)

    return (
        len(target_group_values),
        len(mic_bins) - 1,
        max_seq_len,
    )
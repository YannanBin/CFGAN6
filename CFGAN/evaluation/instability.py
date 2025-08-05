import numpy as np
from Bio import SeqIO
from modlamp.descriptors import GlobalDescriptor
from tqdm import tqdm

# define the set of 20 natural amino acids
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def instability_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.instability_index()
    score = desc.descriptor
    return score.squeeze()

def evaluate_fasta(file_path):
    check_function = instability_score

    fasta_list = list(SeqIO.parse(file_path, "fasta"))
    score_list = []
    for fasta in tqdm(fasta_list):
        fasta_seq = str(fasta.seq)
        # remove non-natural amino acids
        clean_seq = ''.join(filter(lambda x: x in AMINO_ACIDS, fasta_seq))
        if clean_seq:
            score = check_function(clean_seq)
            score_list.append((fasta.id, score))

    return score_list

# calculate the average score and print it to the terminal
def print_average_score(score_list):
    scores = [score for _, score in score_list]
    average_score = np.mean(scores)
    print(f"Average instability score: {average_score}")

# test
file_path = "results\generated_samples_trunc_2025-06-26_pos_samples.fasta"
scores = evaluate_fasta(file_path)
print_average_score(scores)



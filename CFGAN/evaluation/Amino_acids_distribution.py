import pandas as pd
import matplotlib.pyplot as plt

#  load data from csv files
amp_data = pd.read_csv('data\\amp\\discriminator_data.csv')
nonamp_data = pd.read_csv('data\\uniprot\\uniprot_data.csv')
generatedamp_data = pd.read_csv('results\generated_samples_trunc_pos_samples_RLFB2.0.csv')
#  define amino acid list
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# calculate the fraction of each amino acid in each group of data
def get_amino_acid_fraction(data):
    counts = [data.str.count(aa).sum() for aa in amino_acids]
    total = data.str.len().sum()
    return [count / total for count in counts]

amp_fraction = get_amino_acid_fraction(amp_data['sequence'])
nonamp_fraction = get_amino_acid_fraction(nonamp_data['sequence'])
generatedamp_fraction = get_amino_acid_fraction(generatedamp_data['sequence'])

# draw the line chart
plt.figure(figsize=(10, 6))
plt.plot(amino_acids, amp_fraction, label='AMP', color='#2ca02c', linewidth=1)
plt.plot(amino_acids, nonamp_fraction, label='nonAMP', color='#ff7f0e', linewidth=1)
plt.plot(amino_acids, generatedamp_fraction, label='SampleAMP', color='#1f77b4', linewidth=1)

# set chart properties
plt.xlabel('Amino Acids', fontsize=14)
plt.ylabel('Fraction', fontsize=14)
plt.title('Amino Acid Composition Distribution', fontsize=14)
plt.legend(fontsize=12)

# show the chart
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

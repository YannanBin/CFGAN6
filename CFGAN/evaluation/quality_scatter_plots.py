import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

# define a function to load and calculate physical properties
def calculate_properties(file_path, label):
    # read the sequence data
    sequences = pd.read_csv(file_path)['sequence'].dropna().astype(str).tolist()
    # filter valid amino acids
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    filtered_sequences = [''.join([aa for aa in seq if aa in valid_aa]) for seq in sequences if seq]

    # remove empty sequences
    filtered_sequences = [seq for seq in filtered_sequences if len(seq) > 0]
    # create GlobalDescriptor object and calculate properties
    global_desc = GlobalDescriptor(filtered_sequences)
    pep_desc = PeptideDescriptor(filtered_sequences)
    
    global_desc.length()
    length = global_desc.descriptor.flatten()

    pep_desc.calculate_moment()
    hydrophobic_moment = pep_desc.descriptor.flatten()  

    global_desc.isoelectric_point()
    isoelectric_point = global_desc.descriptor.flatten()



    global_desc.calculate_charge()
    charge = global_desc.descriptor.flatten()

    # create DataFrame and return
    data = pd.DataFrame({
        'Length': length,
        # 'Hydrophobicity': hydrophobicity,
        'Hydrophobic Moment': hydrophobic_moment,
        'Isoelectric Point': isoelectric_point,
        'Charge': charge,
        'Label': label  
    })
    return data

# calculate physical properties for AMP, NonAMP and SampleAMP respectively
amp_data = calculate_properties('data/amp/discriminator_data.csv', 'AMP').sample(n=1000, random_state=6)
non_amp_data = calculate_properties('data/neg_data.csv', 'NonAMP').sample(n=1000, random_state=6)
sample_data = calculate_properties('results/generated_samples_trunc_214_pos_sample.csv', 'ProtoAMP').sample(n=1000, random_state=6)
# merge three data groups
data = pd.concat([amp_data, non_amp_data, sample_data], ignore_index=True).dropna()

# set the style and draw the scatter matrix plot
sns.set(style="whitegrid")

pairplot = sns.pairplot(
    data, 
    vars=['Length', 'Hydrophobic Moment', 'Isoelectric Point', 'Charge'], 
    hue='Label', 
    palette={"NonAMP": "#D8975E", "AMP": "#A92E08", "ProtoAMP": "#515792"}, 
    markers=["o", "o", "o"], 
    plot_kws={'alpha': 0.6, 's': 20}
)

# adjust label font size
for ax in pairplot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=14) 
    ax.set_ylabel(ax.get_ylabel(), fontsize=14) 
# get the legend and set the font size
legend = pairplot._legend
for text in legend.get_texts():
    text.set_fontsize(14)  

# save chart as PNG file
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight') 

# show the chart
plt.show()

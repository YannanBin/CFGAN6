import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Model': ['CFGAN_RF2.0', 'CFGAN_RF1.0', 'CFGAN_RF0.5', 'CGAN', 'CFGAN', 'LSTMRNN', 'HydrAMP', 'DiffAMP'],
    '3- tags': [348, 359, 431, 480, 509, 438, 471, 420],
    '3 tags': [360, 348, 349, 326, 328, 301, 357, 335],
    '4 tags': [196, 202, 171, 140, 133, 173, 135, 153],
    '5 tags': [70, 64, 37, 40, 25, 62, 28, 52],
    '5+ tags': [26, 27, 12, 14, 5, 26, 9, 40],
}

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

models = ['CFGAN_RFM2.0', 'CFGAN_RFM1.0', 'CFGAN_RFM0.5', 'CGAN', 'CFGAN', 'LSTMRNN', 'HydrAMP', 'DiffAMP']
labels = ['0 targets', '1-2 targets', '3 targets', '4 targets', '>= 5 targets']

data = [
    [3, 345, 360, 196, 96],
    [2, 357, 348, 202, 91],
    [7 ,424, 349, 171, 49],
    [7, 473, 326, 140, 54],
    [8, 501, 328, 133, 30],
    [9, 429, 301, 173, 88],
    [7, 464, 357, 135, 37],
    [6, 414, 335, 153, 92],
]

df = pd.DataFrame(data, index=models, columns=labels)


plt.figure(figsize=(12, 6))
sns.heatmap(df, annot=True, fmt='d', cmap="YlGnBu",cbar_kws={'label': 'Number'}, annot_kws={"size": 14})


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)  

plt.tight_layout()
plt.show()


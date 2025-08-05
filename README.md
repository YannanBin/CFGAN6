# CFGAN
**Generation of broad-spectrum antimicrobial peptides based on conditional feedback generation adversarial network**

### Overview

CFGAN is a deep learning framework designed to generate broad-spectrum antimicrobial peptides (AMPs) with high efficiency and functional specificity. By incorporating a Conditional Generative Adversarial Network (CGAN) with a reinforcement feedback loop mechanism and Brownian Motion Controller (BMC), CFGAN aims to improve peptide broad-spectrum, optimize antimicrobial activity, and ensure stability during the training process.

<img width="1132" height="917" alt="image" src="https://github.com/user-attachments/assets/f378e1c4-7ea5-436f-a6fa-3c1c1de6ea7b" />

This repository includes the necessary scripts and tools for training the model, evaluating its performance, and generating antimicrobial peptides.	

### Environment

CFGAN is built using Python and Tensorflow.
We recommend the use of [Anaconda](https://www.anaconda.com) to manage your Python environment.

Additionally, we use [Git Large File Storage](https://git-lfs.github.com/) (LFS) to manage data and model checkpoints.
Ensure that you have it installed before cloning this repository.

Clone the repository:
```bash
git clone https://github.com/wrab12/diff-amp
cd CFGAN
```

Create and activate a virtual environment using python 3.9 with `virtualenv` or `conda`,

```
conda create -n CFGAN python=3.7
conda activate CFGAN
```

Install dependencies and the local library with `pip`.

```
pip install -r requirements.txt
```

### Run code

Train our model: Use `train.py`. Download the weights of analyzer from the [Google Drive link](https://drive.google.com/drive/folders/1eZBOEUoiHwWZvu7rORdwojzGAnpcS_iV?dmr=1&ec=wgc-drive-hero-gotoand) weights folder.  place them in the `weights` directory.

Generate antimicrobial peptides: To generate samples using our pre-trained weights, download the weights from the [Google Drive link](https://drive.google.com/drive/folders/1eZBOEUoiHwWZvu7rORdwojzGAnpcS_iV?dmr=1&ec=wgc-drive-hero-gotoand)  models folder and run the following command in the terminal:

```
python ampgan\generate_samples.py -m models\amp_gan_2025-06-14\gan_0348
```

### Contents

| Name           | Description |
| ---            | --- |
| CFGAN/        | Primary source code folder. |
| data/          | Contains all raw data used to construct CFGAN training sets. |
| models/        | Contains the weights of a trained CFGANv2 model. |
| results/       | All code outputs will be saved here. |
| scripts/       | Utility scripts, mostly for data or experiment management. |
| .gitattributes | Controls Git LFS behavior. |
| .gitignore     | Determines which files are considered by Git |
| LICENSE        | MIT license.                                                 |
| README.md      | This file.                                                   |

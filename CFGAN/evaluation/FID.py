# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from numpy import cov
from scipy import linalg
import CFGAN.utils.esm2_feature as esm

import os
import random




def time_format():
    return f"{datetime.now()}|> "


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def calculate_fid(act1, act2):

    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # real dataset loading
    dataset_real = pd.read_csv('data\\amp\\discriminator_data.csv',na_filter = False)
    dataset_real = dataset_real.dropna(subset=['sequence']).sample(n=1000, random_state=1)  
    sequence_list_real = dataset_real['sequence']
    

    # generated dataset loading
    dataset_gen = pd.read_csv('results\generated_samples_trunc_pos_samples_RFM2.0.csv',na_filter = False)
    dataset_gen = dataset_gen.dropna(subset=['sequence']).sample(n=1000, random_state=1)  
    sequence_list_gen = dataset_gen['sequence'] 
    

    gen_embeddings_results = pd.DataFrame()
    real_embeddings_results = pd.DataFrame()
    for seq in sequence_list_real:
        format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
        # employ ESM model for converting and save the converted data in csv format
        one_seq_embeddings = esm.esm_embeddings(peptide_sequence_list)
        real_embeddings_results= pd.concat([real_embeddings_results,one_seq_embeddings])
    
    for seq in sequence_list_gen:
        format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
        # employ ESM model for converting and save the converted data in csv format
        one_seq_embeddings = esm.esm_embeddings(peptide_sequence_list)
        gen_embeddings_results= pd.concat([gen_embeddings_results,one_seq_embeddings])

    act1 = real_embeddings_results.to_numpy()
    act2 = gen_embeddings_results.to_numpy()

    fid = calculate_fid(act1, act2)
    print(f"FID: {fid}")

    


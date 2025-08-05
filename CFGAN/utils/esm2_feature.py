import torch
import esm
import collections
import pandas as pd
import numpy as np

def esm_embeddings(peptide_sequence_list):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long, 
  #         or you have too many sequences for transformation in a single converting, 
  #         you conputer might automatically kill the job.
  
  # load the model
  # NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
  model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  model.eval()  # disables dropout for deterministic results

  # load the peptide sequence list into the bach_converter
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
  ## batch tokens are the embedding results of the whole data set

  # Extract per-residue representations (on CPU)
  with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
      results = model(batch_tokens, repr_layers=[6], return_contacts=True)  
  token_representations = results["representations"][6]

  # Generate per-sequence representations via averaging
  # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
  sequence_representations = []
  for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
  # save dataset
  # sequence_representations is a list and each element is a tensor
  embeddings_results = collections.defaultdict(list)
  for i in range(len(sequence_representations)):
      # tensor can be transformed as numpy sequence_representations[0].numpy() or sequence_representations[0].to_list
      each_seq_rep = sequence_representations[i].tolist()
      for each_element in each_seq_rep:
          embeddings_results[i].append(each_element)
  embeddings_results = pd.DataFrame(embeddings_results).T
  return embeddings_results

if __name__ == '__main__':
    # training dataset loading
    dataset = pd.read_csv('results\generated_samples_trunc_2024-08-09.csv',na_filter = False) # take care the NA sequence problem
    sequence_list = dataset['sequence'] 

    if len(sequence_list) > 1000:
        sequemce_list = sequence_list.sample(n=1000, random_state=1)

    embeddings_results = pd.DataFrame()
    for seq in sequence_list:
        format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
        # employ ESM model for converting and save the converted data in csv format
        one_seq_embeddings = esm_embeddings(peptide_sequence_list)
        embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])

    embeddings_results.to_csv('gen_samples_esm2_feature_8_09.csv')

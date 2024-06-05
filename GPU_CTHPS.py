from datasets import load_metric
import transformers
import pandas as pd
from functools import lru_cache
from transformers import AutoTokenizer
import nltk
from transformers import AutoModelForSeq2SeqLM
import wandb
import torch
from tqdm import tqdm
from evaluate import load
import nltk
from sacremoses import MosesTokenizer
tqdm.pandas()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

wandb.login(key = '8f14c161d9bdaea0c585153d80f5e180a35c2e71')

def bin_round(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

special_tokens = ["Simplify: "]
special_tags = ["WLR","CLR", "LR", "WRR", "DTDR","ICCR","DTRL"]
for t in special_tags:
    for i in range(41):
        special_tokens.append(t+"_"+str(round(i*0.05, 2)))

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens = special_tokens)
max_target_length = 128

mTokenizer = MosesTokenizer(lang='nl')


def get_special_token(type, value, special_tokens = tokenizer.all_special_tokens):
  for token in special_tokens:
    if token.startswith(type) and token.endswith(value):
      return token

asset_val_dataset = pd.read_pickle('plank_dataset_tokenized_7_CT_val.pkl')
asset_val_dataset = asset_val_dataset.iloc[:int(len(asset_val_dataset)/100)]
dataset_parts = {"val":asset_val_dataset}

class CustomTrainDataset(torch.utils.data.Dataset):
    """Dataset class that allows easy requests of a data row."""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return {
            "input_ids":row["Complex_token_input_ids"],
            "attention_mask":row["Complex_token_attention_masks"],
            "labels":row["Simple_token_input_ids"]
        }
val_dataset = CustomTrainDataset(dataset_parts["val"])

bert_score_metric = load("bertscore")
decoded_preds = ["de koe staat in de wei"]
decoded_labels = ["de koe staat niet in de wei"]
bert_score_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="nl")

import sys
sys.path.append('Packages')
import easse.utils as utils_prep
from easse.sari import corpus_sari,get_corpus_sari_operation_scores

bert_score_metric = load("bertscore")
decoded_preds = ["de koe staat in de wei"]
decoded_labels = ["de koe staat niet in de wei"]
bert_score_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="nl")

import sys
sys.path.append('Packages')
import easse.utils as utils_prep
from easse.sari import corpus_sari,get_corpus_sari_operation_scores
from easse.fkgl import corpus_fkgl

# from bleurt import score
# checkpoint = "BLEURT-20"
# bleurt_score_metric = score.BleurtScorer(checkpoint)

chrf_score_metric= load("chrf")
#meteor_score_metric = load('meteor')
# def compute_meteor(ref,pred):
#     return {'meteor':meteor_score_metric.compute(predictions=pred, references=ref)['meteor']}
def compute_chrf(ref, pred):
    return {'chrf': chrf_score_metric.compute(predictions=pred, references=ref)['score']}
def compute_sari(src, ref, pred):
    refs_corpus_sari_format = [ref]

    sari_score = corpus_sari(
            orig_sents=src,
            sys_sents= pred,
            refs_sents=refs_corpus_sari_format
    )
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(src, pred, refs_corpus_sari_format)

    results = {
      "SARI" : sari_score,
      "SARI_ADD" : add_score,
      "SARI_KEEP" : keep_score,
      "SARI_DEL" : del_score
    }
    return results

def compute_bert_score(ref, pred):
    results = bert_score_metric.compute(predictions=pred, references=ref, lang="nl")
    del results["hashcode"]
    for key in results:
        results[key] = sum(results[key])/len(results[key])
    return {"bert-score":results["f1"]}

def compute_fkgl(pred):
  return {"fkgl":corpus_fkgl(pred)}

# def compute_bleurt_score(ref, pred):
#     results = bleurt_score_metric.score(references=ref, candidates=pred)
#     return {"bleurt":sum(results)/len(results)}

def compute_custom(sari, bert_score, fkgl,chrf):
    fkgl_remapped = (100-(fkgl*(100/18)))/100
    return {"custom":(float(sari)/100+float(chrf)/100+bert_score+fkgl_remapped)/4}

@lru_cache(maxsize=1)
def get_model_config():
  return transformers.GenerationConfig(
      max_new_tokens=64,
      min_new_tokens=3,
      do_sample=False,
      num_beams=5,
      num_beam_groups=5,
      temperature=0,
      repetition_penalty=1.5,
      diversity_penalty=0.1
  )

def init_model_from_path(path):
    model =  AutoModelForSeq2SeqLM.from_pretrained(path, config=get_model_config(), device_map = 'auto')
    model.resize_token_embeddings(len(tokenizer))
    return model

best_model = init_model_from_path("final_models/t5-dutch-simplify-CT7")

def simplify_df(df, model, tokenizer, model_config=get_model_config()):

    outputs = []
    batch_size = 64  # Adjust the batch size as needed
    for i in range(0, len(df), batch_size):
        print(i, len(df))
        batch_df = df.iloc[i:i + batch_size]

        # Extract the columns as lists
        input_ids_list = batch_df['Complex_token_input_ids'].tolist()
        attention_masks_list = batch_df['Complex_token_attention_masks'].tolist()

        # Convert lists of lists to tensors
        input_ids_tensors = [torch.tensor(ids) for ids in input_ids_list]
        attention_masks_tensors = [torch.tensor(masks) for masks in attention_masks_list]

        # Pad sequences to ensure they have the same length
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_tensors, batch_first=True,
                                                           padding_value=tokenizer.pad_token_id)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks_tensors, batch_first=True,
                                                                 padding_value=0)

        # Move to the appropriate device (e.g., GPU if available)
        input_ids_padded = input_ids_padded.to(model.device)
        attention_masks_padded = attention_masks_padded.to(model.device)

        # Generate outputs for the batch
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids_padded, attention_mask=attention_masks_padded,
                                        generation_config=model_config)

        # Decode the generated output IDs
        output_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs.extend(output_sentences)

        # Clean up tensors and intermediate variables
        del input_ids_padded
        del attention_masks_padded
        del output_ids
        del batch_df
        del output_sentences

        # Explicit garbage collection
    torch.cuda.empty_cache()

    return outputs

def modify(row, ctokens):

  control_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ctokens]
  if type(row["Complex_token_input_ids"]) == list:
    row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1] + control_token_ids + row["Complex_token_input_ids"][8:])
  else:
    row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1].tolist() + control_token_ids + row["Complex_token_input_ids"][8:].tolist())

  row["Complex_token_attention_masks"] = torch.ones(len(row["Complex_token_input_ids"]), dtype=torch.int)

  return row

def calc_metrics_from_sentence_pairs(sources, decoded_labels, decoded_preds):

  sari_scores = compute_sari(sources, decoded_labels, decoded_preds)
  bert_scores = compute_bert_score(decoded_labels, decoded_preds)
  #bleurt_scores = compute_bleurt_score(decoded_labels, decoded_preds)
  fkgl_score = compute_fkgl(decoded_preds)
  chrf_score = compute_chrf(decoded_labels, decoded_preds)
  #meteor_score = compute_meteor(decoded_labels, decoded_preds)
  custom_score = compute_custom(sari_scores["SARI"],
                                bert_scores["bert-score"],
                                #bleurt_scores["bleurt"],
                                fkgl_score["fkgl"],
                                chrf_score["chrf"])
                                #meteor_score["meteor"])
  return sari_scores | bert_scores | fkgl_score | chrf_score | custom_score#| bleurt_scores

import gc
import random
def hyper_parameter_run(config=None):
    torch.cuda.empty_cache()
    gc.collect()
    # Initialize a new wandb run
    print(config)
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        tokens = [get_special_token("WLR",str(float(config.WLR))),
                  get_special_token("CLR",str(float(config.CLR))),
                  get_special_token("LR",str(float(config.LR))),
                  get_special_token("WRR",str(float(config.WRR))),
                  get_special_token("DTDR",str(float(config.DTDR))),
                  get_special_token("ICCR",str(float(config.ICCR))),
                  get_special_token("DTRL",str(float(config.DTRL)))]

        random_num_tokens = random.randint(1, 7)
        sampled_indices = sorted(random.sample(range(len(tokens)), random_num_tokens))
        tokens = [tokens[i] for i in sampled_indices]

        modified_dataset = dataset_parts["val"].copy().apply(lambda row: modify(row, tokens), axis=1)

        result = simplify_df(modified_dataset, best_model, tokenizer)

        actual_tokens = {
            "actual_WLR": 0,
            "actual_CLR": 0,
            "actual_LR": 0,
            "actual_WRR": 0,
            "actual_DTDR": 0,
            "actual_ICCR": 0,
            "actual_DTRL": 0,
        }
        if 0 in sampled_indices:
            actual_tokens["actual_WLR"] = tokens[sampled_indices.index(0)]
        if 1 in sampled_indices:
            actual_tokens["actual_CLR"] = tokens[sampled_indices.index(1)]
        if 2 in sampled_indices:
            actual_tokens["actual_LR"] = tokens[sampled_indices.index(2)]
        if 3 in sampled_indices:
            actual_tokens["actual_WRR"] = tokens[sampled_indices.index(3)]
        if 4 in sampled_indices:
            actual_tokens["actual_DTDR"] = tokens[sampled_indices.index(4)]
        if 5 in sampled_indices:
            actual_tokens["actual_ICCR"] = tokens[sampled_indices.index(5)]
        if 6 in sampled_indices:
            actual_tokens["actual_DTRL"] = tokens[sampled_indices.index(6)]

        del modified_dataset
        scores = calc_metrics_from_sentence_pairs(dataset_parts["val"]["Complex"].tolist(), dataset_parts["val"]["Simple"].tolist(), result) | actual_tokens
        del result
        wandb.log(scores)

sweep_config = {
  'method': 'random'
}
import numpy as np
parameters_dict = {
    'WLR': {"values":[bin_round(value) for value in np.arange(0.2, 2.05, 0.05)]},
    'CLR': {"values":[bin_round(value) for value in np.arange(0.2, 2.05, 0.05)]},
    'LR': {"values":[bin_round(value) for value in np.arange(0.15, 1.05, 0.05)]},
    'WRR': {"values":[bin_round(value) for value in np.arange(0.2, 1.25, 0.05)]},
    'DTDR': {"values":[0.2,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.75,0.8,0.85,1,1.2,1.25,1.3,1.35,1.5,1.65,1.95,2.0]},
    'ICCR': {"values":[bin_round(value) for value in np.arange(0.1, 1.05, 0.05)]},
    'DTRL': {"values":[bin_round(value) for value in np.arange(0.2, 2.05, 0.05)]}
}
sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="huggingface")
wandb.agent(sweep_id, hyper_parameter_run, count=10000,project="huggingface")
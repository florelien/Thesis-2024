#from datasets import load_metric
import transformers
import pandas as pd
from functools import lru_cache
from transformers import AutoTokenizer
import nltk
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM
import wandb
import torch
from tqdm import tqdm
from evaluate import load
import nltk
from sacremoses import  MosesTokenizer
tqdm.pandas()
import gc
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
#
wandb.login(key = '8f14c161d9bdaea0c585153d80f5e180a35c2e71')
wandb.init(project="huggingface", name="eval_new_method_10ct_ft_model")

special_tokens = [] #"Simplify: " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
special_tags = ["WLR","CLR", "LR", "WRR", "DTDR", "ICCR", "DTRL", "NSC", "PRC", "PPEN"]
for t in special_tags:
    for i in range(41):
        special_tokens.append(t+"_"+str(round(i*0.05, 2)))

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens = special_tokens)
max_target_length = 128

mTokenizer = MosesTokenizer(lang='nl')

test = pd.read_pickle("plank_dataset_tokenized_10_CT_bert_pred_test.pkl")
print(len(test))
print(test.columns)
print(test.iloc[1]["Complex_token_input_ids"])
print(tokenizer.convert_ids_to_tokens(test.iloc[1]["Complex_token_input_ids"]))
print(test.iloc[1]["pred_ct"])


def bin_round(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

def get_special_token(type, value, special_tokens = tokenizer.all_special_tokens):
  for token in special_tokens:
    if token.startswith(type) and token.endswith(value):
      return token

def process(row):
    options = ["WLR","CLR", "LR", "WRR", "DTDR","ICCR","DTRL","NSC","PRC","PPEN"]
    special_toks = [ get_special_token(option,str(min(float(bin_round(num)),2.0))) for option, num in zip(options, row["pred_ct"])]
    special_toks_ids = tokenizer.convert_tokens_to_ids(special_toks)

    row["Complex_token_input_ids"] = torch.cat(
        [
            torch.tensor(row["Complex_token_input_ids"][:5]),
            torch.tensor(special_toks_ids),
            torch.tensor(row["Complex_token_input_ids"][15:])
        ]
    )
    return row["Complex_token_input_ids"]

test["Complex_token_input_ids"] = test.progress_apply(lambda row: process(row), axis=1)

# def concatenate_tokens(row):
#     # Convert control tokens to IDs
#     control_token_ids = tokenizer.convert_tokens_to_ids(row["control_tokens"])
#     # Concatenate parts of the token list
#     concatenated_tokens = torch.cat([torch.tensor(row["Complex_token_input_ids"][:5]),
#                                      torch.tensor(control_token_ids),
#                                      torch.tensor(row["Complex_token_input_ids"][5:])])
#     return concatenated_tokens

#test["Complex_token_input_ids"] = test.apply(concatenate_tokens, axis=1)
test["Complex_token_attention_masks"] = test["Complex_token_input_ids"].apply(lambda item: torch.ones(len(item), dtype=torch.int))
print(test.iloc[1]["Complex_token_input_ids"])
print(tokenizer.convert_ids_to_tokens(test.iloc[1]["Complex_token_input_ids"]))


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

from bleurt import score
checkpoint = "BLEURT-20"
# bleurt_score_metric = score.BleurtScorer(checkpoint)

chrf_score_metric= load("chrf")


def compute_chrf(ref, pred):
    return {'chrf': chrf_score_metric.compute(predictions=pred, references=ref)['score']}
def compute_sari(src, ref, pred):
    refs_corpus_sari_format = [ref]

    sari_score = corpus_sari(
        orig_sents=src,
        sys_sents=pred,
        refs_sents=refs_corpus_sari_format
    )
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(src, pred, refs_corpus_sari_format)

    results = {
        "SARI": sari_score,
        "SARI_ADD": add_score,
        "SARI_KEEP": keep_score,
        "SARI_DEL": del_score
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

def compute_bleurt_score(ref, pred):
    #ref = [r[0] for r in ref]
    #results = bleurt_score_metric.score(references=ref, candidates=pred)
    #return {"bleurt":sum(results)/len(results)}
    results = bleurt_score_metric.score(references=ref, candidates=pred)
    return {"bleurt":sum(results)/len(results)}

def compute_custom(sari, bert_score,bleurt, fkgl,chrf):
    fkgl_remapped = (100-(fkgl*(100/18)))/100
    return {"custom":(float(sari)/100+float(chrf)/100+bert_score+bleurt+fkgl_remapped)/5}


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
    model =  AutoModelForSeq2SeqLM.from_pretrained(path, config=get_model_config())
    model.resize_token_embeddings(len(tokenizer))
    return model



#CUDA_VISIBLE_DEVICES=0 python GPU_TestSet10CT.py
dataset_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 2000, 5000, 10000, 100000, 800000]
scores_list = []
for dataset_size in [0]:

    best_model = init_model_from_path("final_models/10_CT_2EP") #yhavinga/t5-base-dutch"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)

    def simplify_df(df, model, tokenizer, model_config=get_model_config()):

        outputs = []
        batch_size = 12  # Adjust the batch size as needed
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

    def calc_metrics_from_sentence_pairs(sources, decoded_labels, decoded_preds):

      sari_scores = compute_sari(sources, decoded_labels, decoded_preds)
      bert_scores = compute_bert_score(decoded_labels, decoded_preds)
      bleurt_scores = compute_bleurt_score(decoded_labels, decoded_preds)
      fkgl_score = compute_fkgl(decoded_preds)
      chrf_score = compute_chrf(decoded_labels, decoded_preds)
      custom_score = compute_custom(sari_scores["SARI"],
                                    bert_scores["bert-score"],
                                    bleurt_scores["bleurt"],
                                    fkgl_score["fkgl"],
                                    chrf_score["chrf"])
      return sari_scores | bert_scores | fkgl_score | chrf_score | custom_score| bleurt_scores

    def modify(row, ctokens):

      control_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ctokens]
      if type(row["Complex_token_input_ids"]) == list:
        row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1] + control_token_ids + row["Complex_token_input_ids"][8:])
      else:
        row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1].tolist() + control_token_ids + row["Complex_token_input_ids"][8:].tolist())

      row["Complex_token_attention_masks"] = torch.ones(len(row["Complex_token_input_ids"]), dtype=torch.int)

      return row


    #modified_dataset = test.copy().apply(lambda row: modify(row, ['CLR_1.7', 'DTDR_0.4', 'DTRL_0.6']), axis=1)#,
    #print(modified_dataset.iloc[1]["Complex_token_input_ids"])

    result = simplify_df(test, best_model, tokenizer)
    bleurt_score_metric = score.BleurtScorer(checkpoint)
    scores = calc_metrics_from_sentence_pairs(test["Complex"].tolist(), test["Simple"].tolist(), result)

    # del modified_dataset
    # del best_model
    # torch.cuda.empty_cache()
    # gc.collect()
    # print(scores)
    wandb.log(scores)
    # scores_list.append(scores)

exit(0)

# import pickle
# file = open('ds_size_results', 'wb')
# dump information to that file
# pickle.dump(scores_list, file)


#from datasets import load_metric
import transformers
import pandas as pd
from functools import lru_cache
from transformers import AutoTokenizer
import nltk
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM
import wandb
import torch
from tqdm import tqdm
from evaluate import load
import nltk
from sacremoses import  MosesTokenizer
tqdm.pandas()
import gc
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

wandb.login(key = '8f14c161d9bdaea0c585153d80f5e180a35c2e71')
wandb.init(project="huggingface", name="eval_CLR_1.7DTDR_0.4DTRL_0.6_7ct_ft_model")

special_tokens = ["Simplify: "]
special_tags = ["WLR","CLR", "LR", "WRR", "DTDR","ICCR","DTRL"]
for t in special_tags:
    for i in range(41):
        special_tokens.append(t+"_"+str(round(i*0.05, 2)))

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens = special_tokens)
max_target_length = 128

mTokenizer = MosesTokenizer(lang='nl')

test = pd.read_pickle("plank_dataset_tokenized_7_CT_test.pkl")

def concatenate_tokens(row):
    # Convert control tokens to IDs
    control_token_ids = tokenizer.convert_tokens_to_ids(row["control_tokens"])
    # Concatenate parts of the token list
    concatenated_tokens = torch.cat([torch.tensor(row["Complex_token_input_ids"][:5]),
                                     torch.tensor(control_token_ids),
                                     torch.tensor(row["Complex_token_input_ids"][5:])])
    return concatenated_tokens

#test["Complex_token_input_ids"] = test.apply(concatenate_tokens, axis=1)
test["Complex_token_attention_masks"] = test["Complex_token_input_ids"].apply(lambda item: torch.ones(len(item), dtype=torch.int))
print(test.iloc[1]["Complex_token_input_ids"])

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

from bleurt import score
checkpoint = "BLEURT-20"
# bleurt_score_metric = score.BleurtScorer(checkpoint)

chrf_score_metric= load("chrf")


def compute_chrf(ref, pred):
    return {'chrf': chrf_score_metric.compute(predictions=pred, references=ref)['score']}
def compute_sari(src, ref, pred):
    refs_corpus_sari_format = [ref]

    sari_score = corpus_sari(
        orig_sents=src,
        sys_sents=pred,
        refs_sents=refs_corpus_sari_format
    )
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(src, pred, refs_corpus_sari_format)

    results = {
        "SARI": sari_score,
        "SARI_ADD": add_score,
        "SARI_KEEP": keep_score,
        "SARI_DEL": del_score
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

def compute_bleurt_score(ref, pred):
    #ref = [r[0] for r in ref]
    #results = bleurt_score_metric.score(references=ref, candidates=pred)
    #return {"bleurt":sum(results)/len(results)}
    results = bleurt_score_metric.score(references=ref, candidates=pred)
    return {"bleurt":sum(results)/len(results)}

def compute_custom(sari, bert_score,bleurt, fkgl,chrf):
    fkgl_remapped = (100-(fkgl*(100/18)))/100
    return {"custom":(float(sari)/100+float(chrf)/100+bert_score+bleurt+fkgl_remapped)/5}


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
    model =  AutoModelForSeq2SeqLM.from_pretrained(path, config=get_model_config())
    model.resize_token_embeddings(len(tokenizer))
    return model




dataset_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 2000, 5000, 10000, 100000, 800000]
scores_list = []
for dataset_size in [0]:

    best_model = init_model_from_path("final_models/t5-dutch-simplify-CT7") #yhavinga/t5-base-dutch"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.to(device)

    def simplify_df(df, model, tokenizer, model_config=get_model_config()):

        outputs = []
        batch_size = 6  # Adjust the batch size as needed
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

    def calc_metrics_from_sentence_pairs(sources, decoded_labels, decoded_preds):

      sari_scores = compute_sari(sources, decoded_labels, decoded_preds)
      bert_scores = compute_bert_score(decoded_labels, decoded_preds)
      bleurt_scores = compute_bleurt_score(decoded_labels, decoded_preds)
      fkgl_score = compute_fkgl(decoded_preds)
      chrf_score = compute_chrf(decoded_labels, decoded_preds)
      custom_score = compute_custom(sari_scores["SARI"],
                                    bert_scores["bert-score"],
                                    bleurt_scores["bleurt"],
                                    fkgl_score["fkgl"],
                                    chrf_score["chrf"])
      return sari_scores | bert_scores | fkgl_score | chrf_score | custom_score| bleurt_scores

    def modify(row, ctokens):

      control_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ctokens]
      if type(row["Complex_token_input_ids"]) == list:
        row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1] + control_token_ids + row["Complex_token_input_ids"][8:])
      else:
        row["Complex_token_input_ids"] = torch.IntTensor(row["Complex_token_input_ids"][:1].tolist() + control_token_ids + row["Complex_token_input_ids"][8:].tolist())

      row["Complex_token_attention_masks"] = torch.ones(len(row["Complex_token_input_ids"]), dtype=torch.int)

      return row


    modified_dataset = test.copy().apply(lambda row: modify(row, ['CLR_1.7', 'DTDR_0.4', 'DTRL_0.6']), axis=1)#,'CLR_1.7', 'DTDR_0.4', 'DTRL_0.6' 'CLR_0.35', 'LR_0.45', 'WRR_1.15', 'ICCR_0.45'
    print(modified_dataset.iloc[1]["Complex_token_input_ids"])

    result = simplify_df(modified_dataset, best_model, tokenizer)
    bleurt_score_metric = score.BleurtScorer(checkpoint)
    scores = calc_metrics_from_sentence_pairs(test["Complex"].tolist(), test["Simple"].tolist(), result)

    # del modified_dataset
    # del best_model
    # torch.cuda.empty_cache()
    # gc.collect()
    # print(scores)
    wandb.log(scores)
    # scores_list.append(scores)

# import pickle
# file = open('ds_size_results', 'wb')
# dump information to that file
# pickle.dump(scores_list, file)
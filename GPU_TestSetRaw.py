#from datasets import load_metric
import transformers
import pandas as pd
from functools import lru_cache
from transformers import AutoTokenizer
import nltk
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM
import wandb
import os
import torch

from tqdm import tqdm
from evaluate import load
import nltk
from sacremoses import  MosesTokenizer
tqdm.pandas()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

wandb.login(key = '8f14c161d9bdaea0c585153d80f5e180a35c2e71')
wandb.init(project="huggingface", name="eval_raw_model")

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_target_length = 128

mTokenizer = MosesTokenizer(lang='nl')

test = pd.read_pickle("plank_dataset_tokenized_NO_7CT_test.pkl")

test["Complex_token_input_ids"] = test["Complex_token_input_ids"].progress_apply(lambda r: r[1:])
test["Complex_token_attention_masks"] = test["Complex_token_input_ids"].apply(lambda item: torch.ones(len(item), dtype=torch.int))

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

chrf_score_metric= load("chrf")

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

def compute_bleurt_score(ref, pred):
    results = bleurt_score_metric.score(references=ref, candidates=pred)
    return {"bleurt":sum(results)/len(results)}

def compute_fkgl(pred):
  return {"fkgl":corpus_fkgl(pred)}

def custom_hash(tensor):
  special_token_ids = tokenizer.all_special_ids

  # Create a mask indicating whether each element in the tensor is a special token ID
  mask = torch.tensor([element.item() not in special_token_ids for element in tensor.flatten()], dtype=torch.bool)

  # Reshape the mask to match the shape of the original tensor
  mask = mask.reshape(tensor.shape)

  # Apply the mask to the original tensor
  filtered_tensor = tensor[mask]

  return str(filtered_tensor.tolist())

source_val_dict = {}
for index in test.index:
    simple_value = test.at[index, 'Simple_token_input_ids']
    complex_value = test.at[index, 'Complex']
    source_val_dict[custom_hash(simple_value)] = complex_value

def compute_custom(sari, bert_score, bleurt, fkgl,chrf):
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

best_model = init_model_from_path("yhavinga/t5-base-dutch") #yhavinga/t5-base-dutch"

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

def calc_metrics_from_sentence_pairs(sources, decoded_labels, decoded_preds):

  sari_scores = compute_sari(sources, decoded_labels, decoded_preds)
  bert_scores = compute_bert_score(decoded_labels, decoded_preds)
  bleurt_scores = compute_bleurt_score(decoded_labels, decoded_preds)
  fkgl_score = compute_fkgl(decoded_preds)
  chrf_score = compute_chrf(decoded_labels, decoded_preds)
  #meteor_score = compute_meteor(decoded_labels, decoded_preds)
  custom_score = compute_custom(sari_scores["SARI"],
                                bert_scores["bert-score"],
                                bleurt_scores["bleurt"],
                                fkgl_score["fkgl"],
                                chrf_score["chrf"])
                                #meteor_score["meteor"])
  return sari_scores | bert_scores | fkgl_score | chrf_score | custom_score | bleurt_scores


best_model = best_model.to('cuda')

result = simplify_df(test, best_model, tokenizer)
bleurt_score_metric = score.BleurtScorer(checkpoint)
scores = calc_metrics_from_sentence_pairs(test["Complex"].tolist(), test["Simple"].tolist(), result)

print(scores)
wandb.log(scores)
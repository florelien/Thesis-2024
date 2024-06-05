import transformers
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb
import torch
from tqdm import tqdm
from evaluate import load
import nltk
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

tqdm.pandas()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

wandb.login(key = '8f14c161d9bdaea0c585153d80f5e180a35c2e71')

special_tokens = ["Simplify: "]
special_tags = ["WLR","CLR", "LR", "WRR", "DTDR","ICCR","DTRL"]
for t in special_tags:
    for i in range(41):
        special_tokens.append(t+"_"+str(round(i*0.05, 2)))

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens = special_tokens)
max_target_length = 128

def custom_hash(tensor):
  special_token_ids = tokenizer.all_special_ids

  # Create a mask indicating whether each element in the tensor is a special token ID
  mask = torch.tensor([element.item() not in special_token_ids for element in tensor.flatten()], dtype=torch.bool)

  # Reshape the mask to match the shape of the original tensor
  mask = mask.reshape(tensor.shape)

  # Apply the mask to the original tensor
  filtered_tensor = tensor[mask]

  return str(filtered_tensor.tolist())

train = pd.read_pickle('plank_dataset_tokenized_7_CT_train.pkl')
#full_dataset["Complex_token_input_ids"] = full_dataset["Complex_token_input_ids"].apply(lambda row: torch.cat([torch.tensor(row[:10]), torch.tensor(row[15:])]))
#full_dataset["Complex_token_attention_masks"] = full_dataset["Complex_token_input_ids"].apply(lambda item: torch.ones(len(item), dtype=torch.int))
dataset_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 2000, 5000, 10000, 100000, 800000]
for dataset_size in [0]:
    # train, val = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)
    # val, test = train_test_split(val, test_size=0.5, random_state=42, shuffle=False)

    #train = train.sample(n=dataset_size, random_state=42)

    dataset_parts = {"train":train}#, "val":val,"test":test}

    source_val_dict = {}
    # for index in val.index:
    #     simple_value = val.at[index, 'Simple_token_input_ids']
    #     complex_value = val.at[index, 'Complex']
    #     source_val_dict[custom_hash(simple_value)] = complex_value

    class CustomDataset(torch.utils.data.Dataset):
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

    train_dataset = CustomDataset(dataset_parts["train"])
    # val_dataset = CustomDataset(dataset_parts["val"])

    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-4

    model_name = f"t5-dutch-simplify-CT7"
    output_dir = f"Models/{model_name}"

    args = Seq2SeqTrainingArguments(
        output_dir,
        save_strategy="epoch",
        save_total_limit=3,
        seed=42,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps = 1,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        optim="adafactor",
        warmup_steps=10,
        remove_unused_columns=False,
        run_name=f"CT7",
        report_to ="wandb"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding =True)

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
    bleurt_score_metric = score.BleurtScorer(checkpoint)

    chrf_score_metric= load("chrf")
    meteor_score_metric = load('meteor')

    def compute_meteor(ref,pred):
        return {'meteor':meteor_score_metric.compute(predictions=pred, references=ref)['meteor']}
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

    def compute_bleurt_score(ref, pred):
        results = bleurt_score_metric.score(references=ref, candidates=pred)
        return {"bleurt":sum(results)/len(results)}

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
    # for index in val.index:
    #     simple_value = val.at[index, 'Simple_token_input_ids']
    #     complex_value = val.at[index, 'Complex']
    #     source_val_dict[custom_hash(simple_value)] = complex_value

    def compute_custom(sari, bert_score, bleurt, fkgl,chrf, meteor):
        fkgl_remapped = (100-(fkgl*(100/18)))/100
        return {"custom":(float(sari)/100+float(chrf)/100+float(meteor)/100+bert_score+bleurt+fkgl_remapped)/6}


    def compute_metrics(eval_pred):

        return {}
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        sources = []
        for label in labels:
          t = torch.tensor(label)
          sources.append(source_val_dict[custom_hash(t)])

        sari_scores = compute_sari(sources, decoded_labels, decoded_preds)
        bert_scores = compute_bert_score(decoded_labels, decoded_preds)
        bleurt_scores = compute_bleurt_score(decoded_labels, decoded_preds)
        fkgl_score = compute_fkgl(decoded_preds)
        chrf_score = compute_chrf(decoded_labels, decoded_preds)
        meteor_score = compute_meteor(decoded_labels, decoded_preds)
        custom_score = compute_custom(sari_scores["SARI"],
                                      bert_scores["bert-score"],
                                      bleurt_scores["bleurt"],
                                      fkgl_score["fkgl"],
                                      chrf_score["chrf"],
                                      meteor_score["meteor"])
        return sari_scores | bert_scores  |  fkgl_score | chrf_score | meteor_score|  custom_score |bleurt_scores


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

    def model_init():
        model =  AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=get_model_config())
        model.resize_token_embeddings(len(tokenizer))
        return model

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

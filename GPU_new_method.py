from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from tqdm import tqdm
tqdm.pandas()
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
vocab_size = tokenizer.vocab_size
data = pd.read_pickle('plank_dataset_token_values_bert_tokenized_small.pkl')

"""
encoded_corpus = tokenizer(text=data.Complex.tolist(),
                            add_special_tokens=True,
                            return_attention_mask=True)

input_ids = encoded_corpus['input_ids']
attention_mask = encoded_corpus['attention_mask']

data['Complex_input_ids'] = input_ids
data['Complex_attention_masks'] = attention_mask


data["control_token_values"] = data["control_token_values"].progress_apply(lambda x: torch.tensor(x, dtype=torch.float))
data["Complex_input_ids"] = data["Complex_input_ids"].progress_apply(lambda x: torch.tensor(x, dtype=torch.int))
data["Complex_attention_masks"] = data["Complex_attention_masks"].progress_apply(lambda x: torch.tensor(x, dtype=torch.int))
"""

train_data, val_data = train_test_split(data, test_size=0.05, random_state=42)
class CustomDataset(torch.utils.data.Dataset):
    """Dataset class that allows easy requests of a data row."""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return {
            "input_ids": torch.tensor(row["Complex_input_ids"], dtype=torch.int),
            "attention_mask": torch.tensor(row["Complex_attention_masks"], dtype=torch.int),
            "labels": torch.tensor(row["control_token_values"], dtype=torch.float)
        }

# Load the BERT tokenizer and model
model_name = 'wietsedv/bert-base-dutch-cased'  # Use a suitable Dutch BERT model from Hugging Face
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, output_hidden_states=False)
bert_model = BertModel.from_pretrained(model_name, config=config)

# Define a regression head for the BERT model
class BertRegression(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_labels=10):
        super(BertRegression, self).__init__()
        self.bert = bert_model
        self.regressor = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Only take the [CLS] token's hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]
        regression_output = self.regressor(cls_output)
        return regression_output

model = BertRegression(bert_model)

# Prepare the dataset and dataloader
batch_size = 16
data_collator = DataCollatorWithPadding(tokenizer)

train_dataset = CustomDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

val_dataset = CustomDataset(val_data)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 15
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

criterion = nn.MSELoss()

best_val_loss = 100000
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    batch_idx = 0

    # Training
    for batch in train_dataloader:
        #print(f'Batch index: {batch_idx}')
        """print(f'Input IDs shape: {batch["input_ids"].shape}')
        print(f'Attention mask shape: {batch["attention_mask"].shape}')
        print(f'Labels shape: {batch["labels"].shape}')
        print(batch_idx)"""
        batch_idx+=1

        """print(batch["input_ids"])
        print(batch["attention_mask"])
        print(batch["labels"])"""

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        """if batch_idx%5000==0:
            # Validation
            avg_train_loss = total_train_loss / len(train_dataloader)
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device).float()

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs},Batch {batch_idx} Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}')
            model.train()"""

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_save_path = f'bert_regression_model{epoch}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    print(f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}')

# Save the trained model
model_save_path = 'bert_regression_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

print("Training complete.")
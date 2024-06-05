import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()

full_dataset = pd.read_csv("drive/MyDrive/FloThesis/plank_dataset.csv")
full_dataset = full_dataset.replace(r'\\', '', regex=True)
nlp = spacy.load("nl_core_news_sm")

def count_adjectives(sentence):
    doc = nlp(sentence)
    toks = [token.pos_ for token in doc]
    return toks

# Apply the function to each sentence in the DataFrame
full_dataset['Simple_Tags'] = full_dataset['Simple'].progress_apply(count_adjectives)
full_dataset['Complex_Tags'] = full_dataset['Complex'].progress_apply(count_adjectives)
full_dataset.to_pickle("plank_dataset_POS.pkl")
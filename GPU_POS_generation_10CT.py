import pandas as pd
from sacremoses import MosesDetokenizer, MosesTokenizer
from transformers import AutoTokenizer
import nltk
import pickle
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from string import punctuation
from nltk.corpus import stopwords
from functools import lru_cache
import spacy
import numpy as np
from nltk.tag.perceptron import PerceptronTagger
import Levenshtein
from tqdm import tqdm
tqdm.pandas()

tagger = PerceptronTagger(load=False)
tagger.load('file:POS_TAGGER.pickle')

special_tokens = []
special_tags = ["WLR","CLR", "LR", "WRR", "DTDR","ICCR","DTRL","NSC","PRC","PPEN"]
for t in special_tags:
    for i in range(41):
        special_tokens.append(t+"_"+str(round(i*0.05, 2)))
        
full_dataset = pd.read_pickle('full_dataset_processed.pkl')
model_checkpoint = "yhavinga/t5-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens = special_tokens)
max_target_length = 128
mTokenizer = MosesTokenizer(lang='nl')
full_dataset = full_dataset.replace(r'\\', '', regex=True)
		
def m_tokenize(sentence):
  return mTokenizer.tokenize(sentence)
def normal_tokenize(sentence):
  return tokenizer.tokenize(sentence)
  
def bin_round(x, prec=2):
  return x

def is_punctuation(word):
    return ''.join([char for char in word if char not in punctuation]) == ''

def remove_punctuation(text):
    return ' '.join([word for word in m_tokenize(text) if not is_punctuation(word)])

def remove_stopwords(text):
    return ' '.join([w for w in m_tokenize(text) if w.lower() not in stopwords.words('dutch')])
	
def load_dump(filepath):
    return pickle.load(open(filepath, 'rb'))

def dump(obj, filepath):
    pickle.dump(obj, open(filepath, 'wb'))
	
difficult_tags = {
    'nounpl':1,
    'nounsg':1,
    'verbpressg':2,
    'num__card':2,
    'pronpers':1,
    'adj':2,
    'verbpapa':3,
    'adv':2,
    'nounprop':2,
    'verbpresp':3,
    'verbpastsg':3,
    'verbpastpl':3,
    'nounabbr':4,
    'pronadv':2,
    'prondemo':2,
    'prep':1,
    'conjsubo':3,
    'conjcoord':2,
    'det__demo':2,
    'verbinf':2,
    'num__ord':2
}

def get_lexical_complexity_score(complex, simple):
  complex_ranks = get_lexical_complexity_score_helper(complex)
  simple_ranks = get_lexical_complexity_score_helper(simple)

  complex_rank = sum(complex_ranks)/len(complex_ranks)
  simple_rank = sum(simple_ranks)/max(len(complex_ranks), len(simple_ranks))

  return simple_rank/complex_rank

def get_lexical_complexity_score_helper(sentence):
  sentence =sentence.lower()
  words = m_tokenize(remove_stopwords(remove_punctuation(sentence)))
  words = [word for word in words if word in get_word2rank()]
  if len(words) == 0:
      return [10] #pretty high rank
  ranks = [get_rank(word) for word in words]
  return ranks

@lru_cache(maxsize=5000)
def get_rank(word):
  rank = get_word2rank().get(word, len(get_word2rank()))
  ranker = np.log(1 + rank)
  return ranker

@lru_cache(maxsize=1)
def get_word2rank(vocab_size=np.inf):
  return load_dump("model.bin")

@lru_cache(maxsize=1024)
def get_dependency_tree_depth(sentence):
  def get_subtree_depth(node):
      if len(list(node.children)) == 0:
          return 0
      return 1 + max([get_subtree_depth(child) for child in node.children])

  tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
  if len(tree_depths) == 0:
      return 0
  return max(tree_depths)

@lru_cache(maxsize=10 ** 6)
def spacy_process(text):
  return get_spacy_model()(text)

@lru_cache(maxsize=1)
def get_spacy_model():
  model = 'nl_core_news_lg'  # from spacy, Dutch pipeline
  if not spacy.util.is_package(model):
      spacy.cli.download(model)
      spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
  return spacy.load(model)

def get_dependency_tree(text):
  doc = get_spacy_model()(text)
  # Extract dependency tree
  tree = []
  for token in doc:
      tree.append((token.text, token.dep_, token.head.text, token.i, token.idx))
  return tree

@lru_cache(maxsize=25)
def get_POS_tags(text):
  tags = tagger.tag(text.split())
  return [token[1] for token in tags]

def count_tree_rel_length(tree_list):
    c = 0
    sentence = [rel[0] for rel in tree_list]
    rels = [rel[1] for rel in tree_list]

    for ind,rel in enumerate(tree_list):
      c += abs(ind-sentence.index(rel[2]))

    return c/len(tree_list)

def count_tag_points(tag_list):
  listt = 0
  for tag in tag_list:
    if tag in difficult_tags:
      listt+=difficult_tags[tag]
  return listt/len(tag_list)
  
def get_special_token(type, value, special_tokens = tokenizer.all_special_tokens):
  for token in special_tokens:
    if token.startswith(type) and token.endswith(value):
      return token
	  
def calcWLR(raw_complex, raw_simple):
  #WordLengthRatio
  res = bin_round(len(m_tokenize(raw_simple)) / len(m_tokenize(raw_complex)))
  res = min(res, 2)
  return res, get_special_token("WLR", str(res))

def calcCLR(raw_complex, raw_simple):
  #CharLengthRatio
  res = bin_round(len(raw_simple)/len(raw_complex))
  res = min(res, 2)
  return res, get_special_token("CLR", str(res))

def calcLR(raw_complex, raw_simple):
  #LevenshteinRatio
  res = bin_round(Levenshtein.ratio(m_tokenize(raw_complex), m_tokenize(raw_simple)))
  res = min(res, 2)
  return res, get_special_token("LR", str(res))

def calcWRR(raw_complex, raw_simple):
  #WordRankRatio
  res = bin_round(get_lexical_complexity_score(raw_complex,raw_simple))
  res = min(res, 2)
  return res, get_special_token("WRR", str(res))

def calcDTDR(raw_complex, raw_simple):
  #DependencyTreeDepthRatio
  res = bin_round(get_dependency_tree_depth(raw_simple)/get_dependency_tree_depth(raw_complex))
  res = min(res, 2)
  return res, get_special_token("DTDR", str(res))

##########################################################      EXISTING         ########################################################################################

def calcICCR(raw_complex, raw_simple):
  #InverseCopyControlRatio
  simple_tokenized = m_tokenize(remove_punctuation(raw_simple))
  complex_tokenized = m_tokenize(remove_punctuation(raw_complex))

  num_replacements = 0
  for ctoken in complex_tokenized:
    if ctoken not in simple_tokenized:
      num_replacements += 1
  res = bin_round(1-num_replacements/len(complex_tokenized))
  return res, get_special_token("ICCR", str(res))

##########################################################      CUSTOM         ########################################################################################
def calcDTRL(raw_complex, raw_simple):
  #DependencyTreeRelationLength
  simple_rel_length = count_tree_rel_length(get_dependency_tree(raw_simple))
  complex_rel_length = count_tree_rel_length(get_dependency_tree(raw_complex))
  ratio = bin_round(simple_rel_length/complex_rel_length)
  res = min(ratio, 2)
  return res, get_special_token("DTRL", str(res))

def calcNSC(raw_complex, raw_simple):
  #NounSingularCount
  simple_nounsg_count = get_POS_tags(raw_simple).count("nounsg")+1
  complex_nounsg_count = get_POS_tags(raw_complex).count("nounsg")+1
  ratio = bin_round(simple_nounsg_count/complex_nounsg_count)
  res = min(ratio, 2)
  return res, get_special_token("NSC", str(res))

def calcPRC(raw_complex, raw_simple):
  #PrepCount
  simple_prep_count = get_POS_tags(raw_simple).count("prep")+1
  complex_prep_count = get_POS_tags(raw_complex).count("prep")+1
  ratio = bin_round(simple_prep_count/complex_prep_count)
  res = min(ratio, 2)
  return res, get_special_token("PRC", str(res))

def calcPPEN(raw_complex, raw_simple):
  #POStagPenalty
  ratio = bin_round(count_tag_points(get_POS_tags(raw_simple))/count_tag_points(get_POS_tags(raw_complex)))
  res = min(ratio, 2)
  return res, get_special_token("PPEN", str(res))

def get_control_tokens(raw_complex, raw_simple):
  return [
      calcWLR(raw_complex, raw_simple)[0],
      calcCLR(raw_complex, raw_simple)[0],
      calcLR(raw_complex, raw_simple)[0],
      calcWRR(raw_complex, raw_simple)[0],
      calcDTDR(raw_complex, raw_simple)[0],
      calcICCR(raw_complex, raw_simple)[0],
      calcDTRL(raw_complex, raw_simple)[0],
      calcNSC(raw_complex, raw_simple)[0],
      calcPRC(raw_complex, raw_simple)[0],
      calcPPEN(raw_complex, raw_simple)[0]
  ]

full_dataset["control_token_values"] = full_dataset.progress_apply(lambda row: get_control_tokens(row["Complex"], row["Simple"]), axis=1)
full_dataset.to_pickle("plank_dataset_token_values.pkl")
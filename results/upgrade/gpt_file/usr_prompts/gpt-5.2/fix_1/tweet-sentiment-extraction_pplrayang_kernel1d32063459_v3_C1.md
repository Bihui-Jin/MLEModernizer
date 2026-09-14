# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tokenizers==0.21.2
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.7162611484527588

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
from tqdm.notebook import tqdm
import sys
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

%matplotlib inline
warnings.filterwarnings('ignore')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)

batch_size = 32
N = 10

skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

ROBERTA_PATH = '/kaggle/input/robertamodel0524/'
MODEL_CONFIG_PATH = ROBERTA_PATH+'roberta-base-config.json'
MODEL_PATH = ROBERTA_PATH+'roberta-base-pytorch_model.bin'
MODEL_VOCAB_PATH = ROBERTA_PATH+'roberta-base-vocab.json'
MODEL_VOCAB_MERGES_PATH = ROBERTA_PATH+'roberta-base-merges.txt'
outdir = '/kaggle/input/roberta714kernel/'

test_file = '/kaggle/input/tweet-sentiment-extraction/test.csv'
submission_template = '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv'

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
  def __init__(self, df, max_len=MAX_LEN):
    self.df = df
    self.max_len = max_len
    self.labeled = 'selected_text' in df
    self.tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file = MODEL_VOCAB_PATH, 
        merges_file = MODEL_VOCAB_MERGES_PATH, 
        lowercase=True,
        add_prefix_space=True)

  def __getitem__(self, index):
    data = {}
    row = self.df.iloc[index]
    
    ids, masks, tweet, offsets = self.get_input_data(row)
    data['ids'] = ids
    data['masks'] = masks
    data['tweet'] = tweet
    data['offsets'] = offsets
    
    if self.labeled:
      start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
      data['start_idx'] = start_idx
      data['end_idx'] = end_idx
    
    return data

  def __len__(self):
    return len(self.df)
  
  def get_input_data(self, row):
    tweet = " " + " ".join(row.text.lower().split())
    encoding = self.tokenizer.encode(tweet)
    sentiment_id = self.tokenizer.encode(row.sentiment).ids
    ids = [CLS_TOK] + sentiment_id + [SEP_TOK, SEP_TOK] + encoding.ids + [SEP_TOK]
    offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
            
    pad_len = self.max_len - len(ids)
    if pad_len > 0:
      ids += [PAD_TOK] * pad_len
      offsets += [(0, 0)] * pad_len
    
    ids = torch.tensor(ids)
    masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
    offsets = torch.tensor(offsets)
    
    return ids, masks, tweet, offsets
      
  def get_target_idx(self, row, tweet, offsets):
    selected_text = " " +  " ".join(row.selected_text.lower().split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
      if " " + tweet[ind: ind+len_st] == selected_text:
        idx0 = ind
        idx1 = ind + len_st - 1
        break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
      for ct in range(idx0, idx1 + 1):
        char_targets[ct] = 1

    target_idx = []
    for j, (offset1, offset2) in enumerate(offsets):
      if sum(char_targets[offset1: offset2]) > 0:
        target_idx.append(j)

    start_idx = target_idx[0]
    end_idx = target_idx[-1]
    
    return start_idx, end_idx

## === cell 2
def get_test_loader(df, batch_size=32):
  loader = torch.utils.data.DataLoader(
    TweetDataset(df), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=NUM_WORKERS)    
  return loader

## === cell 3
class TweetModel(nn.Module):
  def __init__(self):
    super(TweetModel, self).__init__()
    
    config = RobertaConfig.from_pretrained(
        MODEL_CONFIG_PATH, output_hidden_states=True)    
    self.roberta = RobertaModel.from_pretrained(
        MODEL_PATH, config=config)
    self.dropout = nn.Dropout(LINEAR_DROPOUT)
    self.fc = nn.Linear(config.hidden_size, 2)
    nn.init.normal_(self.fc.weight, std=0.02)
    nn.init.normal_(self.fc.bias, 0)

  def forward(self, input_ids, attention_mask):
    _, _, hs = self.roberta(input_ids, attention_mask)
      
    x = torch.stack([hs[-1], hs[-2], hs[-3]])
    x = torch.mean(x, 0)
    x = self.dropout(x)
    x = self.fc(x)
    start_logits, end_logits = x.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
            
    return start_logits, end_logits

## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
  selected_text = ""
  for ix in range(start_idx, end_idx + 1):
    selected_text += text[offsets[ix][0]: offsets[ix][1]]
    if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
      selected_text += " "
  return selected_text

def jaccard(str1, str2): 
  a = set(str1.lower().split()) 
  b = set(str2.lower().split())
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
  start_pred = np.argmax(start_logits)
  end_pred = np.argmax(end_logits)
  if start_pred > end_pred:
    pred = text
  else:
    pred = get_selected_text(text, start_pred, end_pred, offsets)
      
  true = get_selected_text(text, start_idx, end_idx, offsets)
  
  return jaccard(true, pred)

## === cell 5
%%time

test_df = pd.read_csv(test_file)
test_df['text'] = test_df['text'].astype(str)
test_loader = get_test_loader(test_df)
predictions = []
models = []

print("loading models..")
for fold in tqdm(range(skf.n_splits)):
    model = TweetModel()
    model.cuda()
    model.load_state_dict(torch.load(f'{outdir}roberta_fold{fold+1}.pth'))
    model.eval()
    models.append(model)

for data in tqdm(test_loader):
  ids = data['ids'].cuda()
  masks = data['masks'].cuda()
  tweet = data['tweet']
  offsets = data['offsets'].numpy()

  start_logits = []
  end_logits = []
  for model in models:
    with torch.no_grad():
        output = model(ids, masks)
        start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
        end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

  start_logits = np.mean(start_logits, axis=0)
  end_logits = np.mean(end_logits, axis=0)
  for i in range(len(ids)):    
    start_pred = np.argmax(start_logits[i])
    end_pred = np.argmax(end_logits[i])
    if start_pred > end_pred:
        pred = tweet[i]
    else:
        pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
    predictions.append(pred)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<timed exec> in <module>

/tmp/ipykernel_11/2729044725.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2   loader = torch.utils.data.DataLoader(
----> 3     TweetDataset(df),
      4     batch_size=batch_size,
      5     shuffle=False,

/tmp/ipykernel_11/2408062880.py in __init__(self, df, max_len)
      4     self.max_len = max_len
      5     self.labeled = 'selected_text' in df
----> 6     self.tokenizer = tokenizers.ByteLevelBPETokenizer(
      7         vocab_file = MODEL_VOCAB_PATH,
      8         merges_file = MODEL_VOCAB_MERGES_PATH,

TypeError: ByteLevelBPETokenizer.__init__() got an unexpected keyword argument 'vocab_file'

## === cell 6
sub_df = pd.read_csv(submission_template)
sub_df['selected_text'] = predictions
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
sub_df.to_csv('submission.csv', index=False)
sub_df.head(20)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/188658866.py in <cell line: 0>()
      1 sub_df = pd.read_csv(submission_template)
----> 2 sub_df['selected_text'] = predictions
      3 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
      4 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
      5 sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

NameError: name 'predictions' is not defined

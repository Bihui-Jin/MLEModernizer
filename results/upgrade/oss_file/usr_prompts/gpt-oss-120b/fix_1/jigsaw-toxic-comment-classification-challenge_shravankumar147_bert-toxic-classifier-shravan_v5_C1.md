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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.98112

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip


## === cell 2
TRAIN = './train.csv'
TEST = './test.csv'
TEST_LABEL = './test_labels.csv'
SAMPLE = './sample_submission.csv'
EPOCHS = 2
MAX_TOKEN_COUNT = 128
BATCH_SIZE = 32


## === cell 3
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
df = pd.read_csv(TRAIN)
test_df = pd.read_csv(TEST)
test_label = pd.read_csv(TEST_LABEL)
sample_sub = pd.read_csv(SAMPLE)
df.describe()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/184739175.py in <cell line: 0>()
      1 df = pd.read_csv(TRAIN)
      2 test_df = pd.read_csv(TEST)
----> 3 test_label = pd.read_csv(TEST_LABEL)
      4 sample_sub = pd.read_csv(SAMPLE)
      5 df.describe()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './test_labels.csv'

## === cell 5
df.head()


## === cell 6
test_df.head()


## === cell 7
test_label.head()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2235443609.py in <cell line: 0>()
----> 1 test_label.head()

NameError: name 'test_label' is not defined

## === cell 8
train_df, val_df = train_test_split(df, test_size=0.05)
train_df.shape, val_df.shape, test_df.shape #test hs only ids and comment_text

LABEL_COLUMNS = df.columns.tolist()[2:]
df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh");


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3308535851.py in <cell line: 0>()
----> 1 train_df, val_df = train_test_split(df, test_size=0.05)
      2 train_df.shape, val_df.shape, test_df.shape #test hs only ids and comment_text
      3 
      4 LABEL_COLUMNS = df.columns.tolist()[2:]
      5 df[LABEL_COLUMNS].sum().sort_values().plot(kind="barh");

NameError: name 'train_test_split' is not defined

## === cell 9
train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]

pd.DataFrame(dict(
  toxic=[len(train_toxic)], 
  clean=[len(train_clean.sample(15_000))]
)).plot(kind='barh');

train_df = pd.concat([
  train_toxic,
  train_clean.sample(15_000)
])

train_df.shape, val_df.shape


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2809987222.py in <cell line: 0>()
----> 1 train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
      2 train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
      3 
      4 pd.DataFrame(dict(
      5   toxic=[len(train_toxic)],

NameError: name 'train_df' is not defined

## === cell 10
class ToxicCommentsDataset(Dataset):
    """
    Pass pandas dataframe, and tokeizer along with the max token length[128 default]
    
    Example: 
    -------
    train_dataset = ToxicCommentsDataset(
      train_df,
      tokenizer,
      max_token_len=MAX_TOKEN_COUNT
    )

    sample_item = train_dataset[0]
    
    """
    
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_token_len: int = 128,
        test= False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.test = test
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        _id = data_row['id']
        comment_text = data_row.comment_text
        
        if not self.test:
            labels = data_row[LABEL_COLUMNS]
        
        encoding = self.tokenizer.encode_plus(
            comment_text,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True, # [CLS] & [SEP]
            return_token_type_ids=False,
            return_attention_mask=True, #attention_mask
            return_tensors='pt',
        )
        
        if not self.test:
            return dict(
            _id = _id,
            comment_text=comment_text,
            input_ids = encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )
        else:
            return dict(
                _id = _id,
                comment_text=comment_text,
                input_ids = encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten()
            )
        


## === cell 11
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


## === cell 12
train_dataset = ToxicCommentsDataset(
  train_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)

val_dataset = ToxicCommentsDataset(
  val_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/984963529.py in <cell line: 0>()
      1 # test=False
      2 train_dataset = ToxicCommentsDataset(
----> 3   train_df,
      4   tokenizer,
      5   max_token_len=MAX_TOKEN_COUNT

NameError: name 'train_df' is not defined

## === cell 13
test_dataset = ToxicCommentsDataset(
  test_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT,
  test=True
)


## === cell 14
sample_item = train_dataset[0]
print(sample_item.keys())
print(sample_item["_id"])
print(sample_item["comment_text"])
print(sample_item["input_ids"])
print(sample_item["attention_mask"])
print(sample_item["labels"])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2351611135.py in <cell line: 0>()
----> 1 sample_item = train_dataset[0]
      2 print(sample_item.keys())
      3 print(sample_item["_id"])
      4 print(sample_item["comment_text"])
      5 print(sample_item["input_ids"])

NameError: name 'train_dataset' is not defined

## === cell 15
sample_item = val_dataset[0]
print(sample_item.keys())
print(sample_item["_id"])
print(sample_item["comment_text"])
print(sample_item["input_ids"])
print(sample_item["attention_mask"])
print(sample_item["labels"])


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1692202102.py in <cell line: 0>()
----> 1 sample_item = val_dataset[0]
      2 print(sample_item.keys())
      3 print(sample_item["_id"])
      4 print(sample_item["comment_text"])
      5 print(sample_item["input_ids"])

NameError: name 'val_dataset' is not defined

## === cell 16
sample_item = test_dataset[0]
print(sample_item.keys())
print(sample_item["_id"])
print(sample_item["comment_text"])
print(sample_item["input_ids"])
print(sample_item["attention_mask"])


## === cell 18
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1677681324.py in <cell line: 0>()
----> 1 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
      2 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
      3 test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NameError: name 'train_dataset' is not defined

## === cell 19
len(train_dataloader)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1500177583.py in <cell line: 0>()
----> 1 len(train_dataloader)

NameError: name 'train_dataloader' is not defined

## === cell 20
import gc
gc.collect()


## === cell 22
class ToxicCommentTagger(nn.Module):
    
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True) #load the pretrained bert model
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes) # add a linear layer to the bert
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output) 
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


## === cell 23
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = ToxicCommentTagger(len(LABEL_COLUMNS)).to(device)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3250778849.py in <cell line: 0>()
      1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
----> 2 bert_model = ToxicCommentTagger(len(LABEL_COLUMNS)).to(device)

NameError: name 'LABEL_COLUMNS' is not defined

## === cell 24
print(bert_model)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2661245729.py in <cell line: 0>()
----> 1 print(bert_model)

NameError: name 'bert_model' is not defined

## === cell 25
model = bert_model


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3217198796.py in <cell line: 0>()
----> 1 model = bert_model

NameError: name 'bert_model' is not defined

## === cell 27
N_EPOCHS = EPOCHS


steps_per_epoch=len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/778052312.py in <cell line: 0>()
      2 
      3 
----> 4 steps_per_epoch=len(train_df) // BATCH_SIZE
      5 total_training_steps = steps_per_epoch * N_EPOCHS
      6 warmup_steps = total_training_steps // 5

NameError: name 'train_df' is not defined

## === cell 28
optimizer = AdamW(model.parameters(), lr=2e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3950697079.py in <cell line: 0>()
----> 1 optimizer = AdamW(model.parameters(), lr=2e-5)
      2 
      3 scheduler = get_linear_schedule_with_warmup(
      4     optimizer,
      5     num_warmup_steps=warmup_steps,

NameError: name 'AdamW' is not defined

## === cell 29
def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  avg_loss = 0
  
  total_preds=[]
  for step,batch in enumerate(train_dataloader):
    
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)     

    model.zero_grad()        
    loss, outputs = model(input_ids, attention_mask, labels)

    total_loss = total_loss + loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    outputs=outputs.detach().cpu().numpy()

    total_preds.append(outputs)

  avg_loss = total_loss / len(train_dataloader)
  print(f"{step}: {avg_loss}")
  

    
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds


## === cell 31
from sklearn.metrics import roc_auc_score
def evaluate():
  
  print("\nEvaluating...")
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  total_preds = []
  total_labels = []

  for step,batch in enumerate(val_dataloader):
    
    if step % 50 == 0 and not step == 0:
      
            
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))


    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)   
    with torch.no_grad():
      
      loss, outputs = model(input_ids, attention_mask, labels)

      total_loss = total_loss + loss.item()

      outputs = outputs.detach().cpu().numpy()
      labels = labels.detach().cpu().numpy()
      total_preds.append(outputs)
      total_labels.append(labels)


  avg_loss = total_loss / len(val_dataloader)
  print(f"{step}: {avg_loss}")



  total_preds  = np.concatenate(total_preds, axis=0)
  total_labels = np.concatenate(total_labels, axis=0)
  true = np.array(total_labels)
  pred = np.array(total_preds>0.5)
  for i, name in enumerate(LABEL_COLUMNS):
      try:
          print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
      except Exception as e:
        print(e)
        pass
  print(f"Evaluate loss {total_loss / len(val_dataloader)}")
  return avg_loss, total_preds, total_labels


## === cell 32
%%time
best_valid_loss = float('inf')

train_losses=[]
valid_losses=[]
EPOCHS = 2
for epoch in range(EPOCHS):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS))
    
    train_loss, _ = train()
    
    valid_loss, _, _ = evaluate()
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

/tmp/ipykernel_11/683888827.py in train()
      2 def train():
      3 
----> 4   model.train()
      5 
      6   total_loss, total_accuracy = 0, 0

NameError: name 'model' is not defined

## === cell 33
def test():
  
  print("\nTesting...")
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  total_preds = []
  _ids = []

  for step,batch in enumerate(test_dataloader):
    
    if step % 50 == 0 and not step == 0:
      
            
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))


    _id = batch["_id"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
      
      loss, outputs = model(input_ids, attention_mask)

      total_loss = total_loss + loss

      outputs = outputs#.detach().cpu().numpy()
      _ids.append(_id)
      total_preds.append(outputs)
    
  avg_loss = total_loss / len(test_dataloader) 

  _ids = np.concatenate(_ids, axis=0)
  total_preds  = torch.cat(total_preds, axis=0)
  results = dict(id=_ids,
      predictions = total_preds
      )
    

  return avg_loss, total_preds, results


## === cell 34
from sklearn.metrics import accuracy_score, roc_curve, auc

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs#[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


## === cell 35
avg_loss, total_preds, total_labels = evaluate()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1666399895.py in <cell line: 0>()
----> 1 avg_loss, total_preds, total_labels = evaluate()

/tmp/ipykernel_11/2404538635.py in evaluate()
      6   #t0 = time.time()
      7   # deactivate dropout layers
----> 8   model.eval()
      9 
     10   total_loss, total_accuracy = 0, 0

NameError: name 'model' is not defined

## === cell 36
for i, name in enumerate(LABEL_COLUMNS):
    print(f"label: {name}")
    evaluate_roc(total_preds[:,i]>0.5, total_labels[:,i])


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/107537343.py in <cell line: 0>()
----> 1 for i, name in enumerate(LABEL_COLUMNS):
      2     print(f"label: {name}")
      3     evaluate_roc(total_preds[:,i]>0.5, total_labels[:,i])

NameError: name 'LABEL_COLUMNS' is not defined

## === cell 37
avg_test_loss, total_test_preds, sub = test()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2354088587.py in <cell line: 0>()
----> 1 avg_test_loss, total_test_preds, sub = test()

/tmp/ipykernel_11/947233897.py in test()
      5   #t0 = time.time()
      6   # deactivate dropout layers
----> 7   model.eval()
      8 
      9   total_loss, total_accuracy = 0, 0

NameError: name 'model' is not defined

## === cell 38
D = pd.DataFrame()
D['id'] = sub['id']
D


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1916859410.py in <cell line: 0>()
      1 D = pd.DataFrame()
----> 2 D['id'] = sub['id']
      3 D

NameError: name 'sub' is not defined

## === cell 39
D[LABEL_COLUMNS] = (sub['predictions'].cpu().numpy())
D


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3172266715.py in <cell line: 0>()
----> 1 D[LABEL_COLUMNS] = (sub['predictions'].cpu().numpy())
      2 D

NameError: name 'sub' is not defined

## === cell 40
D.to_csv("submission.csv", index=False)

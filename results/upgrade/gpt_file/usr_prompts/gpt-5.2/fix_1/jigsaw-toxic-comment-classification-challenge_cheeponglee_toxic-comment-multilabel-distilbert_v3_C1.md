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

3.9

# 3. Installed packages

datasets==4.4.1
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
pyarrow==19.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

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

0.98446

# 6. Current score

Not yielded

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
!pip install transformers --quiet
!pip install datasets transformers[sentencepiece] --quiet
!pip install "transformers[sentencepiece]" --quiet


## === cell 2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
import time
import datetime


## === cell 3
df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test_csv = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_csv_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
print(df.columns)
print(df.shape)
target_col= df.columns[2:]
feature_col= df.columns[1:2]
df.head()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1218133583.py in <cell line: 0>()
      1 df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
      2 test_csv = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
----> 3 test_csv_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
      4 print(df.columns)
      5 print(df.shape)

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
    792             # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
    793             # ReadBuffer[bytes], WriteBuffer[bytes]]"
--> 794             handle = _BytesZipFile(
    795                 handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
    796             )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in __init__(self, file, mode, archive_name, **kwargs)
   1035         # error: Incompatible types in assignment (expression has type "ZipFile",
   1036         # base class "_BufferedWriter" defined the type as "BytesIO")
-> 1037         self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
   1038             file, mode, **kwargs
   1039         )

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1293             while True:
   1294                 try:
-> 1295                     self.fp = io.open(file, filemode)
   1296                 except OSError:
   1297                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'

## === cell 4
for col in target_col:
    print(f"The unique value for {col} are {df[col].unique()}")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260416605.py in <cell line: 0>()
----> 1 for col in target_col:
      2     print(f"The unique value for {col} are {df[col].unique()}")

NameError: name 'target_col' is not defined

## === cell 5
for col in target_col:
    print(f"The unique value for {col} are {test_csv_labels[col].unique()}")


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3088611106.py in <cell line: 0>()
----> 1 for col in target_col:
      2     print(f"The unique value for {col} are {test_csv_labels[col].unique()}")

NameError: name 'target_col' is not defined

## === cell 6
print(target_col)
print(feature_col)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/519984670.py in <cell line: 0>()
----> 1 print(target_col)
      2 print(feature_col)

NameError: name 'target_col' is not defined

## === cell 7
df.dtypes


## === cell 8
categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
sns.barplot(x=categories,y=df[categories].sum())
plt.show()


## === cell 9
condition = ((df["toxic"]==1) |
             (df["severe_toxic"]==1) |
             (df["obscene"]==1) |
             (df["threat"]==1) |
             (df["insult"]==1) |
             (df["identity_hate"]==1)
            )

df.loc[condition, "y"]=1
df.loc[~condition, "y"]=0
df["y"] = df["y"].astype(int)


## === cell 10
agg_df = df["y"]\
    .value_counts(normalize=True)\
    .rename('Proportion')\
    .reset_index()
    
agg_df.columns = ["y", "Proportion"]


## === cell 11
splot = sns.barplot(x="y", y='Proportion',data=agg_df)
plt.ylim(0,1)
plt.title("Proportion of Non-Toxic/Toxic Post")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')


## === cell 12
df['len'] = df["comment_text"].str.split().str.len()


## === cell 13
print(f"Average length is {df['len'].mean()} while max length is {df['len'].max()}.")
df['len'].hist(bins=20)
plt.show()


## === cell 14
sns.histplot(data=df, x="len", hue="y", bins=50, alpha=0.2)


## === cell 15
sns.histplot(data=df, x="len", hue="y",
             log_scale=True, element="step", fill=False,
             cumulative=True, stat="density", common_norm=False)


## === cell 16
df = df.rename(columns={"id": "idx"})


## === cell 17
df.head()


## === cell 18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import (AutoTokenizer, AutoModel, 
                          AutoModelForSequenceClassification, 
                          DataCollatorWithPadding, AdamW, get_scheduler,
                          get_linear_schedule_with_warmup,
                          )
import pyarrow as pa
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import datasets
import random
from sklearn.metrics import classification_report


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 19
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2465502310.py in <cell line: 0>()
      1 # Setting up seed value
      2 seed_value = 42
----> 3 random.seed(seed_value)
      4 np.random.seed(seed_value)
      5 torch.manual_seed(seed_value)

NameError: name 'random' is not defined

## === cell 20
train_val_df, test_df = train_test_split(df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value)


## === cell 21
train_df, val_df, = train_test_split(train_val_df[["idx", "comment_text"] + categories], test_size=0.25, random_state=seed_value)


## === cell 22
print(f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively.")
print(f"Proportion of train, validation and test are {round(len(train_df)/len(df),2)}, {round(len(val_df)/len(df),2)}, {round(len(test_df)/len(df),2)} respectively.")


## === cell 23
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)

test_df.reset_index(inplace=True)
test_df.drop("index", axis=1, inplace=True)


## === cell 24
train_df.head()


## === cell 25
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


## === cell 26
train_tokens = tokenizer.batch_encode_plus(train_df["comment_text"].tolist(),
                                           max_length = 200,
                                           pad_to_max_length=True,
                                           truncation=True,
                                           return_token_type_ids=False
                                           )

val_tokens = tokenizer.batch_encode_plus(val_df["comment_text"].tolist(),
                                         max_length = 200,
                                         pad_to_max_length=True,
                                         truncation=True,
                                         return_token_type_ids=False
                                         )

test_tokens = tokenizer.batch_encode_plus(test_df["comment_text"].tolist(),
                                          max_length = 200,
                                          pad_to_max_length=True,
                                          truncation=True,
                                          return_token_type_ids=False
                                          )


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/735523764.py in <cell line: 0>()
      1 # tokenize and encode sequences in the training set
----> 2 train_tokens = tokenizer.batch_encode_plus(train_df["comment_text"].tolist(),
      3                                            max_length = 200,
      4                                            pad_to_max_length=True,
      5                                            truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'pad_to_max_length'

## === cell 27
train_seq = torch.tensor(train_tokens['input_ids'])
train_mask = torch.tensor(train_tokens['attention_mask'])
train_y = torch.tensor(np.array(train_df[categories]))  

val_seq = torch.tensor(val_tokens['input_ids'])
val_mask = torch.tensor(val_tokens['attention_mask'])
val_y = torch.tensor(np.array(val_df[categories]))

test_seq = torch.tensor(test_tokens['input_ids'])
test_mask = torch.tensor(test_tokens['attention_mask'])
test_y = torch.tensor(np.array(test_df[categories]))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/130842253.py in <cell line: 0>()
      1 ## convert lists to tensors
----> 2 train_seq = torch.tensor(train_tokens['input_ids'])
      3 train_mask = torch.tensor(train_tokens['attention_mask'])
      4 # change from a list of 1 & 0 to array then to tensor
      5 train_y = torch.tensor(np.array(train_df[categories]))

NameError: name 'train_tokens' is not defined

## === cell 28
train_y


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4143882592.py in <cell line: 0>()
----> 1 train_y

NameError: name 'train_y' is not defined

## === cell 29
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2397997313.py in <cell line: 0>()
      1 # wrap tensors
----> 2 train_data = TensorDataset(train_seq, train_mask, train_y)
      3 # sampler for sampling the data during training
      4 train_sampler = RandomSampler(train_data)
      5 

NameError: name 'TensorDataset' is not defined

## === cell 30
batch_size = 32

train_dataloader = DataLoader(train_data, 
                              sampler=train_sampler, 
                              batch_size=batch_size,
                              )

val_dataloader = DataLoader(val_data, 
                            sampler = val_sampler, 
                            batch_size=batch_size,
                            )


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3707784815.py in <cell line: 0>()
      3 
      4 # dataLoader for train set
----> 5 train_dataloader = DataLoader(train_data, 
      6                               sampler=train_sampler,
      7                               batch_size=batch_size,

NameError: name 'DataLoader' is not defined

## === cell 31
for step,batch in enumerate(train_dataloader):
    break
print(batch[0])
print(batch[1])
print(batch[2])


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/257988196.py in <cell line: 0>()
----> 1 for step,batch in enumerate(train_dataloader):
      2     break
      3 print(batch[0])
      4 print(batch[1])
      5 print(batch[2])

NameError: name 'train_dataloader' is not defined

## === cell 32
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 6)





## === cell 33
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


## === cell 34
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(),
                  lr = LEARN_RATE, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )





## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2750349472.py in <cell line: 0>()
      2 # Will be training for the whole network instead of just finetuning the weight of the head layer
      3 LEARN_RATE = 3e-5
----> 4 optimizer = AdamW(model.parameters(),
      5                   lr = LEARN_RATE, # args.learning_rate - default is 5e-5, our notebook had 2e-5
      6                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

NameError: name 'AdamW' is not defined

## === cell 35
epochs = 3

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # default value
                                            num_training_steps = total_steps
                                            )


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3933978979.py in <cell line: 0>()
      5 # Total number of training steps is [number of batches] x [number of epochs].
      6 # (Note that this is not the same as the number of training samples).
----> 7 total_steps = len(train_dataloader) * epochs
      8 
      9 # Create the learning rate scheduler.

NameError: name 'train_dataloader' is not defined

## === cell 36
criterion = nn.BCEWithLogitsLoss()




## === cell 37
def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: 
        y_pred = y_pred.sigmoid()
    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()


## === cell 38
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))


## === cell 39
training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        output = model(b_input_ids, 
                       attention_mask=b_input_mask, 
                       )
        logits = output.logits
        loss = criterion(logits, b_labels.float())

        total_train_loss += loss.item()
        
        total_train_accuracy += accuracy_thresh(logits,b_labels.float())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    train_accuracy = total_train_accuracy / len(train_df)
    print("  Accuracy: {0:.5f}".format(train_accuracy))
    
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        

            output = model(b_input_ids, 
                           attention_mask=b_input_mask, 
                           )
            
            logits = output.logits
            loss = criterion(logits, b_labels.float())
            
        total_eval_loss += loss.item()
        
        total_eval_accuracy += accuracy_thresh(logits,b_labels.float())

        logits = logits.detach().cpu().numpy() # maybe irrelevant over here
        label_ids = b_labels.to('cpu').numpy() # maybe irrelevant over here

    val_accuracy = total_eval_accuracy / len(val_df)
    print("  Accuracy: {0:.5f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Accur': train_accuracy,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/159868623.py in <cell line: 0>()
     29 
     30     # For each batch of training data...
---> 31     for step, batch in enumerate(train_dataloader):
     32         # Progress update every 40 batches.
     33         if step % 40 == 0 and not step == 0:

NameError: name 'train_dataloader' is not defined

## === cell 40
pd.set_option('precision', 5)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
OptionError                               Traceback (most recent call last)
/tmp/ipykernel_11/3152575319.py in <cell line: 0>()
      1 # Display floats with two decimal places.
----> 2 pd.set_option('precision', 5)
      3 # Create a DataFrame from our training statistics.
      4 df_stats = pd.DataFrame(data=training_stats)
      5 # Use the 'epoch' as the row index.

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in __call__(self, *args, **kwds)
    272 
    273     def __call__(self, *args, **kwds) -> T:
--> 274         return self.__func__(*args, **kwds)
    275 
    276     # error: Signature of "__doc__" incompatible with supertype "object"

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in _set_option(*args, **kwargs)
    165 
    166     for k, v in zip(args[::2], args[1::2]):
--> 167         key = _get_single_key(k, silent)
    168 
    169         o = _get_registered_option(key)

/usr/local/lib/python3.11/dist-packages/pandas/_config/config.py in _get_single_key(pat, silent)
    132         raise OptionError(f"No such keys(s): {repr(pat)}")
    133     if len(keys) > 1:
--> 134         raise OptionError("Pattern matched multiple keys")
    135     key = keys[0]
    136 

OptionError: Pattern matched multiple keys

## === cell 41
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1606504060.py in <cell line: 0>()
      1 # Plot the learning curve.
----> 2 plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
      3 plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
      4 
      5 # Label the plot.

NameError: name 'df_stats' is not defined

## === cell 42
plt.plot(df_stats['Training Accur'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Accur.'], 'g-o', label="Validation")

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1341991535.py in <cell line: 0>()
      1 # Plot the accuracy curve.
----> 2 plt.plot(df_stats['Training Accur'], 'b-o', label="Training")
      3 plt.plot(df_stats['Valid. Accur.'], 'g-o', label="Validation")
      4 
      5 # Label the plot.

NameError: name 'df_stats' is not defined

## === cell 43
thresh = 0.4
with torch.no_grad():
    outputs = model(test_seq[0:1000].to(device), test_mask[0:1000].to(device))
    pred_probs = torch.sigmoid(outputs.logits)
    pred_probs = pred_probs.cpu().detach().numpy()


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1008802512.py in <cell line: 0>()
      2 thresh = 0.4
      3 with torch.no_grad():
----> 4     outputs = model(test_seq[0:1000].to(device), test_mask[0:1000].to(device))
      5     pred_probs = torch.sigmoid(outputs.logits)
      6     pred_probs = pred_probs.cpu().detach().numpy()

NameError: name 'test_seq' is not defined

## === cell 44
pred_probs


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2658514414.py in <cell line: 0>()
----> 1 pred_probs

NameError: name 'pred_probs' is not defined

## === cell 45
y_pred = (pred_probs > thresh).astype(int)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/857341935.py in <cell line: 0>()
----> 1 y_pred = (pred_probs > thresh).astype(int)

NameError: name 'pred_probs' is not defined

## === cell 46
y_pred


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3830458035.py in <cell line: 0>()
----> 1 y_pred

NameError: name 'y_pred' is not defined

## === cell 47
y_true = np.array(test_df[categories])[0:1000]


## === cell 48
from sklearn.metrics import hamming_loss, accuracy_score


## === cell 49
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    
    Take in np.array for y_true and y_pred. E.g.
    y_true = np.array([[0,1,0],
                       [0,1,1],
                       [1,0,1],
                       [0,0,1]])

    y_pred = np.array([[0,1,1],
                       [0,1,1],
                       [0,1,0],
                       [0,0,0]])
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


## === cell 50
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2115318445.py in <cell line: 0>()
----> 1 print("accuracy_score:", accuracy_score(y_true, y_pred))
      2 print("Hamming_score:", hamming_score(y_true, y_pred))
      3 print("Hamming_loss:", hamming_loss(y_true, y_pred))

NameError: name 'y_pred' is not defined

## === cell 51
PATH = "./toxic_distilBERT_multilabel"
torch.save(model.state_dict(), PATH)


## === cell 52
test_csv.head()


## === cell 53
len(test_csv)


## === cell 54
sub_tokens = tokenizer.batch_encode_plus(test_csv["comment_text"].tolist(),
                                         max_length = 200,
                                         pad_to_max_length=True,
                                         truncation=True,
                                         return_token_type_ids=False
                                         )


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1332018539.py in <cell line: 0>()
      1 # tokenize and encode sequences in the actual test set
----> 2 sub_tokens = tokenizer.batch_encode_plus(test_csv["comment_text"].tolist(),
      3                                          max_length = 200,
      4                                          pad_to_max_length=True,
      5                                          truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'pad_to_max_length'

## === cell 55
sub_seq = torch.tensor(sub_tokens['input_ids'])
sub_mask = torch.tensor(sub_tokens['attention_mask'])


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/129157109.py in <cell line: 0>()
----> 1 sub_seq = torch.tensor(sub_tokens['input_ids'])
      2 sub_mask = torch.tensor(sub_tokens['attention_mask'])

NameError: name 'sub_tokens' is not defined

## === cell 56
sub_data = TensorDataset(sub_seq, sub_mask)


## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/307800067.py in <cell line: 0>()
----> 1 sub_data = TensorDataset(sub_seq, sub_mask)

NameError: name 'TensorDataset' is not defined

## === cell 57
sub_dataloader = DataLoader(sub_data, 
                            batch_size=batch_size)


## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1426846832.py in <cell line: 0>()
      1 # dataLoader for validation set
----> 2 sub_dataloader = DataLoader(sub_data, 
      3                             batch_size=batch_size)

NameError: name 'DataLoader' is not defined

## === cell 58
t0 = time.time()

for step, batch in enumerate(sub_dataloader):
    if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(sub_dataloader), elapsed))
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, b_input_mask)
        pred_probs = torch.sigmoid(outputs.logits)
        if step == 0:
            predictions = pred_probs.cpu().detach().numpy()
        else:
            predictions = np.append(predictions, pred_probs.cpu().detach().numpy(), axis=0)


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2500379690.py in <cell line: 0>()
      2 t0 = time.time()
      3 
----> 4 for step, batch in enumerate(sub_dataloader):
      5     # Progress update every 40 batches.
      6     if step % 40 == 0 and not step == 0:

NameError: name 'sub_dataloader' is not defined

## === cell 59
predictions_df = pd.DataFrame(predictions, columns = categories)


## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174917913.py in <cell line: 0>()
----> 1 predictions_df = pd.DataFrame(predictions, columns = categories)

NameError: name 'predictions' is not defined

## === cell 60
len(predictions_df)


## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2169994269.py in <cell line: 0>()
----> 1 len(predictions_df)

NameError: name 'predictions_df' is not defined

## === cell 61
submission = pd.concat([test_csv["id"], predictions_df], axis=1)


## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2182598802.py in <cell line: 0>()
----> 1 submission = pd.concat([test_csv["id"], predictions_df], axis=1)

NameError: name 'predictions_df' is not defined

## === cell 62
submission.head()


## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4096176616.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 63
submission.to_csv('submission.csv', index=False, header=True)


## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3903519738.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False, header=True)

NameError: name 'submission' is not defined

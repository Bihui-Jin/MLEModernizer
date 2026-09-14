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

0.98418

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
print("Using input dir:", INPUT_DIR)
print("Files:", sorted(os.listdir(INPUT_DIR))[:10])



## === cell 1
import time
import datetime
from sklearn.model_selection import train_test_split



## === cell 2
TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

usecols_train = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
usecols_test = ["id", "comment_text"]

df = pd.read_csv(TRAIN_PATH, usecols=usecols_train, engine="pyarrow")
test_csv = pd.read_csv(TEST_PATH, usecols=usecols_test, engine="pyarrow")

print(df.columns)
print(df.shape)

target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ArrowInvalid                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    265         try:
--> 266             table = pyarrow_csv.read_csv(
    267                 self.src,

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.pyarrow_internal_check_status()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.check_status()

ArrowInvalid: CSV parse error: Expected 8 columns, got 2: Archive-5, have 44 printed pages of A4 size.

The above exception was the direct cause of the following exception:

ParserError                               Traceback (most recent call last)
/tmp/ipykernel_11/2289068862.py in <cell line: 0>()
     16 usecols_test = ["id", "comment_text"]
     17 
---> 18 df = pd.read_csv(TRAIN_PATH, usecols=usecols_train, engine="pyarrow")
     19 test_csv = pd.read_csv(TEST_PATH, usecols=usecols_test, engine="pyarrow")
     20 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1909             try:
   1910                 # error: "ParserBase" has no attribute "read"
-> 1911                 df = self._engine.read()  # type: ignore[attr-defined]
   1912             except Exception:
   1913                 self.close()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    271             )
    272         except pa.ArrowInvalid as e:
--> 273             raise ParserError(e) from e
    274 
    275         dtype_backend = self.kwds["dtype_backend"]

ParserError: CSV parse error: Expected 8 columns, got 2: Archive-5, have 44 printed pages of A4 size.

## === cell 3
pass



## === cell 4
print(target_col)
print(feature_col)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/971347168.py in <cell line: 0>()
----> 1 print(target_col)
      2 print(feature_col)
      3 

NameError: name 'target_col' is not defined

## === cell 5
df.dtypes



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1718546869.py in <cell line: 0>()
----> 1 df.dtypes
      2 

NameError: name 'df' is not defined

## === cell 6
pass



## === cell 7
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

condition = (
    (df["toxic"] == 1)
    | (df["severe_toxic"] == 1)
    | (df["obscene"] == 1)
    | (df["threat"] == 1)
    | (df["insult"] == 1)
    | (df["identity_hate"] == 1)
)

df.loc[condition, "y"] = 1
df.loc[~condition, "y"] = 0
df["y"] = df["y"].astype(int)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3638055077.py in <cell line: 0>()
      2 
      3 condition = (
----> 4     (df["toxic"] == 1)
      5     | (df["severe_toxic"] == 1)
      6     | (df["obscene"] == 1)

NameError: name 'df' is not defined

## === cell 8
agg_df = df["y"].value_counts(normalize=True).rename("Proportion").reset_index()
agg_df.columns = ["y", "Proportion"]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/836918868.py in <cell line: 0>()
----> 1 agg_df = df["y"].value_counts(normalize=True).rename("Proportion").reset_index()
      2 agg_df.columns = ["y", "Proportion"]
      3 

NameError: name 'df' is not defined

## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
df = df.rename(columns={"id": "idx"})



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3417831439.py in <cell line: 0>()
----> 1 df = df.rename(columns={"id": "idx"})
      2 

NameError: name 'df' is not defined

## === cell 15
df.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344414169.py in <cell line: 0>()
----> 1 df.head()
      2 

NameError: name 'df' is not defined

## === cell 16
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler



## === cell 17
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = seed_value + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_value)



## === cell 18
train_val_df, test_df = train_test_split(
    df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2924872649.py in <cell line: 0>()
      1 train_val_df, test_df = train_test_split(
----> 2     df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value
      3 )
      4 

NameError: name 'df' is not defined

## === cell 19
train_df, val_df = train_test_split(
    train_val_df[["idx", "comment_text"] + categories],
    test_size=0.25,
    random_state=seed_value,
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2113462712.py in <cell line: 0>()
      1 train_df, val_df = train_test_split(
----> 2     train_val_df[["idx", "comment_text"] + categories],
      3     test_size=0.25,
      4     random_state=seed_value,
      5 )

NameError: name 'train_val_df' is not defined

## === cell 20
print(
    f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively."
)
print(
    f"Proportion of train, validation and test are {round(len(train_df)/len(df), 2)}, {round(len(val_df)/len(df), 2)}, {round(len(test_df)/len(df), 2)} respectively."
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/524085744.py in <cell line: 0>()
      1 print(
----> 2     f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively."
      3 )
      4 print(
      5     f"Proportion of train, validation and test are {round(len(train_df)/len(df), 2)}, {round(len(val_df)/len(df), 2)}, {round(len(test_df)/len(df), 2)} respectively."

NameError: name 'train_df' is not defined

## === cell 21
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)

test_df.reset_index(inplace=True)
test_df.drop("index", axis=1, inplace=True)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427603471.py in <cell line: 0>()
----> 1 train_df.reset_index(inplace=True)
      2 train_df.drop("index", axis=1, inplace=True)
      3 
      4 val_df.reset_index(inplace=True)
      5 val_df.drop("index", axis=1, inplace=True)

NameError: name 'train_df' is not defined

## === cell 22
train_df.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/577764774.py in <cell line: 0>()
----> 1 train_df.head()
      2 

NameError: name 'train_df' is not defined

## === cell 23
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)



## === cell 24
MAX_LEN = 200


class ToxicDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int, label_cols=None):
        self.texts = frame["comment_text"].astype(str).values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = label_cols
        if label_cols is not None:
            self.labels = frame[label_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        if self.labels is None:
            return input_ids, attention_mask
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, y


train_data = ToxicDataset(train_df, tokenizer, MAX_LEN, label_cols=categories)
val_data = ToxicDataset(val_df, tokenizer, MAX_LEN, label_cols=categories)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433733211.py in <cell line: 0>()
     36 
     37 
---> 38 train_data = ToxicDataset(train_df, tokenizer, MAX_LEN, label_cols=categories)
     39 val_data = ToxicDataset(val_df, tokenizer, MAX_LEN, label_cols=categories)
     40 

NameError: name 'train_df' is not defined

## === cell 25
train_sampler = RandomSampler(train_data)
val_sampler = SequentialSampler(val_data)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/812466007.py in <cell line: 0>()
      1 # Keep the same sampling semantics.
----> 2 train_sampler = RandomSampler(train_data)
      3 val_sampler = SequentialSampler(val_data)
      4 

NameError: name 'train_data' is not defined

## === cell 26
batch_size = 32
num_workers = min(4, os.cpu_count() or 1)

train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)

val_dataloader = DataLoader(
    val_data,
    sampler=val_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2685504124.py in <cell line: 0>()
      5 
      6 train_dataloader = DataLoader(
----> 7     train_data,
      8     sampler=train_sampler,
      9     batch_size=batch_size,

NameError: name 'train_data' is not defined

## === cell 27
for step, batch in enumerate(train_dataloader):
    break
print(batch[0])
print(batch[1])
print(batch[2])



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3805185.py in <cell line: 0>()
----> 1 for step, batch in enumerate(train_dataloader):
      2     break
      3 print(batch[0])
      4 print(batch[1])
      5 print(batch[2])

NameError: name 'train_dataloader' is not defined

## === cell 28
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=6, problem_type="multi_label_classification"
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 29
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



## === cell 30
if torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not available; continuing without it:", repr(e))



## === cell 31
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)



## === cell 32
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1066257881.py in <cell line: 0>()
      1 epochs = 3
----> 2 total_steps = len(train_dataloader) * epochs
      3 
      4 scheduler = get_linear_schedule_with_warmup(
      5     optimizer, num_warmup_steps=0, num_training_steps=total_steps

NameError: name 'train_dataloader' is not defined

## === cell 33
criterion = nn.BCEWithLogitsLoss()




## === cell 34
def accuracy_thresh(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    acc_per_sample = ((y_pred > thresh) == y_true.bool()).float().mean(dim=1)
    return acc_per_sample.sum().item()




## === cell 35
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 36
training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0.0
    total_train_accuracy = 0.0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 200 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device, non_blocking=True)
        b_input_mask = batch[1].to(device, non_blocking=True)
        b_labels = batch[2].to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        output = model(
            input_ids=b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
        )
        logits = output.logits
        loss = criterion(logits, b_labels.float())

        total_train_loss += loss.item()
        total_train_accuracy += accuracy_thresh(logits, b_labels.float())

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

    total_eval_accuracy = 0.0
    total_eval_loss = 0.0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device, non_blocking=True)
        b_input_mask = batch[1].to(device, non_blocking=True)
        b_labels = batch[2].to(device, non_blocking=True)

        with torch.no_grad():
            output = model(
                input_ids=b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )
            logits = output.logits
            loss = criterion(logits, b_labels.float())

        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy_thresh(logits, b_labels.float())

    val_accuracy = total_eval_accuracy / len(val_df)
    print("  Accuracy: {0:.5f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Training Accur": train_accuracy,
            "Valid. Accur.": val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2296042475.py in <cell line: 0>()
     13     model.train()
     14 
---> 15     for step, batch in enumerate(train_dataloader):
     16         if step % 200 == 0 and not step == 0:
     17             elapsed = format_time(time.time() - t0)

NameError: name 'train_dataloader' is not defined

## === cell 37
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index("epoch")
df_stats



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4181818794.py in <cell line: 0>()
      1 pd.set_option("display.precision", 5)
      2 df_stats = pd.DataFrame(data=training_stats)
----> 3 df_stats = df_stats.set_index("epoch")
      4 df_stats
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in set_index(self, keys, drop, append, inplace, verify_integrity)
   6120 
   6121         if missing:
-> 6122             raise KeyError(f"None of {missing} are in the columns")
   6123 
   6124         if inplace:

KeyError: "None of ['epoch'] are in the columns"

## === cell 38
pass



## === cell 39
pass



## === cell 40
pass



## === cell 41
pass



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
PATH = "./toxic_BERT_multilabel.pt"
torch.save(model.state_dict(), PATH)



## === cell 49
test_csv.head()



## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3207365892.py in <cell line: 0>()
----> 1 test_csv.head()
      2 

NameError: name 'test_csv' is not defined

## === cell 50
len(test_csv)



## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1341762266.py in <cell line: 0>()
----> 1 len(test_csv)
      2 

NameError: name 'test_csv' is not defined

## === cell 51
sub_data = ToxicDataset(test_csv, tokenizer, MAX_LEN, label_cols=None)



## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2161332910.py in <cell line: 0>()
      1 # Speed: on-the-fly tokenization for submission too (avoids materializing 552k x 200 tensors).
      2 # Correctness: identical tokenizer params and batching; outputs are the same up to negligible FP diffs.
----> 3 sub_data = ToxicDataset(test_csv, tokenizer, MAX_LEN, label_cols=None)
      4 

NameError: name 'test_csv' is not defined

## === cell 52
sub_batch_size = 64 if torch.cuda.is_available() else batch_size

sub_dataloader = DataLoader(
    sub_data,
    batch_size=sub_batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
)



## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2123190151.py in <cell line: 0>()
      4 
      5 sub_dataloader = DataLoader(
----> 6     sub_data,
      7     batch_size=sub_batch_size,
      8     num_workers=num_workers,

NameError: name 'sub_data' is not defined

## === cell 53
pass



## === cell 54
pass



## === cell 55
t0 = time.time()
model.eval()

pred_list = []
with torch.inference_mode():
    for step, batch in enumerate(sub_dataloader):
        if step % 200 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(sub_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device, non_blocking=True)
        b_input_mask = batch[1].to(device, non_blocking=True)

        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()
        pred_list.append(pred_probs)

predictions = np.concatenate(pred_list, axis=0)
print("Predictions shape:", predictions.shape)



## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894935559.py in <cell line: 0>()
      5 # Speed: inference_mode is faster than no_grad and is semantically equivalent for inference.
      6 with torch.inference_mode():
----> 7     for step, batch in enumerate(sub_dataloader):
      8         if step % 200 == 0 and not step == 0:
      9             elapsed = format_time(time.time() - t0)

NameError: name 'sub_dataloader' is not defined

## === cell 56
predictions_df = pd.DataFrame(predictions, columns=categories)



## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3118710355.py in <cell line: 0>()
----> 1 predictions_df = pd.DataFrame(predictions, columns=categories)
      2 

NameError: name 'predictions' is not defined

## === cell 57
len(predictions_df)



## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4139851960.py in <cell line: 0>()
----> 1 len(predictions_df)
      2 

NameError: name 'predictions_df' is not defined

## === cell 58
submission = pd.concat([test_csv["id"], predictions_df], axis=1)



## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/724051922.py in <cell line: 0>()
----> 1 submission = pd.concat([test_csv["id"], predictions_df], axis=1)
      2 

NameError: name 'test_csv' is not defined

## === cell 59
submission.head()



## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 60
submission = submission[["id"] + categories]
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3855253850.py in <cell line: 0>()
----> 1 submission = submission[["id"] + categories]
      2 submission.to_csv("submission.csv", index=False, header=True)
      3 print("Wrote submission.csv with shape:", submission.shape)
      4 print(submission.head())

NameError: name 'submission' is not defined

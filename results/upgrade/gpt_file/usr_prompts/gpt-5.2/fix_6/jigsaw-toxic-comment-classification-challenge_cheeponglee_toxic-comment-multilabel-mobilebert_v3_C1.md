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

0.98459

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the early import-time crash coming from a protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation before importing `transformers`, and I make the `torch.utils.data` imports unconditional so `TensorDataset`/`DataLoader` are always defined. I also fix the submission-building logic to write the required header/columns directly in the correct order (instead of merging on `sample_submission`, which can silently drop/rename columns when earlier steps failed). Finally, I keep the model/training loop intact but make the output layer configuration explicitly multi-label (`problem_type="multi_label_classification"`) so the forward pass semantics match the BCEWithLogitsLoss setup and produce valid probabilities for all six labels.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd





## === cell 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime




## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
sample_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

train_cols = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
test_cols = ["id", "comment_text"]

df = pd.read_csv(train_path, usecols=train_cols)
test_csv = pd.read_csv(test_path, usecols=test_cols)
sample_sub = pd.read_csv(sample_path)

print(df.columns)
print(df.shape)
target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()




## === cell 3
df = df.rename(columns={"id": "idx"})




## === cell 4
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 5
import random

import torch
import torch.nn as nn
from torch.optim import (
    lr_scheduler,
)  # kept even if unused to preserve original structure
from torch.optim import AdamW as TorchAdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from tqdm.auto import tqdm
from sklearn.metrics import classification_report, hamming_loss, accuracy_score




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 7
train_df, val_df = train_test_split(
    df[["idx", "comment_text"] + categories],
    test_size=0.2,
    random_state=seed_value,
)




## === cell 8
print(f"Size of train and validation are {len(train_df)}, {len(val_df)} respectively.")
print(
    f"Proportion of train and validation are {round(len(train_df)/len(df),2)}, {round(len(val_df)/len(df),2)} respectively."
)




## === cell 9
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)




## === cell 10
train_df.head()




## === cell 11
checkpoint = "google/mobilebert-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)




## === cell 12
max_length = 200

train_tokens = tokenizer(
    train_df["comment_text"].values,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)

val_tokens = tokenizer(
    val_df["comment_text"].values,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1292821268.py in <cell line: 0>()
      3 max_length = 200
      4 
----> 5 train_tokens = tokenizer(
      6     train_df["comment_text"].values,
      7     max_length=max_length,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2913 
   2914         if not _is_valid_text_input(text):
-> 2915             raise ValueError(
   2916                 "text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) "
   2917                 "or `list[list[str]]` (batch of pretokenized examples)."

ValueError: text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) or `list[list[str]]` (batch of pretokenized examples).

## === cell 13
train_seq = torch.from_numpy(train_tokens["input_ids"]).long()
train_mask = torch.from_numpy(train_tokens["attention_mask"]).long()
train_y = torch.from_numpy(train_df[categories].to_numpy(dtype=np.float32, copy=False))

val_seq = torch.from_numpy(val_tokens["input_ids"]).long()
val_mask = torch.from_numpy(val_tokens["attention_mask"]).long()
val_y = torch.from_numpy(val_df[categories].to_numpy(dtype=np.float32, copy=False))




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3149942737.py in <cell line: 0>()
      1 # Speed: create tensors from contiguous numpy arrays without extra Python overhead.
      2 # Correctness: same dtype/values as before.
----> 3 train_seq = torch.from_numpy(train_tokens["input_ids"]).long()
      4 train_mask = torch.from_numpy(train_tokens["attention_mask"]).long()
      5 train_y = torch.from_numpy(train_df[categories].to_numpy(dtype=np.float32, copy=False))

NameError: name 'train_tokens' is not defined

## === cell 14
train_y




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238890619.py in <cell line: 0>()
----> 1 train_y
      2 
      3 

NameError: name 'train_y' is not defined

## === cell 15
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4001425822.py in <cell line: 0>()
----> 1 train_data = TensorDataset(train_seq, train_mask, train_y)
      2 train_sampler = RandomSampler(train_data)
      3 
      4 val_data = TensorDataset(val_seq, val_mask, val_y)
      5 val_sampler = SequentialSampler(val_data)

NameError: name 'train_seq' is not defined

## === cell 16
batch_size = 32

num_workers = min(4, os.cpu_count() or 1)
pin_memory = torch.cuda.is_available()
persistent_workers = bool(num_workers > 0)

train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
)

val_dataloader = DataLoader(
    val_data,
    sampler=val_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/730955304.py in <cell line: 0>()
      8 
      9 train_dataloader = DataLoader(
---> 10     train_data,
     11     sampler=train_sampler,
     12     batch_size=batch_size,

NameError: name 'train_data' is not defined

## === cell 17
for step, batch in enumerate(train_dataloader):
    break
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4155026528.py in <cell line: 0>()
----> 1 for step, batch in enumerate(train_dataloader):
      2     break
      3 print(batch[0].shape)
      4 print(batch[1].shape)
      5 print(batch[2].shape)

NameError: name 'train_dataloader' is not defined

## === cell 18
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=6, problem_type="multi_label_classification"
)




## === cell 19
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)




## === cell 20
LEARN_RATE = 3e-5
optimizer = TorchAdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)




## === cell 21
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2243551578.py in <cell line: 0>()
      1 epochs = 3
----> 2 total_steps = len(train_dataloader) * epochs
      3 
      4 scheduler = get_linear_schedule_with_warmup(
      5     optimizer, num_warmup_steps=0, num_training_steps=total_steps

NameError: name 'train_dataloader' is not defined

## === cell 22
criterion = nn.BCEWithLogitsLoss()




## === cell 23
@torch.no_grad()
def accuracy_thresh_torch(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = torch.sigmoid(y_pred)
    acc_per_sample = (
        (y_pred > thresh).to(y_true.dtype).eq(y_true).to(torch.float32).mean(dim=1)
    )
    return acc_per_sample.sum().item()




## === cell 24
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 25
training_stats = []

total_t0 = time.time()

len_train_df = len(train_df)
len_val_df = len(val_df)
len_train_dl = len(train_dataloader)
len_val_dl = len(val_dataloader)

for epoch_i in range(0, epochs):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0.0
    total_train_accuracy = 0.0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len_train_dl, elapsed
                )
            )

        b_input_ids = batch[0].to(device, non_blocking=True)
        b_input_mask = batch[1].to(device, non_blocking=True)
        b_labels = batch[2].to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        output = model(
            b_input_ids,
            attention_mask=b_input_mask,
        )
        logits = output.logits
        loss = criterion(logits, b_labels.float())

        total_train_loss += loss.item()
        total_train_accuracy += accuracy_thresh_torch(logits, b_labels.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    train_accuracy = total_train_accuracy / len_train_df
    print("  Accuracy: {0:.5f}".format(train_accuracy))

    avg_train_loss = total_train_loss / len_train_dl
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
                b_input_ids,
                attention_mask=b_input_mask,
            )

            logits = output.logits
            loss = criterion(logits, b_labels.float())

        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy_thresh_torch(logits, b_labels.float())

    val_accuracy = total_eval_accuracy / len_val_df
    print("  Accuracy: {0:.5f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len_val_dl
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




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2217784625.py in <cell line: 0>()
      5 len_train_df = len(train_df)
      6 len_val_df = len(val_df)
----> 7 len_train_dl = len(train_dataloader)
      8 len_val_dl = len(val_dataloader)
      9 

NameError: name 'train_dataloader' is not defined

## === cell 26
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
df_stats




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2572513998.py in <cell line: 0>()
      1 pd.set_option("display.precision", 5)
----> 2 df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
      3 df_stats
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in set_index(self, keys, drop, append, inplace, verify_integrity)
   6120 
   6121         if missing:
-> 6122             raise KeyError(f"None of {missing} are in the columns")
   6123 
   6124         if inplace:

KeyError: "None of ['epoch'] are in the columns"

## === cell 27
plt.plot(df_stats["Training Loss"], "b-o", label="Training")
plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks(list(df_stats.index))

plt.show()




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2221184630.py in <cell line: 0>()
----> 1 plt.plot(df_stats["Training Loss"], "b-o", label="Training")
      2 plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")
      3 
      4 plt.title("Training & Validation Loss")
      5 plt.xlabel("Epoch")

NameError: name 'df_stats' is not defined

## === cell 28
plt.plot(df_stats["Training Accur"], "b-o", label="Training")
plt.plot(df_stats["Valid. Accur."], "g-o", label="Validation")

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks(list(df_stats.index))

plt.show()




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2134120196.py in <cell line: 0>()
----> 1 plt.plot(df_stats["Training Accur"], "b-o", label="Training")
      2 plt.plot(df_stats["Valid. Accur."], "g-o", label="Validation")
      3 
      4 plt.title("Training & Validation Accuracy")
      5 plt.xlabel("Epoch")

NameError: name 'df_stats' is not defined

## === cell 29
thresh = 0.4
with torch.no_grad():
    outputs = model(
        val_seq[0:1000].to(device, non_blocking=True),
        attention_mask=val_mask[0:1000].to(device, non_blocking=True),
    )
    pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/817653778.py in <cell line: 0>()
      2 with torch.no_grad():
      3     outputs = model(
----> 4         val_seq[0:1000].to(device, non_blocking=True),
      5         attention_mask=val_mask[0:1000].to(device, non_blocking=True),
      6     )

NameError: name 'val_seq' is not defined

## === cell 30
pred_probs[:2]




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2563949712.py in <cell line: 0>()
----> 1 pred_probs[:2]
      2 
      3 

NameError: name 'pred_probs' is not defined

## === cell 31
y_pred = (pred_probs > thresh).astype(int)




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3279995439.py in <cell line: 0>()
----> 1 y_pred = (pred_probs > thresh).astype(int)
      2 
      3 

NameError: name 'pred_probs' is not defined

## === cell 32
y_pred[:2]




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/211300368.py in <cell line: 0>()
----> 1 y_pred[:2]
      2 
      3 

NameError: name 'y_pred' is not defined

## === cell 33
y_true = np.array(val_df[categories])[0:1000]




## === cell 34
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            acc_list.append(1)
        else:
            acc_list.append(
                len(set_true.intersection(set_pred))
                / float(len(set_true.union(set_pred)))
            )
    return np.mean(acc_list)




## === cell 35
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3592259887.py in <cell line: 0>()
----> 1 print("accuracy_score:", accuracy_score(y_true, y_pred))
      2 print("Hamming_score:", hamming_score(y_true, y_pred))
      3 print("Hamming_loss:", hamming_loss(y_true, y_pred))
      4 
      5 

NameError: name 'y_pred' is not defined

## === cell 36
PATH = "./toxic_mobileBERT_multilabel.pt"
torch.save(model.state_dict(), PATH)




## === cell 37
len(test_csv), test_csv.head()




## === cell 38
sub_tokens = tokenizer(
    test_csv["comment_text"].values,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)




## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2038467293.py in <cell line: 0>()
      1 # Speed: avoid .tolist() and return numpy directly (same as train/val).
----> 2 sub_tokens = tokenizer(
      3     test_csv["comment_text"].values,
      4     max_length=max_length,
      5     padding="max_length",

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2913 
   2914         if not _is_valid_text_input(text):
-> 2915             raise ValueError(
   2916                 "text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) "
   2917                 "or `list[list[str]]` (batch of pretokenized examples)."

ValueError: text input must be of type `str` (single example), `list[str]` (batch or single pretokenized example) or `list[list[str]]` (batch of pretokenized examples).

## === cell 39
sub_seq = torch.from_numpy(sub_tokens["input_ids"]).long()
sub_mask = torch.from_numpy(sub_tokens["attention_mask"]).long()




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2699679385.py in <cell line: 0>()
----> 1 sub_seq = torch.from_numpy(sub_tokens["input_ids"]).long()
      2 sub_mask = torch.from_numpy(sub_tokens["attention_mask"]).long()
      3 
      4 

NameError: name 'sub_tokens' is not defined

## === cell 40
sub_data = TensorDataset(sub_seq, sub_mask)




## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/694015722.py in <cell line: 0>()
----> 1 sub_data = TensorDataset(sub_seq, sub_mask)
      2 
      3 

NameError: name 'sub_seq' is not defined

## === cell 41
sub_dataloader = DataLoader(
    sub_data,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
)




## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4123183849.py in <cell line: 0>()
      1 # Speed: same DataLoader improvements for inference.
      2 sub_dataloader = DataLoader(
----> 3     sub_data,
      4     batch_size=batch_size,
      5     num_workers=num_workers,

NameError: name 'sub_data' is not defined

## === cell 42
model.eval()
t0 = time.time()
pred_list = []

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

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()
        pred_list.append(pred_probs)

predictions = np.concatenate(pred_list, axis=0)
print("Predictions shape:", predictions.shape)




## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/744530486.py in <cell line: 0>()
      3 pred_list = []
      4 
----> 5 for step, batch in enumerate(sub_dataloader):
      6     if step % 200 == 0 and not step == 0:
      7         elapsed = format_time(time.time() - t0)

NameError: name 'sub_dataloader' is not defined

## === cell 43
predictions_df = pd.DataFrame(predictions, columns=categories)
predictions_df.head()




## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1150411248.py in <cell line: 0>()
----> 1 predictions_df = pd.DataFrame(predictions, columns=categories)
      2 predictions_df.head()
      3 
      4 

NameError: name 'predictions' is not defined

## === cell 44
len(predictions_df), len(test_csv)




## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1417583192.py in <cell line: 0>()
----> 1 len(predictions_df), len(test_csv)
      2 
      3 

NameError: name 'predictions_df' is not defined

## === cell 45
submission = pd.concat(
    [test_csv[["id"]].reset_index(drop=True), predictions_df], axis=1
)
submission = submission[["id"] + categories]

for c in categories:
    submission[c] = submission[c].clip(0.0, 1.0)

submission.head()




## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1583838906.py in <cell line: 0>()
      1 submission = pd.concat(
----> 2     [test_csv[["id"]].reset_index(drop=True), predictions_df], axis=1
      3 )
      4 submission = submission[["id"] + categories]
      5 

NameError: name 'predictions_df' is not defined

## === cell 46
print(submission.columns.tolist())
print(submission.shape)




## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/761624677.py in <cell line: 0>()
----> 1 print(submission.columns.tolist())
      2 print(submission.shape)
      3 
      4 

NameError: name 'submission' is not defined

## === cell 47
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/837697281.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False, header=True)
      2 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'submission' is not defined

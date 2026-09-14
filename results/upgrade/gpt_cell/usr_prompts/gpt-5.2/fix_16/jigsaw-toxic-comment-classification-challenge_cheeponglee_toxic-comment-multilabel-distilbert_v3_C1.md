# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import os



## === cell 2
TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
usecols_train = ["id", "comment_text"] + categories
usecols_test = ["id", "comment_text"]

dtype_train = {c: "int8" for c in categories}
dtype_train.update({"id": "string", "comment_text": "string"})
dtype_test = {"id": "string", "comment_text": "string"}

df = pd.read_csv(TRAIN_PATH, usecols=usecols_train, dtype=dtype_train)
test_csv = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=dtype_test)

_test_labels_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip"
)
test_csv_labels = (
    pd.read_csv(_test_labels_path) if os.path.exists(_test_labels_path) else None
)

print(df.columns)
print(df.shape)
target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()



## === cell 3
for col in target_col:
    print(f"The unique value for {col} are {pd.unique(df[col])}")



## === cell 4
if test_csv_labels is None:
    print(
        "test_csv_labels is not available; skipping unique-value inspection for test labels."
    )
else:
    for col in target_col:
        print(f"The unique value for {col} are {pd.unique(test_csv_labels[col])}")



## === cell 5
print(target_col)
print(feature_col)



## === cell 6
df.dtypes



## === cell 7
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 8
condition = df[categories].any(axis=1)
df["y"] = condition.astype("int8")



## === cell 9
agg_df = df["y"].value_counts(normalize=True).rename("Proportion").reset_index()
agg_df.columns = ["y", "Proportion"]



## === cell 10
s = df["comment_text"].fillna("")
df["len"] = (s.str.count(r"\S+") + (s.str.contains(r"\S")).astype("int32") * 0).astype(
    "int32"
)



## === cell 11
print(f"Average length is {df['len'].mean()} while max length is {df['len'].max()}.")



## === cell 12
pass



## === cell 13
pass



## === cell 14
df = df.rename(columns={"id": "idx"})



## === cell 15
df.head()



## === cell 16
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import random
from sklearn.metrics import classification_report

os.environ["TOKENIZERS_PARALLELISM"] = "true"



## === cell 17
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass



## === cell 18
train_val_df, test_df = train_test_split(
    df[["idx", "comment_text"] + categories], test_size=0.2, random_state=seed_value
)



## === cell 19
(
    train_df,
    val_df,
) = train_test_split(
    train_val_df[["idx", "comment_text"] + categories],
    test_size=0.25,
    random_state=seed_value,
)



## === cell 20
print(
    f"Size of train, validation and test are {len(train_df)}, {len(val_df)}, {len(test_df)} respectively."
)
print(
    f"Proportion of train, validation and test are {round(len(train_df)/len(df),2)}, {round(len(val_df)/len(df),2)}, {round(len(test_df)/len(df),2)} respectively."
)



## === cell 21
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)



## === cell 22
train_df.head()



## === cell 23
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)




## === cell 24
def _batch_tokenize_texts(texts, tokenizer, max_length=200, batch_size=2048):
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # dynamic padding still handled by collator
        return_token_type_ids=False,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    return enc


class ToxicDataset(Dataset):
    def __init__(
        self, df, tokenizer, categories=None, max_length=200, pretokenize=True
    ):
        self.has_labels = categories is not None
        self.max_length = max_length
        self.texts = df["comment_text"].fillna("").astype(str).tolist()

        if pretokenize:
            self.encodings = _batch_tokenize_texts(
                self.texts, tokenizer=tokenizer, max_length=max_length
            )
            self.pretokenized = True
        else:
            self.tokenizer = tokenizer
            self.pretokenized = False

        if self.has_labels:
            self.labels = df[categories].to_numpy(dtype=np.int64, copy=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.pretokenized:
            enc = {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
            }
        else:
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )
        if self.has_labels:
            return enc, self.labels[idx]
        return enc


def collate_with_labels(features):
    encs, labels = zip(*features)
    batch = data_collator(list(encs))
    batch["labels"] = torch.as_tensor(np.stack(labels, axis=0), dtype=torch.long)
    return batch


def collate_no_labels(features):
    return data_collator(list(features))




## === cell 25
train_data = ToxicDataset(
    train_df, tokenizer, categories=categories, max_length=200, pretokenize=True
)
val_data = ToxicDataset(
    val_df, tokenizer, categories=categories, max_length=200, pretokenize=True
)
test_data = ToxicDataset(
    test_df, tokenizer, categories=categories, max_length=200, pretokenize=True
)



## === cell 26
batch_size = 32

num_workers = min(8, (os.cpu_count() or 2))

train_sampler = RandomSampler(train_data)
val_sampler = SequentialSampler(val_data)

train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_with_labels,
)

val_dataloader = DataLoader(
    val_data,
    sampler=val_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_with_labels,
)



## === cell 27
for step, batch in enumerate(train_dataloader):
    break
print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["labels"].shape)



## === cell 28
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)



## === cell 29
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass



## === cell 30
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)



## === cell 31
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)



## === cell 32
criterion = nn.BCEWithLogitsLoss()




## === cell 33
def accuracy_thresh(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    acc_per_sample = ((y_pred > thresh) == y_true.bool()).float().mean(dim=1)
    return acc_per_sample.sum().item()




## === cell 34
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 35
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

        b_input_ids = batch["input_ids"].to(device, non_blocking=True)
        b_input_mask = batch["attention_mask"].to(device, non_blocking=True)
        b_labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(
            b_input_ids,
            attention_mask=b_input_mask,
        )
        logits = output.logits
        loss = criterion(logits, b_labels.float())

        total_train_loss += loss.item()
        total_train_accuracy += accuracy_thresh(logits, b_labels)

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

    with torch.inference_mode():
        for batch in val_dataloader:
            b_input_ids = batch["input_ids"].to(device, non_blocking=True)
            b_input_mask = batch["attention_mask"].to(device, non_blocking=True)
            b_labels = batch["labels"].to(device, non_blocking=True)

            output = model(
                b_input_ids,
                attention_mask=b_input_mask,
            )
            logits = output.logits
            loss = criterion(logits, b_labels.float())

            total_eval_loss += loss.item()
            total_eval_accuracy += accuracy_thresh(logits, b_labels)

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



## === cell 36
pd.set_option("precision", 5)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index("epoch")
df_stats



## === cell 37
pass



## === cell 38
pass



## === cell 39
thresh = 0.4
test_preview_loader = DataLoader(
    test_data,
    batch_size=1000,
    sampler=SequentialSampler(test_data),
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_with_labels,
)

with torch.inference_mode():
    batch = next(iter(test_preview_loader))
    outputs = model(
        batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
    )
    pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()



## === cell 40
pred_probs



## === cell 41
y_pred = (pred_probs > thresh).astype(int)



## === cell 42
y_pred



## === cell 43
y_true = batch["labels"].cpu().numpy()



## === cell 44
from sklearn.metrics import hamming_loss, accuracy_score




## === cell 45
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        acc_list.append(tmp_a)
    return np.mean(acc_list)




## === cell 46
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))



## === cell 47
PATH = "./toxic_distilBERT_multilabel"
torch.save(model.state_dict(), PATH)



## === cell 48
test_csv.head()



## === cell 49
len(test_csv)



## === cell 50
sub_data = ToxicDataset(
    test_csv.rename(columns={"id": "idx"}),
    tokenizer,
    categories=None,
    max_length=200,
    pretokenize=True,
)



## === cell 51
sub_dataloader = DataLoader(
    sub_data,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_no_labels,
)



## === cell 52
t0 = time.time()
pred_chunks = []

model.eval()
with torch.inference_mode():
    for step, batch in enumerate(sub_dataloader):
        if step % 200 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(sub_dataloader), elapsed
                )
            )
        b_input_ids = batch["input_ids"].to(device, non_blocking=True)
        b_input_mask = batch["attention_mask"].to(device, non_blocking=True)

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        pred_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        pred_chunks.append(pred_probs)

predictions = np.concatenate(pred_chunks, axis=0)



## === cell 53
predictions_df = pd.DataFrame(predictions, columns=categories)



## === cell 54
len(predictions_df)



## === cell 55
submission = pd.concat([test_csv["id"], predictions_df], axis=1)



## === cell 56
submission.head()



## === cell 57
submission.to_csv("submission.csv", index=False, header=True)

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

read_csv_kwargs = dict(compression="zip")
try:
    df = pd.read_csv(
        TRAIN_PATH, usecols=usecols_train, engine="pyarrow", **read_csv_kwargs
    )
    test_csv = pd.read_csv(
        TEST_PATH, usecols=usecols_test, engine="pyarrow", **read_csv_kwargs
    )
except Exception:
    df = pd.read_csv(TRAIN_PATH, usecols=usecols_train, **read_csv_kwargs)
    test_csv = pd.read_csv(TEST_PATH, usecols=usecols_test, **read_csv_kwargs)

print(df.columns)
print(df.shape)

target_col = df.columns[2:]
feature_col = df.columns[1:2]
df.head()



## === cell 3
pass



## === cell 4
print(target_col)
print(feature_col)



## === cell 5
df.dtypes



## === cell 6
pass



## === cell 7
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

condition = df[categories].astype(np.int8).any(axis=1)

df.loc[condition, "y"] = 1
df.loc[~condition, "y"] = 0
df["y"] = df["y"].astype(int)



## === cell 8
agg_df = df["y"].value_counts(normalize=True).rename("Proportion").reset_index()
agg_df.columns = ["y", "Proportion"]
agg_df



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
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
df = df.rename(columns={"id": "idx"})



## === cell 20
df.head()



## === cell 21
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

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate



## === cell 22
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


def seed_worker(worker_id: int):
    worker_seed = seed_value + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_value)



## === cell 23
train_df, val_df = train_test_split(
    df[["idx", "comment_text"] + categories],
    test_size=0.2,
    random_state=seed_value,
)



## === cell 24
print(f"Size of train and validation are {len(train_df)}, {len(val_df)} respectively.")
print(
    f"Proportion of train and validation are {round(len(train_df)/len(df), 2)}, {round(len(val_df)/len(df), 2)} respectively."
)



## === cell 25
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)



## === cell 26
train_df.head()



## === cell 27
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)



## === cell 28
MAX_LEN = 200

from datasets import Dataset as HFDataset


def hf_tokenize_batch(batch):
    return tokenizer(
        batch["comment_text"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
    )


t_tok0 = time.time()

hf_train = HFDataset.from_pandas(
    train_df[["comment_text"] + categories], preserve_index=False
)
hf_val = HFDataset.from_pandas(
    val_df[["comment_text"] + categories], preserve_index=False
)

num_proc = min(4, os.cpu_count() or 1)

hf_train = hf_train.map(
    hf_tokenize_batch, batched=True, num_proc=num_proc, remove_columns=["comment_text"]
)
hf_val = hf_val.map(
    hf_tokenize_batch, batched=True, num_proc=num_proc, remove_columns=["comment_text"]
)

hf_train = hf_train.with_format(
    "torch", columns=["input_ids", "attention_mask"] + categories
)
hf_val = hf_val.with_format(
    "torch", columns=["input_ids", "attention_mask"] + categories
)

print(
    "Pre-tokenization (train+val) took:",
    str(datetime.timedelta(seconds=int(time.time() - t_tok0))),
)



## === cell 29
train_sampler = RandomSampler(hf_train, generator=g)
val_sampler = SequentialSampler(hf_val)

batch_size = 32
cpu_count = os.cpu_count() or 1
num_workers = min(4, max(0, cpu_count // 2))


def collate_hf_with_labels(batch):
    d = default_collate(batch)
    labels = torch.stack([d[c] for c in categories], dim=1).to(torch.float32)
    return d["input_ids"], d["attention_mask"], labels


train_dataloader = DataLoader(
    hf_train,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    collate_fn=collate_hf_with_labels,
)

val_dataloader = DataLoader(
    hf_val,
    sampler=val_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    collate_fn=collate_hf_with_labels,
)



## === cell 30
pass



## === cell 31
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=6, problem_type="multi_label_classification"
)



## === cell 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



## === cell 33
if torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not available; continuing without it:", repr(e))



## === cell 34
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)



## === cell 35
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)



## === cell 36
criterion = nn.BCEWithLogitsLoss()




## === cell 37
def accuracy_thresh(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    acc_per_sample = (y_pred > thresh).eq(y_true > 0.5).float().mean(dim=1)
    return acc_per_sample.sum().detach().item()




## === cell 38
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 39
training_stats = []
total_t0 = time.time()

train_size = len(train_df)
val_size = len(val_df)

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
        loss = criterion(logits, b_labels)

        total_train_loss += loss.item()
        total_train_accuracy += accuracy_thresh(logits, b_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    train_accuracy = total_train_accuracy / train_size
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

        with torch.inference_mode():
            output = model(
                input_ids=b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )
            logits = output.logits
            loss = criterion(logits, b_labels)

        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy_thresh(logits, b_labels)

    val_accuracy = total_eval_accuracy / val_size
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



## === cell 40
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index("epoch")
df_stats



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
pass



## === cell 49
pass



## === cell 50
pass



## === cell 51
PATH = "./toxic_BERT_multilabel.pt"
torch.save(model.state_dict(), PATH)



## === cell 52
test_csv.head()



## === cell 53
len(test_csv)



## === cell 54
t_tok1 = time.time()

hf_test = HFDataset.from_pandas(test_csv[["comment_text"]], preserve_index=False)
hf_test = hf_test.map(
    hf_tokenize_batch, batched=True, num_proc=num_proc, remove_columns=["comment_text"]
)
hf_test = hf_test.with_format("torch", columns=["input_ids", "attention_mask"])

print(
    "Pre-tokenization (test) took:",
    str(datetime.timedelta(seconds=int(time.time() - t_tok1))),
)



## === cell 55
sub_batch_size = 64 if torch.cuda.is_available() else batch_size


def collate_hf_no_labels(batch):
    d = default_collate(batch)
    return d["input_ids"], d["attention_mask"]


sub_dataloader = DataLoader(
    hf_test,
    batch_size=sub_batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    collate_fn=collate_hf_no_labels,
)



## === cell 56
pass



## === cell 57
pass



## === cell 58
t0 = time.time()
model.eval()

n_test = len(hf_test)
predictions = np.empty((n_test, 6), dtype=np.float32)

offset = 0
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
        pred_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()

        bs = pred_probs.shape[0]
        predictions[offset : offset + bs] = pred_probs
        offset += bs

print("Predictions shape:", predictions.shape)



## === cell 59
predictions_df = pd.DataFrame(predictions, columns=categories)



## === cell 60
len(predictions_df)



## === cell 61
submission = pd.concat([test_csv["id"], predictions_df], axis=1)



## === cell 62
submission.head()



## === cell 63
submission = submission[["id"] + categories]
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

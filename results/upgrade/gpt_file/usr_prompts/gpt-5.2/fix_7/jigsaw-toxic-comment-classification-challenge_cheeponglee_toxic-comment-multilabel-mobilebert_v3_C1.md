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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
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

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from tqdm.auto import tqdm
from sklearn.metrics import classification_report, hamming_loss, accuracy_score



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
def to_str_list(s: pd.Series):
    return s.fillna("").astype(str).tolist()


max_length = 200

train_tokens = tokenizer(
    to_str_list(train_df["comment_text"]),
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)

val_tokens = tokenizer(
    to_str_list(val_df["comment_text"]),
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)



## === cell 13
train_seq = torch.from_numpy(train_tokens["input_ids"]).long()
train_mask = torch.from_numpy(train_tokens["attention_mask"]).long()
train_y = torch.from_numpy(train_df[categories].to_numpy(dtype=np.float32, copy=False))

val_seq = torch.from_numpy(val_tokens["input_ids"]).long()
val_mask = torch.from_numpy(val_tokens["attention_mask"]).long()
val_y = torch.from_numpy(val_df[categories].to_numpy(dtype=np.float32, copy=False))



## === cell 14
train_y



## === cell 15
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)



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



## === cell 17
for step, batch in enumerate(train_dataloader):
    break
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)



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



## === cell 26
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
df_stats



## === cell 27
plt.plot(df_stats["Training Loss"], "b-o", label="Training")
plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks(list(df_stats.index))

plt.show()



## === cell 28
plt.plot(df_stats["Training Accur"], "b-o", label="Training")
plt.plot(df_stats["Valid. Accur."], "g-o", label="Validation")

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks(list(df_stats.index))

plt.show()



## === cell 29
thresh = 0.4
with torch.no_grad():
    outputs = model(
        val_seq[0:1000].to(device, non_blocking=True),
        attention_mask=val_mask[0:1000].to(device, non_blocking=True),
    )
    pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()



## === cell 30
pred_probs[:2]



## === cell 31
y_pred = (pred_probs > thresh).astype(int)



## === cell 32
y_pred[:2]



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



## === cell 36
PATH = "./toxic_mobileBERT_multilabel.pt"
torch.save(model.state_dict(), PATH)



## === cell 37
len(test_csv), test_csv.head()



## === cell 38
sub_tokens = tokenizer(
    to_str_list(test_csv["comment_text"]),
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors="np",
)



## === cell 39
sub_seq = torch.from_numpy(sub_tokens["input_ids"]).long()
sub_mask = torch.from_numpy(sub_tokens["attention_mask"]).long()



## === cell 40
sub_data = TensorDataset(sub_seq, sub_mask)



## === cell 41
sub_dataloader = DataLoader(
    sub_data,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
)



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



## === cell 43
predictions_df = pd.DataFrame(predictions, columns=categories)
predictions_df.head()



## === cell 44
len(predictions_df), len(test_csv)



## === cell 45
submission = pd.concat(
    [test_csv[["id"]].reset_index(drop=True), predictions_df], axis=1
)
submission = submission[["id"] + categories]

for c in categories:
    submission[c] = submission[c].clip(0.0, 1.0)

submission.head()



## === cell 46
print(submission.columns.tolist())
print(submission.shape)



## === cell 47
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head(2).to_string(index=False))

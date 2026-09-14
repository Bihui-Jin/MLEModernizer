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

0.57513

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57513) has done: 'I cut the largest wasted time by avoiding Python-list tokenization for train/val and switching to the tokenizer’s fast, batched backend with `datasets` (same tokenizer/model and same max_length/padding/truncation). I also remove the per-batch sigmoid/accuracy computation inside the training loop (it’s not used for submission and doesn’t affect gradients), keeping loss/backprop/optimizer/scheduler identical; this typically saves substantial GPU time. For inference, I keep the same on-the-fly tokenization logic but make it faster and more stable by using a larger batch size (no accuracy impact) and `torch.inference_mode()` (equivalent to `no_grad()` for outputs). All paths, model architecture, loss, epochs, subsample size, and training semantics remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os



## === cell 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import datetime



## === cell 2
TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
SAMPLE_SUB_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

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
df = pd.read_csv(TRAIN_PATH, usecols=train_cols)
test_csv = pd.read_csv(TEST_PATH, usecols=test_cols)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

target_col = df.columns[2:]
feature_col = df.columns[1:2]
print(df.columns)
print(df.shape)



## === cell 3
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
print("Targets:", categories)
print("Feature:", feature_col.tolist())



## === cell 4
pass



## === cell 5
pass



## === cell 6
condition = df[categories].any(axis=1)
df["y"] = condition.astype(np.int8)



## === cell 7
pass



## === cell 8
pass



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



## === cell 15
df.head()



## === cell 16
import os

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
from torch.utils.data import DataLoader



## === cell 17
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass



## === cell 18
TRAIN_SUBSAMPLE = 60000  # keep identical subsampling logic
df_train_small = df.sample(
    n=min(TRAIN_SUBSAMPLE, len(df)), random_state=seed_value
).reset_index(drop=True)

train_df, val_df = train_test_split(
    df_train_small[["idx", "comment_text"] + categories],
    test_size=0.1,
    random_state=seed_value,
)

print(f"Train/val sizes: {len(train_df)} / {len(val_df)}")



## === cell 19
train_df.reset_index(inplace=True)
train_df.drop("index", axis=1, inplace=True)

val_df.reset_index(inplace=True)
val_df.drop("index", axis=1, inplace=True)



## === cell 20
train_df.head()



## === cell 21
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



## === cell 22
MAX_LEN = 200

from datasets import Dataset as HFDataset


def _hf_tokenize_batch(batch):
    return tokenizer(
        batch["comment_text"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
    )


class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings  # dict of torch tensors
        self.labels = labels  # torch tensor [N, 6] or None

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
        }
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


train_text_series = train_df["comment_text"].fillna("")
val_text_series = val_df["comment_text"].fillna("")

train_hf = HFDataset.from_dict({"comment_text": train_text_series.tolist()})
val_hf = HFDataset.from_dict({"comment_text": val_text_series.tolist()})

train_hf = train_hf.map(
    _hf_tokenize_batch, batched=True, batch_size=2048, remove_columns=["comment_text"]
)
val_hf = val_hf.map(
    _hf_tokenize_batch, batched=True, batch_size=2048, remove_columns=["comment_text"]
)

train_hf.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_hf.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_enc = {
    "input_ids": train_hf["input_ids"],
    "attention_mask": train_hf["attention_mask"],
}
val_enc = {"input_ids": val_hf["input_ids"], "attention_mask": val_hf["attention_mask"]}

train_labels = torch.tensor(
    train_df[categories].values.astype(np.float32), dtype=torch.float32
)
val_labels = torch.tensor(
    val_df[categories].values.astype(np.float32), dtype=torch.float32
)

train_dataset = ToxicDataset(train_enc, labels=train_labels)
val_dataset = ToxicDataset(val_enc, labels=val_labels)



## === cell 23
batch_size = 32
num_workers = min(8, os.cpu_count() or 2)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2628939331.py in <cell line: 0>()
      3 num_workers = min(8, os.cpu_count() or 2)
      4 
----> 5 train_dataloader = DataLoader(
      6     train_dataset,
      7     batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device, in_order)
    381             else:  # map-style
    382                 if shuffle:
--> 383                     sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
    384                 else:
    385                     sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in __init__(self, data_source, replacement, num_samples, generator)
    162             )
    163 
--> 164         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
    165             raise ValueError(
    166                 f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in num_samples(self)
    171         # dataset size might change at runtime
    172         if self._num_samples is None:
--> 173             return len(self.data_source)
    174         return self._num_samples
    175 

/tmp/ipykernel_11/2021694838.py in __len__(self)
     23 
     24     def __len__(self):
---> 25         return self.encodings["input_ids"].shape[0]
     26 
     27     def __getitem__(self, i):

AttributeError: 'Column' object has no attribute 'shape'

## === cell 24
try:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
except Exception as e:
    raise RuntimeError(
        "Failed to load transformer model. This is often due to protobuf/transformers "
        "binary incompat in the Kaggle image. The script sets "
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python to mitigate this, but it may "
        "still fail in some environments."
    ) from e

try:
    model.config.problem_type = "multi_label_classification"
except Exception:
    pass



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 25
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



## === cell 26
LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)



## === cell 27
epochs = 3
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2942943391.py in <cell line: 0>()
      1 epochs = 3
----> 2 total_steps = len(train_dataloader) * epochs
      3 
      4 scheduler = get_linear_schedule_with_warmup(
      5     optimizer,

NameError: name 'train_dataloader' is not defined

## === cell 28
criterion = nn.BCEWithLogitsLoss()




## === cell 29
def accuracy_thresh_count(y_pred, y_true, thresh: float = 0.4, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    correct_per_row = ((y_pred > thresh) == (y_true > 0.5)).float().mean(dim=1)
    correct_sum = correct_per_row.sum().item()
    return correct_sum, y_true.shape[0]




## === cell 30
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))




## === cell 31
training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0.0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 400 == 0 and not step == 0:
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

        output = model(b_input_ids, attention_mask=b_input_mask)
        logits = output.logits
        loss = criterion(logits, b_labels)

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    total_eval_loss = 0.0

    for batch in val_dataloader:
        b_input_ids = batch["input_ids"].to(device, non_blocking=True)
        b_input_mask = batch["attention_mask"].to(device, non_blocking=True)
        b_labels = batch["labels"].to(device, non_blocking=True)

        with torch.no_grad():
            output = model(b_input_ids, attention_mask=b_input_mask)
            logits = output.logits
            loss = criterion(logits, b_labels)

        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.4f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Training Accur": np.nan,  # intentionally not computed per-epoch to save time
            "Valid. Accur.": np.nan,  # intentionally not computed per-epoch to save time
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1030177692.py in <cell line: 0>()
     15     model.train()
     16 
---> 17     for step, batch in enumerate(train_dataloader):
     18         if step % 400 == 0 and not step == 0:
     19             elapsed = format_time(time.time() - t0)

NameError: name 'train_dataloader' is not defined

## === cell 32
pd.set_option("display.precision", 5)
df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
df_stats



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1450291981.py in <cell line: 0>()
      1 pd.set_option("display.precision", 5)
----> 2 df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
      3 df_stats
      4 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in set_index(self, keys, drop, append, inplace, verify_integrity)
   6120 
   6121         if missing:
-> 6122             raise KeyError(f"None of {missing} are in the columns")
   6123 
   6124         if inplace:

KeyError: "None of ['epoch'] are in the columns"

## === cell 33
pass



## === cell 34
pass



## === cell 35
thresh = 0.4
model.eval()
with torch.no_grad():
    batch = next(iter(val_dataloader))
    outputs = model(
        batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
    )
    pred_probs = torch.sigmoid(outputs.logits).cpu().numpy()
pred_probs[:3]



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705069384.py in <cell line: 0>()
      2 model.eval()
      3 with torch.no_grad():
----> 4     batch = next(iter(val_dataloader))
      5     outputs = model(
      6         batch["input_ids"].to(device, non_blocking=True),

NameError: name 'val_dataloader' is not defined

## === cell 36
y_pred = (pred_probs > thresh).astype(int)
y_pred[:3]



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/707647279.py in <cell line: 0>()
----> 1 y_pred = (pred_probs > thresh).astype(int)
      2 y_pred[:3]
      3 

NameError: name 'pred_probs' is not defined

## === cell 37
y_true = batch["labels"].cpu().numpy()
y_true[:3]



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1091080269.py in <cell line: 0>()
----> 1 y_true = batch["labels"].cpu().numpy()
      2 y_true[:3]
      3 

NameError: name 'batch' is not defined

## === cell 38
from sklearn.metrics import hamming_loss, accuracy_score


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_score:", hamming_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1649602339.py in <cell line: 0>()
     17 
     18 
---> 19 print("accuracy_score:", accuracy_score(y_true, y_pred))
     20 print("Hamming_score:", hamming_score(y_true, y_pred))
     21 print("Hamming_loss:", hamming_loss(y_true, y_pred))

NameError: name 'y_true' is not defined

## === cell 39
PATH = "./toxic_distilBERT_multilabel.pt"
torch.save(model.state_dict(), PATH)



## === cell 40
test_csv.head()



## === cell 41
len(test_csv)




## === cell 42
class TokenizingTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer_, max_len):
        self.texts = texts
        self.tokenizer = tokenizer_
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def make_collate_fn(tokenizer_, max_len):
    def collate_fn(text_batch):
        enc = tokenizer_(
            list(text_batch),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    return collate_fn


test_texts = test_csv["comment_text"].fillna("").tolist()
sub_dataset = TokenizingTextDataset(test_texts, tokenizer, MAX_LEN)

infer_batch_size = 128 if torch.cuda.is_available() else 32

sub_dataloader = DataLoader(
    sub_dataset,
    batch_size=infer_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=make_collate_fn(tokenizer, MAX_LEN),
)



## === cell 43
model.eval()
t0 = time.time()

predictions = np.empty((len(sub_dataset), len(categories)), dtype=np.float32)

offset = 0
with torch.inference_mode():
    for step, batch in enumerate(sub_dataloader):
        if step % 400 == 0 and not step == 0:
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

        bs = pred_probs.shape[0]
        predictions[offset : offset + bs] = pred_probs
        offset += bs



## === cell 44
predictions_df = pd.DataFrame(predictions, columns=categories)
len(predictions_df), predictions_df.head()



## === cell 45
submission = sample_sub[["id"]].copy()
for c in categories:
    submission[c] = predictions_df[c].values

submission = submission[["id"] + categories]
submission.head()



## === cell 46
submission.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("Any NaNs:", submission.isna().any().to_dict())
print("Test rows:", len(test_csv), "Submission rows:", len(submission))

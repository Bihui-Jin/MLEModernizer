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
import os, random, time, datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AdamW,
    get_linear_schedule_with_warmup,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

df = pd.read_csv(train_path)
test_csv = pd.read_csv(test_path)

print("Train columns:", df.columns.tolist())
print("Test columns :", test_csv.columns.tolist())



## === cell 3
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
feature_col = "comment_text"

condition = df[target_cols].sum(axis=1) > 0
df["y"] = condition.astype(int)



## === cell 4
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_val_df, test_df = train_test_split(
    df[["id", feature_col] + target_cols],
    test_size=0.2,
    random_state=seed,
    stratify=df["y"],
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.25, random_state=seed, stratify=train_val_df["y"]
)

print(f"Sizes → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'y'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2215982476.py in <cell line: 0>()
     13 )
     14 train_df, val_df = train_test_split(
---> 15     train_val_df, test_size=0.25, random_state=seed, stratify=train_val_df["y"]
     16 )
     17 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'y'

## === cell 5
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3003528452.py in <cell line: 0>()
      1 # reset indices for safety
----> 2 train_df = train_df.reset_index(drop=True)
      3 val_df = val_df.reset_index(drop=True)
      4 test_df = test_df.reset_index(drop=True)
      5 

NameError: name 'train_df' is not defined

## === cell 6
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


def encode_texts(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
    )


train_enc = encode_texts(train_df[feature_col].tolist())
val_enc = encode_texts(val_df[feature_col].tolist())
test_enc = encode_texts(test_df[feature_col].tolist())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1058151030.py in <cell line: 0>()
     16 
     17 
---> 18 train_enc = encode_texts(train_df[feature_col].tolist())
     19 val_enc = encode_texts(val_df[feature_col].tolist())
     20 test_enc = encode_texts(test_df[feature_col].tolist())

NameError: name 'train_df' is not defined

## === cell 7
train_seq = train_enc["input_ids"]
train_mask = train_enc["attention_mask"]
train_y = torch.tensor(train_df[target_cols].values, dtype=torch.float)

val_seq = val_enc["input_ids"]
val_mask = val_enc["attention_mask"]
val_y = torch.tensor(val_df[target_cols].values, dtype=torch.float)

test_seq = test_enc["input_ids"]
test_mask = test_enc["attention_mask"]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1953312838.py in <cell line: 0>()
      1 # tensors for inputs and labels
----> 2 train_seq = train_enc["input_ids"]
      3 train_mask = train_enc["attention_mask"]
      4 train_y = torch.tensor(train_df[target_cols].values, dtype=torch.float)
      5 

NameError: name 'train_enc' is not defined

## === cell 8
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2026645127.py in <cell line: 0>()
      1 batch_size = 32
      2 
----> 3 train_data = TensorDataset(train_seq, train_mask, train_y)
      4 train_sampler = RandomSampler(train_data)
      5 train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

NameError: name 'train_seq' is not defined

## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(target_cols)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
criterion = nn.BCEWithLogitsLoss()

epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def accuracy_thresh(y_pred, y_true, thresh: float = 0.4):
    y_pred = torch.sigmoid(y_pred) if y_pred.shape[-1] else y_pred
    return ((y_pred > thresh).float() == y_true).float().mean().item()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3907116159.py in <cell line: 0>()
      5 model.to(device)
      6 
----> 7 optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
      8 criterion = nn.BCEWithLogitsLoss()
      9 

NameError: name 'AdamW' is not defined

## === cell 10
for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    model.train()
    total_loss, total_acc = 0, 0
    for step, (ids, mask, labels) in enumerate(train_loader):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_acc += accuracy_thresh(outputs.logits, labels)
    avg_train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)

    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            outputs = model(ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            val_acc += accuracy_thresh(outputs.logits, labels)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)

    print(f"  Train loss: {avg_train_loss:.4f}  |  Train acc: {train_acc:.4f}")
    print(f"  Val   loss: {avg_val_loss:.4f}  |  Val   acc: {val_acc:.4f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3503460749.py in <cell line: 0>()
----> 1 for epoch in range(1, epochs + 1):
      2     print(f"\nEpoch {epoch}/{epochs}")
      3     model.train()
      4     total_loss, total_acc = 0, 0
      5     for step, (ids, mask, labels) in enumerate(train_loader):

NameError: name 'epochs' is not defined

## === cell 11
model.eval()
predictions = []
test_dataset = TensorDataset(test_seq, test_mask)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for ids, mask in test_loader:
        ids, mask = ids.to(device), mask.to(device)
        outputs = model(ids, attention_mask=mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        predictions.append(probs)

pred_array = np.vstack(predictions)  # shape (num_test, 6)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4294697752.py in <cell line: 0>()
      2 model.eval()
      3 predictions = []
----> 4 test_dataset = TensorDataset(test_seq, test_mask)
      5 test_loader = DataLoader(test_dataset, batch_size=batch_size)
      6 

NameError: name 'test_seq' is not defined

## === cell 12
submission = pd.DataFrame(pred_array, columns=target_cols)
submission.insert(0, "id", test_csv["id"])
print(submission.head())

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3934025249.py in <cell line: 0>()
      1 # Build submission
----> 2 submission = pd.DataFrame(pred_array, columns=target_cols)
      3 submission.insert(0, "id", test_csv["id"])
      4 print(submission.head())
      5 

NameError: name 'pred_array' is not defined

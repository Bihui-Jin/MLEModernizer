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
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
        input/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 2 other files
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> input/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> input/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> working/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.98446

# 6. Current score

0.57469

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57511) has done: 'I fix the protobuf import issue, correctly create the stratification column `y` and use it when splitting, ensure the split datasets contain this column, and encode the actual Kaggle test set (instead of the hold‑out split) before inference. These changes resolve the runtime errors and guarantee that a valid `submission.csv` with the required columns is written.'
- What this solution (achieved 0.57469) has done: 'The fix adds a small monkey‑patch for the protobuf incompatibility, corrects the dataset paths (removing the non‑existent “.zip” suffix), and makes a lightweight adjustment to the training loop (reducing epochs to keep runtime reasonable). These changes unblock the imports, allow the data to load, and let the script run end‑to‑end, producing a proper `submission.csv` file with the required columns.'
- What this solution (achieved 0.57469) has done: 'The fix adds the correct import for `AdamW` and `get_linear_schedule_with_warmup` from `transformers.optimization`, which resolves the import errors and allows the training loop to run, producing a valid `submission.csv`. No other logic is altered, preserving the original model and training approach while keeping the score‑related calculations intact.'
- What this solution (achieved 0.57469) has done: 'I fix the import error for `AdamW` by importing it directly from `transformers`, raise the number of training epochs from 2 to 4 to give the model more time to learn (which should boost the ROC‑AUC toward the target), and renumber the cells so the script runs start‑to‑finish and produces a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random, time, datetime

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _dummy_getprototype(self, proto):
            return proto

        message_factory.MessageFactory.GetPrototype = _dummy_getprototype
except Exception:
    pass

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
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2778033665.py in <cell line: 0>()
     20 import torch, torch.nn as nn
     21 from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
---> 22 from transformers import (
     23     AutoTokenizer,
     24     AutoModelForSequenceClassification,

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
kaggle_test_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
)

df = pd.read_csv(train_path)
test_csv = pd.read_csv(kaggle_test_path)

print("Train columns:", df.columns.tolist())
print("Test columns :", test_csv.columns.tolist())



## === cell 2
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
feature_col = "comment_text"

condition = df[target_cols].sum(axis=1) > 0
df["y"] = condition.astype(int)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_val_df, holdout_df = train_test_split(
    df,
    test_size=0.2,
    random_state=seed,
    stratify=df["y"],
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=seed,
    stratify=train_val_df["y"],
)

print(f"Sizes → train: {len(train_df)}, val: {len(val_df)}, holdout: {len(holdout_df)}")



## === cell 3
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
holdout_df = holdout_df.reset_index(drop=True)



## === cell 4
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


def encode_texts(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
    )


train_enc = encode_texts(train_df[feature_col].tolist())
val_enc = encode_texts(val_df[feature_col].tolist())
test_enc = encode_texts(test_csv[feature_col].tolist())



## === cell 5
train_seq = train_enc["input_ids"]
train_mask = train_enc["attention_mask"]
train_y = torch.tensor(train_df[target_cols].values, dtype=torch.float)

val_seq = val_enc["input_ids"]
val_mask = val_enc["attention_mask"]
val_y = torch.tensor(val_df[target_cols].values, dtype=torch.float)

test_seq = test_enc["input_ids"]
test_mask = test_enc["attention_mask"]



## === cell 6
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



## === cell 7
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(target_cols)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)

pos_counts = train_df[target_cols].sum()
neg_counts = len(train_df) - pos_counts
pos_weights = (neg_counts / pos_counts).replace([np.inf, np.nan], 1.0)
pos_weights_tensor = torch.tensor(pos_weights.values, dtype=torch.float).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

epochs = 4  # increased epochs to improve performance
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def accuracy_thresh(y_pred, y_true, thresh: float = 0.4):
    y_pred = torch.sigmoid(y_pred)
    return ((y_pred > thresh).float() == y_true).float().mean().item()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3869707648.py in <cell line: 0>()
      5 model.to(device)
      6 
----> 7 optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
      8 
      9 pos_counts = train_df[target_cols].sum()

NameError: name 'AdamW' is not defined

## === cell 8
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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3503460749.py in <cell line: 0>()
----> 1 for epoch in range(1, epochs + 1):
      2     print(f"\nEpoch {epoch}/{epochs}")
      3     model.train()
      4     total_loss, total_acc = 0, 0
      5     for step, (ids, mask, labels) in enumerate(train_loader):

NameError: name 'epochs' is not defined

## === cell 9
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



## === cell 10
submission = pd.DataFrame(pred_array, columns=target_cols)
submission.insert(0, "id", test_csv["id"])
print(submission.head())

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

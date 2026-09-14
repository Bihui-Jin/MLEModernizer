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
import os, random, time, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)



## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

df = pd.read_csv(train_path, compression="zip")
test_df = pd.read_csv(test_path, compression="zip")

print("train shape:", df.shape, "test shape:", test_df.shape)

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 2
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=seed_value,
    shuffle=True,
)

print("train split:", train_df.shape, "val split:", val_df.shape)



## === cell 3
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def encode_texts(texts):
    return tokenizer(
        texts,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


train_enc = encode_texts(train_df["comment_text"].tolist())
val_enc = encode_texts(val_df["comment_text"].tolist())
test_enc = encode_texts(test_df["comment_text"].tolist())

train_seq, train_mask = train_enc["input_ids"], train_enc["attention_mask"]
val_seq, val_mask = val_enc["input_ids"], val_enc["attention_mask"]
test_seq, test_mask = test_enc["input_ids"], test_enc["attention_mask"]

train_labels = torch.tensor(train_df[categories].values, dtype=torch.float)
val_labels = torch.tensor(val_df[categories].values, dtype=torch.float)



## === cell 4
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_labels)
val_data = TensorDataset(val_seq, val_mask, val_labels)

train_dataloader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=batch_size,
)

val_dataloader = DataLoader(
    val_data,
    sampler=SequentialSampler(val_data),
    batch_size=batch_size,
)



## === cell 5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
model.to(device)

LEARN_RATE = 3e-5
optimizer = optim.AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)

epochs = 2  # short runtime; can be increased for better AUC
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss()


def accuracy_thresh(y_pred, y_true, thresh: float = 0.4):
    y_pred = torch.sigmoid(y_pred)
    return ((y_pred > thresh) == y_true).float().mean().item()




## === cell 6
for epoch_i in range(epochs):
    print(f"\nEpoch {epoch_i+1}/{epochs}")
    model.train()
    total_loss = 0
    total_acc = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        loss = criterion(outputs.logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += accuracy_thresh(outputs.logits, b_labels)

    avg_train_loss = total_loss / len(train_dataloader)
    avg_train_acc = total_acc / len(train_dataloader)
    print(f"  Train loss: {avg_train_loss:.4f}  acc: {avg_train_acc:.4f}")

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs.logits, b_labels)
            val_loss += loss.item()
            val_acc += accuracy_thresh(outputs.logits, b_labels)

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_acc = val_acc / len(val_dataloader)
    print(f"  Val loss: {avg_val_loss:.4f}  acc: {avg_val_acc:.4f}")



## === cell 7
test_data = TensorDataset(test_seq, test_mask)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_input_mask = [b.to(device) for b in batch]
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        probs = torch.sigmoid(outputs.logits)
        all_preds.append(probs.cpu().numpy())

predictions = np.concatenate(all_preds, axis=0)



## === cell 8
predictions_df = pd.DataFrame(predictions, columns=categories)
submission = pd.concat([test_df["id"], predictions_df], axis=1)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

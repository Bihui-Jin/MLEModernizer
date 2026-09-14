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

3.10

# 3. Installed packages

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.98112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

BASE_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

EPOCHS = 2
MAX_TOKEN_COUNT = 64
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_PATH)
df = df.sample(n=50000, random_state=RANDOM_SEED).reset_index(drop=True)
test_df = pd.read_csv(TEST_PATH)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/605010704.py in <cell line: 0>()
      1 # Load data (sample a subset for faster training)
----> 2 df = pd.read_csv(TRAIN_PATH)
      3 df = df.sample(n=50000, random_state=RANDOM_SEED).reset_index(drop=True)
      4 test_df = pd.read_csv(TEST_PATH)
      5 

NameError: name 'TRAIN_PATH' is not defined

## === cell 2
train_df, val_df = train_test_split(
    df,
    test_size=0.05,
    random_state=RANDOM_SEED,
    stratify=df["toxic"],
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/175972762.py in <cell line: 0>()
      1 train_df, val_df = train_test_split(
----> 2     df,
      3     test_size=0.05,
      4     random_state=RANDOM_SEED,
      5     stratify=df["toxic"],

NameError: name 'df' is not defined

## === cell 3
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")




## === cell 4
class ToxicCommentsDataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, tokenizer, max_len: int = 128, test: bool = False
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment = row["comment_text"]
        enc = self.tokenizer.encode_plus(
            comment,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "_id": row["id"],  # scalar id
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }
        if not self.test:
            labels = torch.FloatTensor(row[LABEL_COLUMNS].values)
            item["labels"] = labels
        return item




## === cell 5
train_dataset = ToxicCommentsDataset(
    train_df, tokenizer, max_len=MAX_TOKEN_COUNT, test=False
)
val_dataset = ToxicCommentsDataset(
    val_df, tokenizer, max_len=MAX_TOKEN_COUNT, test=False
)
test_dataset = ToxicCommentsDataset(
    test_df, tokenizer, max_len=MAX_TOKEN_COUNT, test=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1328883125.py in <cell line: 0>()
      1 train_dataset = ToxicCommentsDataset(
----> 2     train_df, tokenizer, max_len=MAX_TOKEN_COUNT, test=False
      3 )
      4 val_dataset = ToxicCommentsDataset(
      5     val_df, tokenizer, max_len=MAX_TOKEN_COUNT, test=False

NameError: name 'train_df' is not defined

## === cell 6
class ToxicCommentTagger(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased", return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)  # (batch, n_classes)
        loss = 0.0
        if labels is not None:
            loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        return loss, probs




## === cell 7
model = ToxicCommentTagger(n_classes=len(LABEL_COLUMNS)).to(DEVICE)

steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 5,
    num_training_steps=total_steps,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2494850201.py in <cell line: 0>()
----> 1 model = ToxicCommentTagger(n_classes=len(LABEL_COLUMNS)).to(DEVICE)
      2 
      3 steps_per_epoch = len(train_loader)
      4 total_steps = steps_per_epoch * EPOCHS
      5 optimizer = AdamW(model.parameters(), lr=2e-5)

NameError: name 'LABEL_COLUMNS' is not defined

## === cell 8
def train_one_epoch():
    model.train()
    total_loss = 0.0
    all_preds = []
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        loss, probs = model(input_ids, attn_mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.append(probs.detach().cpu())
    avg_loss = total_loss / len(train_loader)
    return avg_loss, torch.cat(all_preds, dim=0)




## === cell 9
def evaluate():
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            loss, probs = model(input_ids, attn_mask, labels)
            total_loss += loss.item()
            preds.append(probs.cpu())
            trues.append(labels.cpu())
    avg_loss = total_loss / len(val_loader)
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    aucs = []
    for i, name in enumerate(LABEL_COLUMNS):
        try:
            auc = roc_auc_score(trues[:, i], preds[:, i])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"{name} ROC AUC: {auc:.4f}")
    print(f"Mean ROC AUC: {np.nanmean(aucs):.4f}")
    return avg_loss, preds, trues




## === cell 10
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, _ = train_one_epoch()
    val_loss, _, _ = evaluate()
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved best model")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1786965185.py in <cell line: 0>()
      1 best_val_loss = float("inf")
----> 2 for epoch in range(EPOCHS):
      3     print(f"\nEpoch {epoch+1}/{EPOCHS}")
      4     train_loss, _ = train_one_epoch()
      5     val_loss, _, _ = evaluate()

NameError: name 'EPOCHS' is not defined

## === cell 11
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3625677320.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
      2 model.eval()
      3 
      4 

NameError: name 'model' is not defined

## === cell 12
def predict_test():
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["_id"]  # scalar id
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            _, probs = model(input_ids, attn_mask)  # no labels
            all_ids.append(ids)
            all_preds.append(probs.cpu())
    preds = torch.cat(all_preds, dim=0).numpy()
    return all_ids, preds




## === cell 13
test_ids, test_preds = predict_test()
submission = pd.DataFrame({"id": test_ids})
for i, col in enumerate(LABEL_COLUMNS):
    submission[col] = test_preds[:, i]
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1552852268.py in <cell line: 0>()
----> 1 test_ids, test_preds = predict_test()
      2 submission = pd.DataFrame({"id": test_ids})
      3 for i, col in enumerate(LABEL_COLUMNS):
      4     submission[col] = test_preds[:, i]
      5 submission.to_csv("submission.csv", index=False)

/tmp/ipykernel_55/660175072.py in predict_test()
      3     all_preds = []
      4     with torch.no_grad():
----> 5         for batch in test_loader:
      6             ids = batch["_id"]  # scalar id
      7             input_ids = batch["input_ids"].to(DEVICE)

NameError: name 'test_loader' is not defined

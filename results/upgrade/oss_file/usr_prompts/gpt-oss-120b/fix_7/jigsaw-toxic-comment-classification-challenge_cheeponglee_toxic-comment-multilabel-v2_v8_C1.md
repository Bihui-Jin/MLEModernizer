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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random, time, datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Dataset,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True



## === cell 1
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)



## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

df = pd.read_csv(train_path)
test_csv = pd.read_csv(test_path)

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
target_col = categories
feature_col = ["comment_text"]



## === cell 3
print(f"Train shape: {df.shape}, Test shape: {test_csv.shape}")
print("Target columns:", target_col)



## === cell 4
df = df.rename(columns={"id": "idx"})
test_csv = test_csv.rename(columns={"id": "idx"})


train_val_df, test_df = train_test_split(
    df[["idx", "comment_text"] + categories],
    test_size=0.2,
    random_state=seed_value,
    stratify=df[categories].idxmax(axis=1),  # simple stratification
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=seed_value,
    stratify=train_val_df[categories].idxmax(axis=1),
)

train_df = train_df.sample(n=min(20000, len(train_df)), random_state=seed_value)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



## === cell 5
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def encode_dataframe(df):
    return tokenizer.batch_encode_plus(
        df["comment_text"].tolist(),
        max_length=200,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
    )


train_enc = encode_dataframe(train_df)
val_enc = encode_dataframe(val_df)

train_labels = torch.tensor(train_df[categories].values, dtype=torch.float)
val_labels = torch.tensor(val_df[categories].values, dtype=torch.float)



## === cell 6
batch_size = 32

train_data = TensorDataset(
    train_enc["input_ids"], train_enc["attention_mask"], train_labels
)
val_data = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)

train_dataloader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=batch_size,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_data,
    sampler=SequentialSampler(val_data),
    batch_size=batch_size,
    pin_memory=True,
)



## === cell 7
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(categories)
)
model.to(device)

LEARN_RATE = 3e-5
optimizer = AdamW(model.parameters(), lr=LEARN_RATE, eps=1e-8)

epochs = 2
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss()




## === cell 8
def accuracy_thresh(y_pred, y_true, thresh: float = 0.4):
    """Binary accuracy per sample, averaged over the batch."""
    y_pred = torch.sigmoid(y_pred)
    return ((y_pred > thresh) == y_true.byte()).float().mean().item()


model.train()
for epoch_i in range(epochs):
    total_train_loss = 0.0
    total_train_acc = 0.0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        loss = criterion(outputs.logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        total_train_acc += accuracy_thresh(outputs.logits, b_labels)

    avg_loss = total_train_loss / len(train_dataloader)
    avg_acc = total_train_acc / len(train_dataloader)
    print(f"Epoch {epoch_i+1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")



## === cell 9
test_enc = encode_dataframe(test_csv)

test_data = TensorDataset(test_enc["input_ids"], test_enc["attention_mask"])

inference_batch_size = 128

test_loader = DataLoader(
    test_data,
    sampler=SequentialSampler(test_data),
    batch_size=inference_batch_size,
    pin_memory=True,
    num_workers=4,
)

model.eval()
predictions = np.empty((len(test_csv), len(categories)), dtype=np.float32)

with torch.no_grad():
    start_idx = 0
    for batch in tqdm(test_loader, desc="Predicting"):
        b_input_ids, b_input_mask = [b.to(device) for b in batch]
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        batch_len = probs.shape[0]
        predictions[start_idx : start_idx + batch_len] = probs
        start_idx += batch_len



## === cell 10
pred_df = pd.DataFrame(predictions, columns=categories)
submission = pd.concat([test_csv["idx"], pred_df], axis=1)
submission = submission.rename(columns={"idx": "id"})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

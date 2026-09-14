# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"

final_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

kfolds = 5

perm = np.random.RandomState(SEED).permutation(len(train))
fold_ids = np.empty(len(train), dtype=np.int64)
fold_ids[perm] = np.arange(len(train), dtype=np.int64) % kfolds
train["kfold"] = fold_ids

final_df.head()



## === cell 1
final_df["text"] = (
    "User prompt: "
    + final_df["prompt"].astype(str)
    + "\n\nModel A :\n"
    + final_df["response_a"].astype(str)
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"].astype(str)
)
print(final_df["text"].iloc[0])

train["text"] = (
    "User prompt: "
    + train["prompt"].astype(str)
    + "\n\nModel A :\n"
    + train["response_a"].astype(str)
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"].astype(str)
)



## === cell 2
print(len(final_df))
final_df.head()



## === cell 3
from torch import nn

final_texts = final_df["text"].values
batch_size = 8
num_classes = 3


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=128,
        hidden_dim2=64,
        num_classes=num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        output = self.fc(h_n[-1])  # use final hidden state
        logits = self.fc2(output)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def encode(tokens, vocab):
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)


def predict(text, model, vocab):
    tokens = text.split()
    encoded = encode(tokens, vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(encoded)
        probs = F.softmax(logits, dim=1)
        return probs


def _pad_collate(batch, pad_idx=0):
    xs, ys = zip(*batch)
    maxlen = max(x.size(0) for x in xs)
    xb = torch.full((len(xs), maxlen), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        xb[i, : x.size(0)] = x
    yb = torch.tensor(ys, dtype=torch.long)
    return xb, yb


def train_one_fold_gru(train_texts, train_labels, vocab, epochs=1, lr=2e-3):
    model = GRUClassifier(vocab_size=len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    encoded_texts = [encode(t.split(), vocab) for t in train_texts]
    dataset = list(zip(encoded_texts, train_labels))

    g = torch.Generator()
    g.manual_seed(SEED)

    model.train()
    for ep in range(epochs):
        idx = torch.randperm(len(dataset), generator=g).tolist()
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            batch = [dataset[j] for j in batch_idx]
            xb, yb = _pad_collate(batch, pad_idx=0)
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model




## === cell 4
final_df["winner_model_a"] = 0.0
final_df["winner_model_b"] = 0.0
final_df["winner_tie"] = 0.0

epoches_to_use = [0, 0, 0, 0, 0]

y_class = (
    train[["winner_model_a", "winner_model_b", "winner_tie"]]
    .values.argmax(axis=1)
    .astype(np.int64)
)

for kfold in range(kfolds):
    print(f"Prediction fold: {kfold}")

    trn_mask = train["kfold"] != kfold
    train_texts_fold = train.loc[trn_mask, "text"].values
    train_y_fold = y_class[trn_mask.to_numpy()]

    train_tokenized = [t.split() for t in train_texts_fold]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in Counter(w for sent in train_tokenized for w in sent):
        if w not in vocab:
            vocab[w] = len(vocab)

    model_path = f"/kaggle/input/10050-training-5fold-lmsys/gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"

    if os.path.exists(model_path):
        model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
        state = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(state)
        model_loaded.eval()
    else:
        print(f"  Checkpoint not found, training fallback model for fold {kfold}...")
        model_loaded = train_one_fold_gru(
            train_texts=train_texts_fold,
            train_labels=train_y_fold,
            vocab=vocab,
            epochs=1,  # keep runtime within limits; minimal viable training
            lr=2e-3,
        )

    for i, text in enumerate(final_texts):
        probs = predict(text, model_loaded, vocab)
        final_df.at[i, "winner_model_a"] += float(probs[0, 0]) / kfolds
        final_df.at[i, "winner_model_b"] += float(probs[0, 1]) / kfolds
        final_df.at[i, "winner_tie"] += float(probs[0, 2]) / kfolds

eps = 1e-15
probs_mat = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(
    dtype=np.float64
)
probs_mat = np.clip(probs_mat, eps, 1.0)
row_sums = probs_mat.sum(axis=1, keepdims=True)
probs_mat = probs_mat / row_sums
final_df[["winner_model_a", "winner_model_b", "winner_tie"]] = probs_mat



## === cell 5
sub = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()

eps = 1e-15
mat = sub[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(dtype=np.float64)
mat = np.clip(mat, eps, 1.0)
mat = mat / mat.sum(axis=1, keepdims=True)
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = mat

row_sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
max_dev = float(np.max(np.abs(row_sums.to_numpy() - 1.0)))
print("Max deviation from 1.0 row-sum:", max_dev)

sub.to_csv("submission.csv", index=False)
print("Done. Wrote submission.csv with shape:", sub.shape)
print(sub.head())

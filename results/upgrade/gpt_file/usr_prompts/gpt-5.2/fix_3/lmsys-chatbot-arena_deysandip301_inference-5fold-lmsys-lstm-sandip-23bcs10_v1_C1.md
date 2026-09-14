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
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

final_df = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train:", train.shape, "test:", final_df.shape, "sample:", sample_sub.shape)
print(train.head(2))



## === cell 1
final_df["text"] = (
    "User prompt: "
    + final_df["prompt"].astype(str)
    + "\n\nModel A :\n"
    + final_df["response_a"].astype(str)
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"].astype(str)
)
train["text"] = (
    "User prompt: "
    + train["prompt"].astype(str)
    + "\n\nModel A :\n"
    + train["response_a"].astype(str)
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"].astype(str)
)

print(final_df["text"].iloc[0][:500])



## === cell 2
final_texts = final_df["text"].values
batch_size = 8
num_classes = 3
kfolds = 5


class LSTMClassifier(nn.Module):
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
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])  # use final hidden state
        logits = self.fc2(output)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

train_tokenized = [t.split() for t in train["text"].values]
vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)

PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]


def encode(tokens):
    return torch.tensor([vocab.get(w, UNK_IDX) for w in tokens], dtype=torch.long)


def encode_padded(text, max_len):
    tokens = text.split()
    ids = [vocab.get(w, UNK_IDX) for w in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def predict(model_loaded, text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)
    logits = model_loaded(encoded)
    probs = F.softmax(logits, dim=1)
    return probs


@torch.no_grad()
def predict_padded(model_loaded, text, max_len):
    encoded = encode_padded(text, max_len=max_len).unsqueeze(0).to(device)
    logits = model_loaded(encoded)
    probs = F.softmax(logits, dim=1)
    return probs


target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
if not all(c in train.columns for c in target_cols):
    raise ValueError(f"Missing target columns in train.csv: expected {target_cols}")

y = train[target_cols].values.astype(np.float32)
y_idx = y.argmax(axis=1).astype(np.int64)
print("Label distribution (0=a,1=b,2=tie):", np.bincount(y_idx, minlength=3))



## === cell 3
CKPT_DIR = "/kaggle/input/training-5fold-lmsys-lstm-sandip-dey-23bcs10114"


def _existing_ckpts():
    paths = []
    for kfold in range(kfolds):
        p = os.path.join(CKPT_DIR, f"lstm_classifier_kfold_{kfold}_best.pth")
        if os.path.exists(p):
            paths.append(p)
    return paths


existing = _existing_ckpts()
use_external_ckpts = len(existing) == kfolds
print(
    "External checkpoints found:",
    len(existing),
    "=>",
    "using" if use_external_ckpts else "training fallback",
)


def make_folds(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr_idx, val_idx))
    return out


def train_one_fold(train_texts, train_labels, max_len, vocab_size, epochs=2, lr=1e-3):
    model = LSTMClassifier(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n = len(train_texts)
    order = np.arange(n)

    for ep in range(epochs):
        np.random.shuffle(order)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            xb = torch.stack(
                [encode_padded(train_texts[i], max_len=max_len) for i in batch_idx]
            ).to(device)
            yb = torch.tensor(train_labels[batch_idx], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(batch_idx)
        print(f"  epoch {ep+1}/{epochs} loss: {total_loss/n:.5f}")
    model.eval()
    return model


MAX_LEN = 256

class_0_probs, class_1_probs, class_2_probs = [], [], []

folds = make_folds(len(train), k=kfolds, seed=SEED)

for kfold in range(kfolds):
    print(f"\nFold {kfold+1}/{kfolds}")
    if use_external_ckpts:
        ckpt_path = os.path.join(CKPT_DIR, f"lstm_classifier_kfold_{kfold}_best.pth")
        print(f"  loading ckpt: {os.path.basename(ckpt_path)}")
        model_loaded = LSTMClassifier(vocab_size=len(vocab)).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model_loaded.load_state_dict(state)
        model_loaded.eval()

        c0, c1, c2 = [], [], []
        for text in tqdm(final_texts, total=len(final_texts), leave=False):
            ans = predict(model_loaded, text)
            c0.append(float(ans[0, 0].cpu()))
            c1.append(float(ans[0, 1].cpu()))
            c2.append(float(ans[0, 2].cpu()))
    else:
        tr_idx, val_idx = folds[kfold]
        model_loaded = train_one_fold(
            train_texts=train["text"].values[tr_idx],
            train_labels=y_idx[tr_idx],
            max_len=MAX_LEN,
            vocab_size=len(vocab),
            epochs=2,
            lr=1e-3,
        )

        c0, c1, c2 = [], [], []
        for text in tqdm(final_texts, total=len(final_texts), leave=False):
            ans = predict_padded(model_loaded, text, max_len=MAX_LEN)
            c0.append(float(ans[0, 0].cpu()))
            c1.append(float(ans[0, 1].cpu()))
            c2.append(float(ans[0, 2].cpu()))

    class_0_probs.append(c0)
    class_1_probs.append(c1)
    class_2_probs.append(c2)

if not (len(class_0_probs) == len(class_1_probs) == len(class_2_probs) == kfolds):
    raise RuntimeError("Missing fold predictions; cannot continue.")
if any(len(p) != len(final_df) for p in class_0_probs):
    raise RuntimeError("Prediction length mismatch vs test set; cannot continue.")



## === cell 4
p0 = np.mean(np.array(class_0_probs, dtype=np.float64), axis=0)
p1 = np.mean(np.array(class_1_probs, dtype=np.float64), axis=0)
p2 = np.mean(np.array(class_2_probs, dtype=np.float64), axis=0)

P = np.vstack([p0, p1, p2]).T  # shape (n_test, 3)
P = np.clip(P, 1e-15, 1.0)  # keep strictly positive for logloss stability
row_sums = P.sum(axis=1, keepdims=True)
P = P / row_sums

final_df["winner_model_a"] = P[:, 0]
final_df["winner_model_b"] = P[:, 1]
final_df["winner_tie"] = P[:, 2]

check_sum = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
print("Row sum min/max:", float(check_sum.min()), float(check_sum.max()))
print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())



## === cell 5
sub = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()

sub = sample_sub[["id"]].merge(sub, on="id", how="left")
if sub[["winner_model_a", "winner_model_b", "winner_tie"]].isna().any().any():
    raise ValueError(
        "NaNs found in submission after aligning with sample_submission ids."
    )

probs = sub[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(
    dtype=np.float64
)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = probs

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Saved to:", os.path.abspath("submission.csv"))

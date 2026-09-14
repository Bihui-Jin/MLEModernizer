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

from torch.utils.data import TensorDataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    + final_df["prompt"].fillna("").astype(str)
    + "\n\nModel A :\n"
    + final_df["response_a"].fillna("").astype(str)
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"].fillna("").astype(str)
)
train["text"] = (
    "User prompt: "
    + train["prompt"].fillna("").astype(str)
    + "\n\nModel A :\n"
    + train["response_a"].fillna("").astype(str)
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"].fillna("").astype(str)
)

print(final_df["text"].iloc[0][:500])



## === cell 2
final_texts = final_df["text"].to_numpy()
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

train_texts = train["text"].to_numpy()

train_texts_list = np.where(pd.isna(train_texts), "", train_texts).astype(str).tolist()
test_texts_list = np.where(pd.isna(final_texts), "", final_texts).astype(str).tolist()

split_ = str.split
train_tokenized = [split_(t) for t in train_texts_list]
test_tokenized = [split_(t) for t in test_texts_list]

from collections import Counter

vocab = {"<pad>": 0, "<unk>": 1}

counts = Counter()
counts.update(tok for toks in train_tokenized for tok in toks)
for w, _ in counts.most_common():
    if w not in vocab:
        vocab[w] = len(vocab)

PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]

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


def tokens_to_ids_lists(token_lists, tok2id, unk_idx):
    get_id = tok2id.get
    return [[get_id(tok, unk_idx) for tok in toks] for toks in token_lists]


def encode_padded_from_ids_lists_np(ids_lists, max_len, pad_idx):
    n = len(ids_lists)
    out = np.full((n, max_len), pad_idx, dtype=np.int64)
    for i, ids in enumerate(ids_lists):
        if not ids:
            continue
        m = min(len(ids), max_len)
        out[i, :m] = ids[:m]
    return out


@torch.no_grad()
def predict_dataset_padded_loader(model_loaded, loader, device):
    model_loaded.eval()
    n = len(loader.dataset)
    probs = np.empty((n, 3), dtype=np.float64)
    offset = 0
    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device, non_blocking=True)
        logits = model_loaded(xb)
        p = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs[offset : offset + p.shape[0]] = p
        offset += p.shape[0]
    return probs


def train_one_fold_encoded(
    X_train_ids_cpu_long, y_train_idx, vocab_size, epochs=2, lr=1e-3
):
    model = LSTMClassifier(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(
        X_train_ids_cpu_long,
        torch.as_tensor(y_train_idx, dtype=torch.long),
    )
    g = torch.Generator()
    g.manual_seed(SEED)  # deterministic shuffling per epoch across runs

    nw = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    model.train()
    n = len(ds)
    for ep in range(epochs):
        total_loss = 0.0
        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * xb_cpu.shape[0]
        print(f"  epoch {ep+1}/{epochs} loss: {total_loss/n:.5f}")

    model.eval()
    return model


MAX_LEN = 256

tok2id = vocab  # already a dict token->id

train_ids_lists = tokens_to_ids_lists(train_tokenized, tok2id, UNK_IDX)
test_ids_lists = tokens_to_ids_lists(test_tokenized, tok2id, UNK_IDX)

X_test_ids_np = encode_padded_from_ids_lists_np(test_ids_lists, MAX_LEN, PAD_IDX)
X_train_all_ids_np = encode_padded_from_ids_lists_np(train_ids_lists, MAX_LEN, PAD_IDX)

X_test_ids_cpu = torch.from_numpy(np.ascontiguousarray(X_test_ids_np)).long()
X_train_all_ids_cpu = torch.from_numpy(np.ascontiguousarray(X_train_all_ids_np)).long()

folds = make_folds(len(train), k=kfolds, seed=SEED)

infer_batch_size = 512 if device.type == "cuda" else 64
infer_nw = 2 if device.type == "cuda" else 0
test_ds = TensorDataset(X_test_ids_cpu)
test_loader = DataLoader(
    test_ds,
    batch_size=infer_batch_size,
    shuffle=False,
    num_workers=infer_nw,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(infer_nw > 0),
    prefetch_factor=2 if infer_nw > 0 else None,
)

probs_sum = np.zeros((len(final_df), 3), dtype=np.float64)

reusable_model = None
if use_external_ckpts:
    reusable_model = LSTMClassifier(vocab_size=len(vocab)).to(device)

for kfold in range(kfolds):
    print(f"\nFold {kfold+1}/{kfolds}")
    if use_external_ckpts:
        ckpt_path = os.path.join(CKPT_DIR, f"lstm_classifier_kfold_{kfold}_best.pth")
        print(f"  loading ckpt: {os.path.basename(ckpt_path)}")
        state = torch.load(ckpt_path, map_location=device)
        reusable_model.load_state_dict(state)
        reusable_model.eval()

        probs = predict_dataset_padded_loader(
            model_loaded=reusable_model,
            loader=test_loader,
            device=device,
        )
    else:
        tr_idx, val_idx = folds[kfold]
        X_train_ids_cpu = X_train_all_ids_cpu[tr_idx]
        model_loaded = train_one_fold_encoded(
            X_train_ids_cpu_long=X_train_ids_cpu,
            y_train_idx=y_idx[tr_idx],
            vocab_size=len(vocab),
            epochs=2,
            lr=1e-3,
        )
        probs = predict_dataset_padded_loader(
            model_loaded=model_loaded,
            loader=test_loader,
            device=device,
        )

    probs_sum += probs

P = probs_sum / float(kfolds)



## === cell 4
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

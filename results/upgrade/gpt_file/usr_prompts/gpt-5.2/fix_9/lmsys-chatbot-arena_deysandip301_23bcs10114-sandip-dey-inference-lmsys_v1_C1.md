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
import re

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

_cpu_cnt = os.cpu_count() or 1
torch.set_num_threads(min(8, _cpu_cnt))
torch.set_num_interop_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

try:
    train = pd.read_csv(train_path, engine="pyarrow")
    final_df = pd.read_csv(test_path, engine="pyarrow")
except Exception:
    train = pd.read_csv(train_path)
    final_df = pd.read_csv(test_path)

print(train.shape, final_df.shape)
train.head()



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

target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
y = train[target_cols].values.astype(np.float32)
train["label"] = y.argmax(axis=1).astype(np.int64)

print(train["label"].value_counts().sort_index())



## === cell 2
batch_size = 8
num_classes = 3
kfolds = 5


class RNNClassifier(nn.Module):
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
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.rnn(packed)
        else:
            _, h_n = self.rnn(x)
        output = self.fc(h_n[-1])  # final hidden state
        logits = self.fc2(output)
        return logits


_WS_SPLIT = re.compile(r"\s+")


def fast_split(text: str):
    text = text.strip()
    if not text:
        return []
    return _WS_SPLIT.split(text)


def seed_worker(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_vocab_and_encode(texts_list, initial_vocab=None):
    if initial_vocab is None:
        vocab = {"<pad>": 0, "<unk>": 1}
    else:
        vocab = dict(initial_vocab)
        if "<pad>" not in vocab:
            vocab["<pad>"] = 0
        if "<unk>" not in vocab:
            vocab["<unk>"] = 1
    unk_id = vocab.get("<unk>", 1)

    tokenized = [fast_split(t) for t in texts_list]

    seen = set(vocab.keys())
    new_tokens = []
    add_new = new_tokens.append
    for toks in tokenized:
        for w in toks:
            if w not in seen:
                seen.add(w)
                add_new(w)
    for w in new_tokens:
        vocab[w] = len(vocab)

    get = vocab.get
    encoded = []
    lengths = np.empty(len(tokenized), dtype=np.int32)
    for i, toks in enumerate(tokenized):
        arr = np.fromiter(
            (get(w, unk_id) for w in toks), dtype=np.int32, count=len(toks)
        )
        encoded.append(arr)
        lengths[i] = arr.size

    return vocab, encoded, lengths


class EncodedDatasetNumpy(Dataset):
    def __init__(self, encoded_list_np, labels_np, indices_np):
        self.encoded_list_np = encoded_list_np
        self.labels_np = labels_np
        self.indices_np = indices_np

    def __len__(self):
        return int(self.indices_np.size)

    def __getitem__(self, i):
        idx = int(self.indices_np[i])
        return self.encoded_list_np[idx], int(self.labels_np[idx])


class EncodedInferenceDatasetNumpy(Dataset):
    def __init__(self, encoded_list_np, indices_np=None):
        self.encoded_list_np = encoded_list_np
        self.indices_np = indices_np

    def __len__(self):
        return (
            int(self.indices_np.size)
            if self.indices_np is not None
            else len(self.encoded_list_np)
        )

    def __getitem__(self, idx):
        if self.indices_np is None:
            return self.encoded_list_np[idx]
        return self.encoded_list_np[int(self.indices_np[idx])]


def collate_fn_numpy(batch):
    xs_np, ys = zip(*batch)
    lengths = torch.tensor([x.size for x in xs_np], dtype=torch.long)
    xs = [torch.from_numpy(x) for x in xs_np]  # int32 tensors
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_padded, lengths, ys


def collate_infer_numpy(batch):
    lengths = torch.tensor([x.size for x in batch], dtype=torch.long)
    xs = [torch.from_numpy(x) for x in batch]  # int32 tensors
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()
    return xs_padded, lengths




## === cell 3
n = len(train)
idx = np.arange(n)
np.random.shuffle(idx)
fold_id = np.zeros(n, dtype=np.int64)
fold_id[idx] = np.arange(n) % kfolds
train["kfold"] = fold_id

train_texts = train["text"].tolist()
test_texts = final_df["text"].tolist()

print("Fold counts:", pd.Series(train["kfold"]).value_counts().sort_index().to_dict())

global_vocab, train_encoded_all, train_lens = build_vocab_and_encode(
    train_texts, initial_vocab=None
)
global_vocab, test_encoded_all, test_lens = build_vocab_and_encode(
    test_texts, initial_vocab=global_vocab
)

print("Global vocab size:", len(global_vocab))

train_labels_all = train["label"].values.astype(np.int64)

test_order = np.argsort(test_lens, kind="mergesort")
test_inv = np.empty_like(test_order)
test_inv[test_order] = np.arange(test_order.size)
n_test = len(test_encoded_all)



## === cell 4
EPOCHS = 1
LR = 1e-3

all_fold_probs = []

cpu_cnt = os.cpu_count() or 0
num_workers = 2 if cpu_cnt > 2 else 0
prefetch_factor = 4 if num_workers > 0 else None
persistent_workers = True if num_workers > 0 else False

g = torch.Generator()
g.manual_seed(SEED)

test_ds_sorted = EncodedInferenceDatasetNumpy(test_encoded_all, indices_np=test_order)
test_dl = DataLoader(
    test_ds_sorted,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_infer_numpy,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=persistent_workers,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
    prefetch_factor=prefetch_factor,
)

for kfold in range(kfolds):
    print(f"\n=== Fold {kfold} ===")

    trn_mask = train["kfold"].values != kfold
    trn_idx = np.flatnonzero(trn_mask)

    trn_idx_sorted = trn_idx[np.argsort(train_lens[trn_idx], kind="mergesort")]

    model = RNNClassifier(vocab_size=len(global_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trn_ds = EncodedDatasetNumpy(train_encoded_all, train_labels_all, trn_idx_sorted)
    trn_dl = DataLoader(
        trn_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_numpy,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
        prefetch_factor=prefetch_factor,
    )

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(trn_dl, desc=f"train epoch {epoch+1}/{EPOCHS}", leave=False)
        running = 0.0
        seen = 0
        for xb, lengths, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lengths=lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running += float(loss.item()) * bs
            seen += bs
        if seen:
            pbar.set_postfix(loss=running / seen)

    model.eval()
    probs_sorted = np.empty((n_test, num_classes), dtype=np.float32)
    write_pos = 0
    with torch.inference_mode():
        for xb, lengths in tqdm(test_dl, desc="predict test", leave=False):
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(xb, lengths=lengths)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            bs = probs.shape[0]
            probs_sorted[write_pos : write_pos + bs] = probs
            write_pos += bs

    probs_fold = probs_sorted[test_inv]
    all_fold_probs.append(probs_fold)



## === cell 5
probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)  # [n_test, 3]

probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

final_df["winner_model_a"] = probs[:, 0]
final_df["winner_model_b"] = probs[:, 1]
final_df["winner_tie"] = probs[:, 2]

row_sums = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
print("Row sum stats:", float(row_sums.min()), float(row_sums.max()))

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## === cell 6
sub_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    sub_path, index=False
)
print("Wrote:", sub_path)
print(pd.read_csv(sub_path).head())

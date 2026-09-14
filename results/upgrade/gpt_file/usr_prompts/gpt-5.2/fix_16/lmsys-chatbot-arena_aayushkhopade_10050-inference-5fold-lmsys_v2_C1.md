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

try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
except Exception:
    pass

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
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import re

_TOKEN_RE = re.compile(r"\S+")

final_texts = final_df["text"].values
train_texts_all = train["text"].values

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

if device.type == "cuda":
    _DL_NUM_WORKERS = 2
else:
    _DL_NUM_WORKERS = 0
_DL_PERSISTENT = bool(_DL_NUM_WORKERS > 0)


def _pad_collate(batch, pad_idx=0):
    xs, ys = zip(*batch)
    xb = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    yb = torch.tensor(ys, dtype=torch.long)
    return xb, yb


def _pad_collate_xonly(batch, pad_idx=0):
    xb = pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    return xb


@torch.no_grad()
def predict_proba_batched(encoded_texts, model, infer_bs=128):
    loader = DataLoader(
        encoded_texts,
        batch_size=infer_bs,
        shuffle=False,
        collate_fn=_pad_collate_xonly,
        num_workers=_DL_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_DL_PERSISTENT,
        prefetch_factor=4 if _DL_NUM_WORKERS > 0 else None,
        in_order=True,
    )
    out = []
    model.eval()
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu())
    return torch.cat(out, dim=0).numpy()


def _iter_tokens_equiv_Splus(text: str):
    return text.split()


def _build_global_vocab_from_texts_fast(texts: np.ndarray, token_re: re.Pattern):
    counts = Counter()
    for t in texts:
        for tok in _iter_tokens_equiv_Splus(t):
            counts[tok] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in counts:
        vocab[w] = len(vocab)
    return vocab


def _pack_texts_to_ids_fast(texts: np.ndarray, token_re: re.Pattern, vocab: dict):
    n = len(texts)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    flat_list = []
    unk_id = 1
    get = vocab.get
    total = 0
    for i, t in enumerate(texts):
        toks = _iter_tokens_equiv_Splus(t)
        total += len(toks)
        offsets[i + 1] = total
        for tok in toks:
            flat_list.append(get(tok, unk_id))
    flat = np.asarray(flat_list, dtype=np.int32)
    return flat, offsets


class PackedEncodedDataset(Dataset):
    def __init__(
        self,
        flat_ids: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
        ys=None,
    ):
        self.flat_t = torch.as_tensor(flat_ids, dtype=torch.long)  # 1D tensor
        self.offsets = offsets
        self.indices = indices
        self.ys = ys

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        s = int(self.offsets[i])
        e = int(self.offsets[i + 1])
        x = self.flat_t.narrow(0, s, e - s)  # view, no copy
        if self.ys is None:
            return x
        return x, int(self.ys[j])


@torch.no_grad()
def predict_proba_batched_from_packed(
    flat_ids_fold, offsets, indices, model, infer_bs=128
):
    dataset = PackedEncodedDataset(flat_ids_fold, offsets, indices, ys=None)
    loader = DataLoader(
        dataset,
        batch_size=infer_bs,
        shuffle=False,
        collate_fn=_pad_collate_xonly,
        num_workers=_DL_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_DL_PERSISTENT,
        prefetch_factor=4 if _DL_NUM_WORKERS > 0 else None,
        in_order=True,
    )
    out = []
    model.eval()
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu())
    return torch.cat(out, dim=0).numpy()


def train_one_fold_gru_from_packed(
    flat_ids_fold, offsets, indices, labels_for_indices, vocab_size, epochs=1, lr=2e-3
):
    model = GRUClassifier(vocab_size=vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = PackedEncodedDataset(
        flat_ids_fold, offsets, indices, ys=labels_for_indices
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=_pad_collate,
        num_workers=_DL_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_DL_PERSISTENT,
        prefetch_factor=4 if _DL_NUM_WORKERS > 0 else None,
        in_order=True,
    )

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

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

final_probs = np.zeros((len(final_df), 3), dtype=np.float64)

kfold_col = train["kfold"].to_numpy(dtype=np.int64)

global_vocab = _build_global_vocab_from_texts_fast(train_texts_all, _TOKEN_RE)
global_vocab_size = len(global_vocab)

train_flat_ids, train_offsets = _pack_texts_to_ids_fast(
    train_texts_all, _TOKEN_RE, global_vocab
)
final_flat_ids, final_offsets = _pack_texts_to_ids_fast(
    final_texts, _TOKEN_RE, global_vocab
)

fold_to_trn_idx = []
for k in range(kfolds):
    trn_idx = np.nonzero(kfold_col != k)[0]
    fold_to_trn_idx.append(trn_idx)


def _token_row_ids_from_offsets(offsets: np.ndarray) -> np.ndarray:
    lens = np.diff(offsets).astype(np.int64, copy=False)
    return np.repeat(np.arange(lens.shape[0], dtype=np.int32), lens)


train_token_row_ids = _token_row_ids_from_offsets(train_offsets)
assert train_token_row_ids.shape[0] == train_flat_ids.shape[0]


def _present_token_ids_by_fold_linear(
    flat_ids: np.ndarray,
    token_row_ids: np.ndarray,
    fold_col: np.ndarray,
    kfolds: int,
    vocab_size: int,
):
    masks = np.zeros(vocab_size, dtype=np.int32)

    occ_folds = 1 << fold_col[token_row_ids].astype(np.int32, copy=False)
    np.bitwise_or.at(masks, flat_ids.astype(np.int64, copy=False), occ_folds)

    all_folds_mask = (1 << kfolds) - 1
    out = []
    gids = np.arange(vocab_size, dtype=np.int32)
    not_all = masks != all_folds_mask
    gids_gt1 = gids > 1  # exclude pad/unk
    for k in range(kfolds):
        not_in_k = (masks & (1 << k)) == 0
        ok = not_in_k & not_all & gids_gt1
        out.append(gids[ok])
    return out


present_gids_per_fold = _present_token_ids_by_fold_linear(
    train_flat_ids, train_token_row_ids, kfold_col, kfolds, global_vocab_size
)

fold_vocab_sizes = []
fold_globalid_to_foldid = []

for kfold in range(kfolds):
    present_gids = present_gids_per_fold[kfold]

    globalid_to_foldid = np.ones(global_vocab_size, dtype=np.int32)
    globalid_to_foldid[0] = 0  # pad
    globalid_to_foldid[1] = 1  # unk
    next_id = 2

    for gid in present_gids:
        globalid_to_foldid[int(gid)] = next_id
        next_id += 1

    fold_globalid_to_foldid.append(globalid_to_foldid)
    fold_vocab_sizes.append(int(next_id))


fold_train_flat_remapped = []
fold_test_flat_remapped = []
for kfold in range(kfolds):
    m = fold_globalid_to_foldid[kfold]
    fold_train_flat_remapped.append(m[train_flat_ids].astype(np.int32, copy=False))
    fold_test_flat_remapped.append(m[final_flat_ids].astype(np.int32, copy=False))


class RemappedPackedDataset(Dataset):
    def __init__(
        self,
        flat_fold_ids: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
        ys=None,
    ):
        self.flat_t = torch.as_tensor(flat_fold_ids, dtype=torch.long)
        self.offsets = offsets
        self.indices = indices
        self.ys = ys

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        s = int(self.offsets[i])
        e = int(self.offsets[i + 1])
        x = self.flat_t.narrow(0, s, e - s)
        if self.ys is None:
            return x
        return x, int(self.ys[j])


@torch.no_grad()
def predict_proba_batched_from_remapped(
    flat_fold_ids, offsets, indices, model, infer_bs=128
):
    dataset = RemappedPackedDataset(flat_fold_ids, offsets, indices, ys=None)
    loader = DataLoader(
        dataset,
        batch_size=infer_bs,
        shuffle=False,
        collate_fn=_pad_collate_xonly,
        num_workers=_DL_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_DL_PERSISTENT,
        prefetch_factor=4 if _DL_NUM_WORKERS > 0 else None,
        in_order=True,
    )
    out = []
    model.eval()
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        out.append(probs.detach().cpu())
    return torch.cat(out, dim=0).numpy()


def train_one_fold_gru_from_remapped(
    flat_fold_ids,
    offsets,
    indices,
    labels_for_indices,
    vocab_size,
    epochs=1,
    lr=2e-3,
):
    model = GRUClassifier(vocab_size=vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = RemappedPackedDataset(
        flat_fold_ids, offsets, indices, ys=labels_for_indices
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=_pad_collate,
        num_workers=_DL_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_DL_PERSISTENT,
        prefetch_factor=4 if _DL_NUM_WORKERS > 0 else None,
        in_order=True,
    )

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


indices_all_test = np.arange(len(final_df), dtype=np.int64)

for kfold in range(kfolds):
    print(f"Prediction fold: {kfold}")

    trn_idx = fold_to_trn_idx[kfold]
    trn_mask = kfold_col != kfold  # keep identical label selection semantics

    vocab_size = fold_vocab_sizes[kfold]

    model_path = f"/kaggle/input/10050-training-5fold-lmsys/gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"

    if os.path.exists(model_path):
        model_loaded = GRUClassifier(vocab_size=vocab_size).to(device)
        state = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(state)
        model_loaded.eval()
    else:
        print(f"  Checkpoint not found, training fallback model for fold {kfold}...")

        train_y_fold = y_class[trn_mask]

        model_loaded = train_one_fold_gru_from_remapped(
            flat_fold_ids=fold_train_flat_remapped[kfold],
            offsets=train_offsets,
            indices=trn_idx,
            labels_for_indices=train_y_fold,
            vocab_size=vocab_size,
            epochs=1,  # unchanged fallback behavior
            lr=2e-3,
        )

    probs = predict_proba_batched_from_remapped(
        flat_fold_ids=fold_test_flat_remapped[kfold],
        offsets=final_offsets,
        indices=indices_all_test,
        model=model_loaded,
        infer_bs=256,
    )
    final_probs += probs / kfolds

eps = 1e-15
final_probs = np.clip(final_probs, eps, 1.0)
final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
final_df[["winner_model_a", "winner_model_b", "winner_tie"]] = final_probs



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

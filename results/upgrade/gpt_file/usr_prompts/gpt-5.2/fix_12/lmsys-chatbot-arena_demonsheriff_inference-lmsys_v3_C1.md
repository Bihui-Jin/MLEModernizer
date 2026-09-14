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
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn



## === cell 1
DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"

final_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

kfolds = 5
rng = np.random.RandomState(42)
perm = rng.permutation(len(train))
fold_ids = np.empty(len(train), dtype=np.int64)
fold_ids[perm] = np.arange(len(train)) % kfolds
train["kfold"] = fold_ids

final_df.head()



## === cell 2
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



## === cell 3
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
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        x = self.fc(h_last)
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def make_fast_collate_fn_prepadded_with_lengths():
    def collate(batch):
        xs, lens, ys = zip(*batch)
        out_x = torch.stack(xs, dim=0)
        out_l = torch.tensor(lens, dtype=torch.long)
        out_y = torch.tensor(ys, dtype=torch.long)
        return out_x, out_l, out_y

    return collate


class EncodedDataset(Dataset):
    def __init__(self, encoded_sequences, lengths, labels):
        self.encoded_sequences = encoded_sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        x = self.encoded_sequences[idx]
        l = int(self.lengths[idx])
        if self.labels is None:
            y = -1
        else:
            y = int(self.labels[idx])
        return x, l, y


class EncodedSubset(Dataset):
    def __init__(self, base_sequences, base_lengths, base_labels, indices):
        self.base_sequences = base_sequences
        self.base_lengths = base_lengths
        self.base_labels = base_labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.base_sequences[idx]
        l = int(self.base_lengths[idx])
        y = int(self.base_labels[idx])
        return x, l, y


_FAST_COLLATE = make_fast_collate_fn_prepadded_with_lengths()


def predict_proba_batched(
    model, encoded_sequences, lengths, batch_size=32, num_workers=0
):
    persistent = num_workers > 0
    ds = EncodedDataset(
        encoded_sequences=encoded_sequences, lengths=lengths, labels=None
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_FAST_COLLATE,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
        drop_last=False,
    )
    all_probs = []
    model.eval()
    with torch.inference_mode():
        for xb, lb, _ in dl:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            logits = model(xb, lengths=lb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def train_one_fold(
    model,
    train_ds,
    epochs=1,
    lr=1e-3,
    num_workers=0,
    generator=None,
):
    persistent = num_workers > 0
    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_FAST_COLLATE,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
        drop_last=False,
    )

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, lb, yb in dl:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lengths=lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model




## === cell 4
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

train_texts_all = train["text"].values
test_texts_all = final_texts

_tok_cache = {}


def _ws_tokens_cached(s: str):
    s = str(s)
    r = _tok_cache.get(s)
    if r is None:
        r = s.split()
        _tok_cache[s] = r
    return r


train_tokens = [_ws_tokens_cached(t) for t in train_texts_all]
test_tokens = [_ws_tokens_cached(t) for t in test_texts_all]


def build_vocab_and_lengths_from_tokens(train_toks, test_toks, min_freq=1):
    ctr = Counter()
    train_lens = np.fromiter(
        (len(t) for t in train_toks), dtype=np.int32, count=len(train_toks)
    )
    test_lens = np.fromiter(
        (len(t) for t in test_toks), dtype=np.int32, count=len(test_toks)
    )
    for toks in train_toks:
        ctr.update(toks)
    for toks in test_toks:
        ctr.update(toks)

    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in ctr.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab, train_lens, test_lens


MAX_LEN_CAP = 512
PCTL = 99.0


def choose_max_len(train_lens, test_lens):
    combined = np.concatenate([train_lens, test_lens])
    p = int(np.percentile(combined, PCTL))
    mx = int(combined.max())
    return min(MAX_LEN_CAP, max(1, min(mx, p)))


def encode_tokens_to_padded_tensor(tokens_list, vocab, max_len, pad_idx=0):
    unk = 1
    get = vocab.get
    n = len(tokens_list)
    out_np = np.full((n, max_len), pad_idx, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, toks in enumerate(tokens_list):
        L = len(toks)
        if L > max_len:
            L = max_len
        lengths[i] = L
        if L:
            row = out_np[i]
            for j in range(L):
                row[j] = get(toks[j], unk)
    out = torch.from_numpy(out_np)
    return out, lengths


vocab, train_lens, test_lens = build_vocab_and_lengths_from_tokens(
    train_tokens, test_tokens, min_freq=1
)
max_len = choose_max_len(train_lens, test_lens)
print("max_len_used:", max_len, "vocab:", len(vocab))

encoded_train_all, train_lengths_all = encode_tokens_to_padded_tensor(
    train_tokens, vocab, max_len=max_len, pad_idx=0
)
encoded_test, test_lengths = encode_tokens_to_padded_tensor(
    test_tokens, vocab, max_len=max_len, pad_idx=0
)

label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
y_all = train[label_cols].values.argmax(axis=1)

epochs_to_train = 1

cpu_workers = 0

all_test_probs = []

kfold_arr = train["kfold"].values
for kfold in range(kfolds):
    print(f"fold: {kfold}")

    tr_idx = np.flatnonzero(kfold_arr != kfold).astype(np.int64, copy=False)

    train_ds = EncodedSubset(
        base_sequences=encoded_train_all,
        base_lengths=train_lengths_all,
        base_labels=y_all,
        indices=tr_idx,
    )

    model_loaded = GRUClassifier(vocab_size=len(vocab), pad_idx=0)

    if hasattr(torch, "compile"):
        try:
            model_loaded = torch.compile(
                model_loaded, mode="reduce-overhead", fullgraph=False
            )
        except Exception:
            pass

    gen = torch.Generator()
    gen.manual_seed(seed + kfold)

    model_loaded = train_one_fold(
        model_loaded,
        train_ds=train_ds,
        epochs=epochs_to_train,
        lr=1e-3,
        num_workers=cpu_workers,
        generator=gen,
    )

    probs = predict_proba_batched(
        model_loaded,
        encoded_sequences=encoded_test,
        lengths=test_lengths,
        batch_size=64,  # safe throughput increase; identical semantics
        num_workers=cpu_workers,
    )
    all_test_probs.append(probs)

probs = np.mean(np.stack(all_test_probs, axis=0), axis=0)  # (n_test, 3)

probs = np.clip(probs, 1e-6, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

final_df["winner_model_a"] = probs[:, 0]
final_df["winner_model_b"] = probs[:, 1]
final_df["winner_tie"] = probs[:, 2]

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## === cell 5
sub_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    sub_path, index=False
)

chk = pd.read_csv(sub_path)
assert list(chk.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
s = chk[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
assert np.allclose(s, 1.0, atol=1e-6), (s.min(), s.max())
assert len(chk) == len(final_df) == 5748
print("Wrote:", sub_path, "rows:", len(chk))

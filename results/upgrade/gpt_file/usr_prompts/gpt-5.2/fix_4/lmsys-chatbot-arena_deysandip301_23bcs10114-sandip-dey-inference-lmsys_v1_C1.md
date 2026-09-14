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
from torch.nn.utils.rnn import pad_sequence
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

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

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        output = self.fc(h_n[-1])  # final hidden state
        logits = self.fc2(output)
        return logits


def build_vocab_from_token_lists(token_lists):
    vocab = {"<pad>": 0, "<unk>": 1}
    c = Counter()
    for toks in token_lists:
        c.update(toks)
    for w in c:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode_tokens_to_tensor(tokens, vocab):
    get = vocab.get
    return torch.tensor([get(w, 1) for w in tokens], dtype=torch.long)


class EncodedDataset(Dataset):
    def __init__(self, encoded_list, labels, indices):
        self.encoded_list = encoded_list
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.encoded_list[idx], int(self.labels[idx])


class EncodedInferenceDataset(Dataset):
    def __init__(self, encoded_list):
        self.encoded_list = encoded_list

    def __len__(self):
        return len(self.encoded_list)

    def __getitem__(self, idx):
        return self.encoded_list[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_padded, ys


def collate_infer(batch):
    xs_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return xs_padded


def seed_worker(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)




## === cell 3
n = len(train)
idx = np.arange(n)
np.random.shuffle(idx)
fold_id = np.zeros(n, dtype=np.int64)
fold_id[idx] = np.arange(n) % kfolds
train["kfold"] = fold_id

train_texts = train["text"].tolist()
test_texts = final_df["text"].tolist()

train_tokens_all = [t.split() for t in train_texts]
final_tokens_all = [t.split() for t in test_texts]

print("Fold counts:", pd.Series(train["kfold"]).value_counts().sort_index().to_dict())

global_vocab = build_vocab_from_token_lists(train_tokens_all + final_tokens_all)
print("Global vocab size:", len(global_vocab))

train_encoded_all = [
    encode_tokens_to_tensor(toks, global_vocab) for toks in train_tokens_all
]
test_encoded_all = [
    encode_tokens_to_tensor(toks, global_vocab) for toks in final_tokens_all
]
train_labels_all = train["label"].values.astype(np.int64)

train_lens = np.fromiter(
    (len(x) for x in train_encoded_all), dtype=np.int32, count=len(train_encoded_all)
)



## === cell 4
EPOCHS = 1
LR = 1e-3

all_fold_probs = []

num_workers = min(4, (os.cpu_count() or 2))
g = torch.Generator()
g.manual_seed(SEED)

for kfold in range(kfolds):
    print(f"\n=== Fold {kfold} ===")

    trn_mask = train["kfold"].values != kfold
    val_mask = ~trn_mask

    trn_idx = np.flatnonzero(trn_mask)
    val_idx = np.flatnonzero(val_mask)  # kept for parity (not used later)

    trn_idx_sorted = trn_idx[np.argsort(train_lens[trn_idx], kind="mergesort")]

    model = RNNClassifier(vocab_size=len(global_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trn_ds = EncodedDataset(train_encoded_all, train_labels_all, trn_idx_sorted)
    trn_dl = DataLoader(
        trn_ds,
        batch_size=batch_size,
        shuffle=False,  # deterministic; order is already "length-sorted" for speed
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(trn_dl, desc=f"train epoch {epoch+1}/{EPOCHS}", leave=False)
        running = 0.0
        seen = 0
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(loss=running / max(1, seen))

    model.eval()
    test_ds = EncodedInferenceDataset(test_encoded_all)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_infer,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    probs_fold = []
    with torch.no_grad():
        for xb in tqdm(test_dl, desc="predict test", leave=False):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_fold.append(probs)

    probs_fold = np.vstack(probs_fold)  # [n_test, 3]
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

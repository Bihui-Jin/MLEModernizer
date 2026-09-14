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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

final_df = pd.read_csv(test_path)
train = pd.read_csv(train_path)

kfolds = 5
if "kfold" not in train.columns:
    idx = np.arange(len(train))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    fold_id = np.zeros(len(train), dtype=np.int64)
    fold_id[idx] = np.arange(len(train)) % kfolds
    train["kfold"] = fold_id

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
train["text"] = (
    "User prompt: "
    + train["prompt"].astype(str)
    + "\n\nModel A :\n"
    + train["response_a"].astype(str)
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"].astype(str)
)

print(final_df["text"].iloc[0])




## === cell 2
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
        h_last = h_n[-1]
        x = self.fc(h_last)
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def build_vocab_from_texts_stream(texts_iterable):
    vocab = {"<pad>": 0, "<unk>": 1}
    counts = Counter()
    for t in texts_iterable:
        counts.update(str(t).split())
    for word, _ in sorted(counts.items(), key=lambda x: x[0]):
        vocab[word] = len(vocab)
    return vocab


def encode_tokenlist(tokens, vocab):
    unk = 1
    get = vocab.get
    return [get(w, unk) for w in tokens]


def pad_sequences_to_fixed_numpy(seqs, pad_idx=0, max_len=None):
    bs = len(seqs)
    if bs == 0:
        return np.empty((0, 1), dtype=np.int64), np.empty((0,), dtype=np.int32)
    lengths = np.fromiter((len(s) for s in seqs), count=bs, dtype=np.int32)
    if max_len is None:
        max_len = int(lengths.max()) if bs else 1
    x_np = np.full((bs, max_len), pad_idx, dtype=np.int64)
    for i, s in enumerate(seqs):
        if s:
            ss = s[:max_len]  # defensive; should not truncate when max_len=max(lengths)
            x_np[i, : len(ss)] = ss
    return x_np, lengths


@torch.no_grad()
def predict_proba_from_tensor(model, x_tensor, batch_size=8):
    model.eval()
    probs_all = []
    n = x_tensor.shape[0]
    pin = device.type == "cuda"
    for start in range(0, n, batch_size):
        x = x_tensor[start : start + batch_size]
        if pin:
            x = x.pin_memory()
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)


train_texts = train["text"].values
test_texts = final_df["text"].values

vocab = build_vocab_from_texts_stream(np.concatenate([train_texts, test_texts]))

train_tokens = [str(t).split() for t in train_texts]
test_tokens = [str(t).split() for t in test_texts]

train_encoded = [encode_tokenlist(tok, vocab) for tok in train_tokens]
test_encoded = [encode_tokenlist(tok, vocab) for tok in test_tokens]

train_lens = np.fromiter(
    (len(s) for s in train_encoded), count=len(train_encoded), dtype=np.int32
)
test_lens = np.fromiter(
    (len(s) for s in test_encoded), count=len(test_encoded), dtype=np.int32
)
MAX_LEN = int(max(train_lens.max(initial=1), test_lens.max(initial=1)))
print("Global MAX_LEN:", MAX_LEN)

train_x_np, _ = pad_sequences_to_fixed_numpy(train_encoded, pad_idx=0, max_len=MAX_LEN)
test_x_np, _ = pad_sequences_to_fixed_numpy(test_encoded, pad_idx=0, max_len=MAX_LEN)

train_x = torch.from_numpy(train_x_np)
test_x = torch.from_numpy(test_x_np)

del train_x_np, test_x_np, train_tokens, test_tokens  # free memory early




## === cell 3
def get_labels(df):
    y = df[["winner_model_a", "winner_model_b", "winner_tie"]].values
    return np.argmax(y, axis=1).astype(np.int64)


def train_one_fold_preencoded(
    train_indices,
    train_x_all,  # pre-padded full tensor (N, MAX_LEN)
    labels_all,
    vocab_size,
    epochs=1,
    lr=2e-3,
    batch_size=8,
):
    model = GRUClassifier(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_tensor = train_x_all.index_select(
        0, torch.as_tensor(train_indices, dtype=torch.long)
    )
    y_np = labels_all[train_indices].astype(np.int64, copy=False)
    y_tensor = torch.from_numpy(y_np)

    ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    model.train()
    n = len(ds)
    pin = device.type == "cuda"
    for ep in range(epochs):
        g = torch.Generator()
        g.manual_seed(SEED + ep)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,  # deterministic, avoids worker overhead
            pin_memory=pin,
            drop_last=False,
        )

        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * xb.size(0)

        avg_loss = total_loss / max(1, n)
        print(f"  epoch {ep+1}/{epochs} - loss: {avg_loss:.4f}")

    return model


epoches_to_use = [0, 0, 0, 0, 0]  # kept for compatibility, not used for loading anymore

class_0_probs = []
class_1_probs = []
class_2_probs = []

EPOCHS = 1
LR = 2e-3

labels_all = get_labels(train)
kfold_arr = train["kfold"].to_numpy(dtype=np.int64)

for kfold in range(kfolds):
    print(f"training+prediction fold: {kfold}")

    train_mask = kfold_arr != kfold
    train_indices = np.nonzero(train_mask)[0].astype(np.int64, copy=False)

    model_loaded = train_one_fold_preencoded(
        train_indices=train_indices,
        train_x_all=train_x,
        labels_all=labels_all,
        vocab_size=len(vocab),
        epochs=EPOCHS,
        lr=LR,
        batch_size=batch_size,
    )

    probs = predict_proba_from_tensor(
        model=model_loaded,
        x_tensor=test_x,
        batch_size=batch_size,
    )

    class_0_probs.append(probs[:, 0].tolist())
    class_1_probs.append(probs[:, 1].tolist())
    class_2_probs.append(probs[:, 2].tolist())

print("Collected fold predictions:", len(class_0_probs), "folds")




## === cell 4
p0 = np.mean(np.array(class_0_probs), axis=0)
p1 = np.mean(np.array(class_1_probs), axis=0)
p2 = np.mean(np.array(class_2_probs), axis=0)

eps = 1e-7
probs = np.vstack([p0, p1, p2]).T
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

final_df["winner_model_a"] = probs[:, 0]
final_df["winner_model_b"] = probs[:, 1]
final_df["winner_tie"] = probs[:, 2]

sub_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    sub_path, index=False
)

chk = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
print("submission path:", sub_path)
print("row-sum min/max:", float(chk.min()), float(chk.max()))
print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())
print(
    "submission shape:",
    final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].shape,
)

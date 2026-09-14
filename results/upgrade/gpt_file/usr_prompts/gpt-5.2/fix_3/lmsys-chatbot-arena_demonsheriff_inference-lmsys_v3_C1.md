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
from tqdm import tqdm
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


def encode(tokens, vocab):
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = str(self.texts[idx]).split()
        x = encode(tokens, self.vocab)
        if self.labels is None:
            y = -1
        else:
            y = int(self.labels[idx])
        return x, y


def make_collate_fn(pad_idx=0):
    def collate(batch):
        xs, ys = zip(*batch)
        xs = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    return collate


def build_vocab_from_tokenized(list_of_token_lists, min_freq=1):
    ctr = Counter()
    for toks in list_of_token_lists:
        ctr.update(toks)
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in ctr.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def predict_proba_batched(model, texts, vocab, batch_size=32):
    ds = TextDataset(texts=texts, labels=None, vocab=vocab)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(pad_idx=0)
    )
    all_probs = []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def train_one_fold(model, train_texts, train_labels, vocab, epochs=1, lr=1e-3):
    ds = TextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(pad_idx=0)
    )

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
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

label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
y_all = train[label_cols].values.argmax(axis=1)

epochs_to_train = 1

all_test_probs = []

for kfold in range(kfolds):
    print(f"fold: {kfold}")

    tr_idx = train["kfold"].values != kfold
    va_idx = train["kfold"].values == kfold

    train_texts = train.loc[tr_idx, "text"].values
    train_labels = y_all[tr_idx]

    train_tokenized = [str(t).split() for t in train_texts]
    test_tokenized = [str(t).split() for t in final_texts]
    vocab = build_vocab_from_tokenized(train_tokenized + test_tokenized, min_freq=1)

    model_loaded = GRUClassifier(vocab_size=len(vocab))

    model_loaded = train_one_fold(
        model_loaded,
        train_texts=train_texts,
        train_labels=train_labels,
        vocab=vocab,
        epochs=epochs_to_train,
        lr=1e-3,
    )

    probs = predict_proba_batched(model_loaded, final_texts, vocab=vocab, batch_size=32)
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

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    alt = "/kaggle/input/lmsys-chatbot-arena/lmsys-chatbot-arena"
    if os.path.exists(os.path.join(alt, "train.csv")):
        DATA_DIR = alt

final_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

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
print(len(final_df))
final_df.head()



## === cell 3
kfolds = 5
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
fold_id = np.zeros(len(train), dtype=np.int64)
fold_id[idx] = np.arange(len(train)) % kfolds
train["kfold"] = fold_id

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


MAX_LEN = 256


def encode(tokens):
    ids = [vocab.get(w, 1) for w in tokens][:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids = ids + [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)  # (1, L)
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        return probs


def build_vocab_from_tokens(token_lists_a, token_lists_b):
    vocab_local = {"<pad>": 0, "<unk>": 1}
    for toks in token_lists_a:
        for w in toks:
            if w not in vocab_local:
                vocab_local[w] = len(vocab_local)
    for toks in token_lists_b:
        for w in toks:
            if w not in vocab_local:
                vocab_local[w] = len(vocab_local)
    return vocab_local


def build_vocab(texts_a, texts_b):
    vocab_local = {"<pad>": 0, "<unk>": 1}
    for t in texts_a:
        for w in t.split():
            if w not in vocab_local:
                vocab_local[w] = len(vocab_local)
    for t in texts_b:
        for w in t.split():
            if w not in vocab_local:
                vocab_local[w] = len(vocab_local)
    return vocab_local


def get_labels_from_train_df(df):
    y = df[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(dtype=np.int64)
    return torch.tensor(np.argmax(y, axis=1), dtype=torch.long)


def train_one_fold(model, train_texts_fold, y_fold, epochs=1, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(train_texts_fold)
    indices = np.arange(n)

    for ep in range(epochs):
        rng_local = np.random.RandomState(SEED + 1000 + ep)
        rng_local.shuffle(indices)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_texts = train_texts_fold[batch_idx]
            xb = torch.stack([encode(t.split()) for t in batch_texts], dim=0).to(device)
            yb = y_fold[batch_idx].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)

        print(f"  epoch {ep+1}/{epochs} - loss: {total_loss / max(n,1):.5f}")

    model.eval()
    return model


def encode_texts_to_tensor_from_tokens(token_lists, vocab_local, max_len=MAX_LEN):
    n = len(token_lists)
    out = np.zeros((n, max_len), dtype=np.int64)  # pad=0
    unk = 1
    get = vocab_local.get
    for i in range(n):
        toks = token_lists[i]
        if not toks:
            continue
        m = len(toks)
        if m > max_len:
            m = max_len
        row = out[i]
        for j in range(m):
            row[j] = get(toks[j], unk)
    return torch.from_numpy(out)


def encode_texts_to_tensor(texts, vocab_local, max_len=MAX_LEN):
    n = len(texts)
    out = np.zeros((n, max_len), dtype=np.int64)  # pad=0
    unk = 1
    get = vocab_local.get
    for i in range(n):
        toks = texts[i].split()
        if not toks:
            continue
        m = len(toks)
        if m > max_len:
            m = max_len
        row = out[i]
        for j in range(m):
            row[j] = get(toks[j], unk)
    return torch.from_numpy(out)


def predict_proba_batched(model, encoded_tensor, infer_bs=256):
    model.eval()
    n = encoded_tensor.shape[0]
    probs_all = np.empty((n, 3), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, infer_bs):
            xb = encoded_tensor[i : i + infer_bs].to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all[i : i + infer_bs] = probs
    return probs_all




## === cell 4
epoches_to_use = [0, 0, 0, 0, 0]  # kept as provided

class_0_probs = []
class_1_probs = []
class_2_probs = []

MODEL_DIR = "/kaggle/input/gru-train-5fold"

y_all = get_labels_from_train_df(train)

INFER_BS = 256 if device.type == "cuda" else 64

train_text_all = train["text"].values
kfold_arr = train["kfold"].values

train_tokens_all = [t.split() for t in train_text_all]
final_tokens_all = [t.split() for t in final_texts]

fold_to_train_tokens = []
fold_to_test_tokens = []
fold_to_y = []
for kfold in range(kfolds):
    mask_test = kfold_arr == kfold
    fold_to_test_tokens.append([train_tokens_all[i] for i in np.flatnonzero(mask_test)])
    fold_to_train_tokens.append(
        [train_tokens_all[i] for i in np.flatnonzero(~mask_test)]
    )
    fold_to_y.append(y_all[~mask_test])

encoded_final_cache = {}

for kfold in range(kfolds):
    print(f"prediction fold: {kfold}")

    test_tokens_fold = fold_to_test_tokens[kfold]
    train_tokens_fold = fold_to_train_tokens[kfold]
    y_fold = fold_to_y[kfold]

    vocab = build_vocab_from_tokens(test_tokens_fold, train_tokens_fold)

    model_path = os.path.join(
        MODEL_DIR, f"gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    )

    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(state)
        model_loaded.eval()
    else:
        print(f"  missing weights: {model_path}")
        print(
            "  fallback: training fold model on-the-fly (1 epoch) to enable inference."
        )
        train_texts_fold = train_text_all[~(kfold_arr == kfold)]
        model_loaded = train_one_fold(
            model_loaded,
            train_texts_fold=train_texts_fold,
            y_fold=y_fold,
            epochs=1,
            lr=1e-3,
        )

    if kfold not in encoded_final_cache:
        encoded_final_cache[kfold] = encode_texts_to_tensor_from_tokens(
            final_tokens_all, vocab, MAX_LEN
        )
    encoded_final = encoded_final_cache[kfold]

    probs = predict_proba_batched(model_loaded, encoded_final, infer_bs=INFER_BS)

    class_0_probs.append(probs[:, 0].tolist())
    class_1_probs.append(probs[:, 1].tolist())
    class_2_probs.append(probs[:, 2].tolist())



## === cell 5
n_test = len(final_df)
if len(class_0_probs) == 0:
    preds = np.full((n_test, 3), 1.0 / 3.0, dtype=np.float64)
else:
    p0 = np.mean(np.array(class_0_probs, dtype=np.float64), axis=0)
    p1 = np.mean(np.array(class_1_probs, dtype=np.float64), axis=0)
    p2 = np.mean(np.array(class_2_probs, dtype=np.float64), axis=0)
    preds = np.stack([p0, p1, p2], axis=1)  # (n_test, 3)

eps = 1e-12
preds = np.clip(preds, eps, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

final_df["winner_model_a"] = preds[:, 0]
final_df["winner_model_b"] = preds[:, 1]
final_df["winner_tie"] = preds[:, 2]

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## === cell 6
sub_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    sub_path, index=False
)

check = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy()
row_sums = check.sum(axis=1)
print("Saved:", sub_path)
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())
print("Done")

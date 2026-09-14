# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

1.103134364358877

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
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

TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
TRAIN_PATH = "/kaggle/input/lmsys-chatbot-arena/train.csv"
CKPT_DIR = "/kaggle/input/training-5fold-lmsys"

final_df = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)

final_df.head()



## === cell 1
kfolds = 5
train = train.copy()
train["kfold"] = (train["id"].astype(np.int64) % kfolds).astype(int)

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
        output = self.fc(h_n[-1])
        logits = self.fc2(output)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence], dtype=torch.long)


def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        return probs




## === cell 4
def find_checkpoint(ckpt_dir, kfold, epoch):
    expected = os.path.join(
        ckpt_dir, f"lstm_classifier_kfold_{kfold}_epoch_{epoch}.pth"
    )
    if os.path.exists(expected):
        return expected

    patterns = [
        os.path.join(ckpt_dir, f"**/*kfold*{kfold}*epoch*{epoch}*.pth"),
        os.path.join(ckpt_dir, f"**/*fold*{kfold}*epoch*{epoch}*.pth"),
        os.path.join(ckpt_dir, f"**/*{kfold}*{epoch}*.pth"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    matches = sorted(set(matches))
    return matches[0] if matches else None


epoches_to_use = [0, 0, 0, 0, 0]

class_0_probs = []
class_1_probs = []
class_2_probs = []

for kfold in range(kfolds):
    print(f"prediction fold: {kfold}")

    test_texts = train[train["kfold"] == kfold]["text"].values
    train_texts = train[train["kfold"] != kfold]["text"].values

    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]

    print(len(test_tokenized) + len(train_tokenized))

    vocab = {"<pad>": 0, "<unk>": 1}

    for word in Counter(w for sent in test_tokenized for w in sent):
        vocab[word] = len(vocab)

    for word in Counter(w for sent in train_tokenized for w in sent):
        vocab[word] = len(vocab)

    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

    ckpt_path = find_checkpoint(CKPT_DIR, kfold, epoches_to_use[kfold])
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find checkpoint for kfold={kfold}, epoch={epoches_to_use[kfold]} in {CKPT_DIR}. "
            f"Available sample: {glob.glob(os.path.join(CKPT_DIR, '**/*.pth'), recursive=True)[:5]}"
        )

    state = torch.load(ckpt_path, map_location=device)
    model_loaded.load_state_dict(state)
    model_loaded.eval()

    class_0_prob = []
    class_1_prob = []
    class_2_prob = []

    for text in final_texts:
        ans = predict(text)
        class_0_prob.append(float(ans[0][0]))
        class_1_prob.append(float(ans[0][1]))
        class_2_prob.append(float(ans[0][2]))

    class_0_probs.append(class_0_prob)
    class_1_probs.append(class_1_prob)
    class_2_probs.append(class_2_prob)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1913300993.py in <cell line: 0>()
     50     ckpt_path = find_checkpoint(CKPT_DIR, kfold, epoches_to_use[kfold])
     51     if ckpt_path is None:
---> 52         raise FileNotFoundError(
     53             f"Could not find checkpoint for kfold={kfold}, epoch={epoches_to_use[kfold]} in {CKPT_DIR}. "
     54             f"Available sample: {glob.glob(os.path.join(CKPT_DIR, '**/*.pth'), recursive=True)[:5]}"

FileNotFoundError: Could not find checkpoint for kfold=0, epoch=0 in /kaggle/input/training-5fold-lmsys. Available sample: []

## === cell 5
p0 = np.sum(np.array(class_0_probs), axis=0) / kfolds
p1 = np.sum(np.array(class_1_probs), axis=0) / kfolds
p2 = np.sum(np.array(class_2_probs), axis=0) / kfolds

probs = np.vstack([p0, p1, p2]).T

eps = 1e-12
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

final_df["winner_model_a"] = probs[:, 0]
final_df["winner_model_b"] = probs[:, 1]
final_df["winner_tie"] = probs[:, 2]

row_sums = (
    final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
)
print("Row sum min/max:", row_sums.min(), row_sums.max())

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/296183501.py in <cell line: 0>()
     11 probs = probs / probs.sum(axis=1, keepdims=True)
     12 
---> 13 final_df["winner_model_a"] = probs[:, 0]
     14 final_df["winner_model_b"] = probs[:, 1]
     15 final_df["winner_tie"] = probs[:, 2]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (1) does not match length of index (5748)

## === cell 6
sub_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    sub_path, index=False
)
print("Wrote:", sub_path)
print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1165400978.py in <cell line: 0>()
      1 # Write submission
      2 sub_path = "submission.csv"
----> 3 final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
      4     sub_path, index=False
      5 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['winner_model_a', 'winner_model_b', 'winner_tie'] not in index"

## === cell 7
print(final_df)

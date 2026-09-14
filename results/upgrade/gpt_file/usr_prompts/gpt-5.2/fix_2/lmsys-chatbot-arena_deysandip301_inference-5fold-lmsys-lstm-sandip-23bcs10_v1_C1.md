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

1.154269340976232

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from tqdm import tqdm

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

final_df = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train:", train.shape, "test:", final_df.shape, "sample:", sample_sub.shape)
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


def encode(tokens):
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)


@torch.no_grad()
def predict(model_loaded, text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)
    logits = model_loaded(encoded)
    probs = F.softmax(logits, dim=1)
    return probs




## === cell 3
CKPT_DIR = "/kaggle/input/training-5fold-lmsys-lstm-sandip-dey-23bcs10114"
class_0_probs, class_1_probs, class_2_probs = [], [], []

for kfold in range(kfolds):
    ckpt_path = os.path.join(CKPT_DIR, f"lstm_classifier_kfold_{kfold}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"prediction fold: {kfold}  ckpt: {os.path.basename(ckpt_path)}")

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

    class_0_probs.append(c0)
    class_1_probs.append(c1)
    class_2_probs.append(c2)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2676474876.py in <cell line: 0>()
      6     ckpt_path = os.path.join(CKPT_DIR, f"lstm_classifier_kfold_{kfold}_best.pth")
      7     if not os.path.exists(ckpt_path):
----> 8         raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
      9 
     10     print(f"prediction fold: {kfold}  ckpt: {os.path.basename(ckpt_path)}")

FileNotFoundError: Missing checkpoint: /kaggle/input/training-5fold-lmsys-lstm-sandip-dey-23bcs10114/lstm_classifier_kfold_0_best.pth

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
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1113948087.py in <cell line: 0>()
     10 P = P / row_sums
     11 
---> 12 final_df["winner_model_a"] = P[:, 0]
     13 final_df["winner_model_b"] = P[:, 1]
     14 final_df["winner_tie"] = P[:, 2]

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

## === cell 5
sub = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()

sub = sample_sub[["id"]].merge(sub, on="id", how="left")
if sub[["winner_model_a", "winner_model_b", "winner_tie"]].isna().any().any():
    raise ValueError(
        "NaNs found in submission after aligning with sample_submission ids."
    )

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/4224578703.py in <cell line: 0>()
      1 # Ensure correct column order and ids aligned; write valid CSV submission
----> 2 sub = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()
      3 
      4 # Optional: align to sample_submission ids ordering if needed (score-neutral, format-robust)
      5 sub = sample_sub[["id"]].merge(sub, on="id", how="left")

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

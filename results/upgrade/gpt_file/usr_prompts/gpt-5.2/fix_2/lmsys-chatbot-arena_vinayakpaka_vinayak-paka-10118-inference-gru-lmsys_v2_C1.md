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

1.1023131124146155

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

BASE = "/kaggle/input/lmsys-chatbot-arena"
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sub_path = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(train_path)
final_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

print(train.shape, final_df.shape, sample_sub.shape)
print(train.columns.tolist())
print(final_df.columns.tolist())



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
kfold = 0
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

n_val = int(0.1 * len(train))
val_idx = idx[:n_val]
tr_idx = idx[n_val:]

test_texts = train.iloc[val_idx][
    "text"
].values  # naming preserved from original notebook
train_texts = train.iloc[tr_idx]["text"].values

final_texts = final_df["text"].values

print("split sizes:", len(train_texts), len(test_texts), "final:", len(final_texts))



## === cell 3
batch_size = 8
num_classes = 3

test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]
print("tokenized:", len(test_tokenized) + len(train_tokenized))

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)

pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]


def encode(tokens):
    return torch.tensor([vocab.get(w, unk_id) for w in tokens], dtype=torch.long)


print("vocab size:", len(vocab))




## === cell 4
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
        output = self.fc(h_n[-1])  # final hidden state
        logits = self.fc2(output)
        return logits




## === cell 5
weights_path = (
    "/kaggle/input/vinayak-paka-10118-training-gru-lmsys/GRU_classifier_1.pth"
)
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")

model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
state = torch.load(weights_path, map_location="cpu")
model_loaded.load_state_dict(state)
model_loaded.eval()

print("Loaded weights from:", weights_path)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/646670581.py in <cell line: 0>()
      4 )
      5 if not os.path.exists(weights_path):
----> 6     raise FileNotFoundError(f"Model weights not found at: {weights_path}")
      7 
      8 model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

FileNotFoundError: Model weights not found at: /kaggle/input/vinayak-paka-10118-training-gru-lmsys/GRU_classifier_1.pth

## === cell 6
@torch.no_grad()
def predict_batch(text_list):
    encoded_list = [encode(t.split()) for t in text_list]
    encoded_list = [
        e if e.numel() > 0 else torch.tensor([unk_id], dtype=torch.long)
        for e in encoded_list
    ]
    x = pad_sequence(encoded_list, batch_first=True, padding_value=pad_id).to(device)
    logits = model_loaded(x)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


probs_all = []
for i in range(0, len(final_texts), batch_size):
    batch_texts = final_texts[i : i + batch_size]
    probs_all.append(predict_batch(batch_texts))

probs_all = np.vstack(probs_all)
print(
    "probs shape:",
    probs_all.shape,
    "row-sum min/max:",
    probs_all.sum(axis=1).min(),
    probs_all.sum(axis=1).max(),
)

class_0_prob = probs_all[:, 0].astype(float).tolist()
class_1_prob = probs_all[:, 1].astype(float).tolist()
class_2_prob = probs_all[:, 2].astype(float).tolist()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3602273879.py in <cell line: 0>()
     17 for i in range(0, len(final_texts), batch_size):
     18     batch_texts = final_texts[i : i + batch_size]
---> 19     probs_all.append(predict_batch(batch_texts))
     20 
     21 probs_all = np.vstack(probs_all)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/3602273879.py in predict_batch(text_list)
      9     ]
     10     x = pad_sequence(encoded_list, batch_first=True, padding_value=pad_id).to(device)
---> 11     logits = model_loaded(x)
     12     probs = F.softmax(logits, dim=1)
     13     return probs.detach().cpu().numpy()

NameError: name 'model_loaded' is not defined

## === cell 7
if not (len(class_0_prob) == len(final_df) == len(class_1_prob) == len(class_2_prob)):
    raise ValueError("Prediction length mismatch with test dataframe length.")

final_df["winner_model_a"] = class_0_prob
final_df["winner_model_b"] = class_1_prob
final_df["winner_tie"] = class_2_prob

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1127425847.py in <cell line: 0>()
      1 # Fix: ensure lengths match and columns exist; keep required submission column names
----> 2 if not (len(class_0_prob) == len(final_df) == len(class_1_prob) == len(class_2_prob)):
      3     raise ValueError("Prediction length mismatch with test dataframe length.")
      4 
      5 final_df["winner_model_a"] = class_0_prob

NameError: name 'class_0_prob' is not defined

## === cell 8
sub = sample_sub[["id"]].merge(
    final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]],
    on="id",
    how="left",
    validate="one_to_one",
)

if sub[["winner_model_a", "winner_model_b", "winner_tie"]].isna().any().any():
    raise ValueError(
        "Found missing predictions after merging with sample_submission ids."
    )

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2722691726.py in <cell line: 0>()
      1 # Fix: align to sample_submission ids/order to avoid any accidental misalignment; write valid CSV.
      2 sub = sample_sub[["id"]].merge(
----> 3     final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]],
      4     on="id",
      5     how="left",

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

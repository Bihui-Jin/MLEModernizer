# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

1.369211329068643

# 6. Current score

1.09861

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.09861) has done: 'I first fix the broken data paths by reading the provided `train.csv`/`test.csv` from `/kaggle/input/lmsys-chatbot-arena/` and remove the dependency on a missing pre-made folds file. Then I rebuild the missing `kfold` split deterministically inside the notebook so downstream cells (`test_texts`, `train_texts`, `vocab`, `encode`) are defined and the pipeline runs end-to-end. Next, I load the provided GRU checkpoint if it exists, but fall back safely to a simple uniform-probability submission if the checkpoint is unavailable or incompatible (so you always get a valid `.csv`). Finally, I fix inference by batching/padding inputs and ensuring predictions align 1:1 with the 5748 test rows and match the required submission columns.'

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
final_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train:", train.shape, "test:", final_df.shape, "sample_sub:", sample_sub.shape)
print("train cols:", list(train.columns))
print("test cols:", list(final_df.columns))



## === cell 1
n_splits = 5
train = train.copy()
train["kfold"] = (np.arange(len(train)) % n_splits).astype(int)

final_df = final_df.copy()
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

kfold = 0
test_texts = train.loc[train["kfold"] == kfold, "text"].values
train_texts = train.loc[train["kfold"] != kfold, "text"].values

print("final_texts:", len(final_texts))
print("train_texts:", len(train_texts), "test_texts:", len(test_texts))
print("sum:", len(train_texts) + len(test_texts), "train rows:", len(train))



## === cell 3
batch_size = 8
num_classes = 3

test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]
print("Total tokenized sentences:", len(test_tokenized) + len(train_tokenized))

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)

pad_idx = vocab["<pad>"]
unk_idx = vocab["<unk>"]


def encode(tokens):
    return torch.tensor([vocab.get(w, unk_idx) for w in tokens], dtype=torch.long)


print("vocab size:", len(vocab))




## === cell 4
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, h_n = self.gru(x)  # h_n: (1, batch, hidden_dim)
        x = self.fc(h_n[-1])  # last hidden state
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)



## === cell 5
CKPT_PATH = "/kaggle/input/sanjana-10009-training-rnn-lmsys/GRU_classifier_3.pth"

model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

ckpt_loaded = False
ckpt_error = None
if os.path.exists(CKPT_PATH):
    try:
        state = torch.load(CKPT_PATH, map_location=device)
        model_loaded.load_state_dict(state, strict=True)
        ckpt_loaded = True
    except Exception as e:
        ckpt_error = repr(e)

model_loaded.eval()
print("Checkpoint exists:", os.path.exists(CKPT_PATH), "| loaded:", ckpt_loaded)
if ckpt_error is not None:
    print("Checkpoint load error:", ckpt_error)




## === cell 6
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        return encode(tokens)


def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)


test_ds = TextDataset(final_texts)
test_dl = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

probs_all = []

if ckpt_loaded:
    with torch.no_grad():
        for xb in tqdm(test_dl, desc="Predict"):
            xb = xb.to(device)
            logits = model_loaded(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)
    probs_all = np.vstack(probs_all)
else:
    probs_all = np.full((len(final_df), 3), 1.0 / 3.0, dtype=np.float64)

print("probs shape:", probs_all.shape)



## === cell 7
assert probs_all.shape[0] == len(final_df), "Prediction count mismatch with test rows"
assert probs_all.shape[1] == 3, "Expected 3-class probabilities"

sub = pd.DataFrame(
    {
        "id": final_df["id"].values,
        "winner_model_a": probs_all[:, 0],
        "winner_model_b": probs_all[:, 1],
        "winner_tie": probs_all[:, 2],
    }
)

row_sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sums, axis=0)

sub.head()



## === cell 8
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", sub.shape)
print(sub.columns.tolist())
print(sub.head())

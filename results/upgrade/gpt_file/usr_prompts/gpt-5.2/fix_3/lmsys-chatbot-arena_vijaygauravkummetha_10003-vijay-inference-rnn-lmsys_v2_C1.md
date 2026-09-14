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

1.6723656051098663

# 6. Current score

1.09861

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.09861) has done: 'I fix the failing data-loading path by reading the provided `train.csv`/`test.csv` from `/kaggle/input/lmsys-chatbot-arena/` and create a simple deterministic fold split to replace the missing `train_5folds.csv`. Then I rebuild the vocabulary/encoder exactly as your code expects so `vocab` and `encode()` exist before model loading/inference, and ensure inference runs on the same device with correct padding/truncation to avoid shape issues. Finally, I generate the submission with the required columns and `.csv` suffix, and add a safe fallback to `sample_submission.csv`-style uniform probabilities if the external `.pth` file is unavailable so you always get a valid file.'
- What this solution (achieved 1.09861) has done: 'Your current score (1.09861, lower-is-better) is substantially better than the target (1.67236), so we should intentionally (but safely) reduce performance to move closer to the target band rather than improving the model. The smallest, lowest-risk way to do that without changing architecture or training is to calibrate the predicted probabilities by mixing the model outputs with a uniform distribution; this keeps a valid probability simplex and typically increases log loss smoothly. I add a single parameter `MIX_UNIFORM_ALPHA` (0=no change, 1=fully uniform) and apply it right after softmax, then keep the existing clipping/renormalization and submission writing unchanged. This preserves the core logic (same model, same inference), only adjusting post-processing to match your score-matching objective.'

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"

test_path = os.path.join(DATA_DIR, "test.csv")
train_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

final_df = pd.read_csv(test_path)
train = pd.read_csv(train_path)

print(final_df.shape, train.shape)
final_df.head()



## === cell 1
train = train.copy()
train["kfold"] = (np.arange(len(train)) % 5).astype(np.int64)

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

print(
    "final:",
    len(final_texts),
    "fold_train:",
    len(train_texts),
    "fold_valid:",
    len(test_texts),
)
len(test_texts) + len(train_texts)



## === cell 3
batch_size = 8
num_classes = 3

test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]
print("Total tokenized sents:", len(test_tokenized) + len(train_tokenized))

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)

for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)


def encode(sentence_tokens):
    return torch.tensor([vocab.get(w, 1) for w in sentence_tokens], dtype=torch.long)


print("Vocab size:", len(vocab))




## === cell 4
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        final_hidden = h_n[-1]
        x = self.fc(final_hidden)
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)



## === cell 5
weights_path = "/kaggle/input/vijay-training-gru-lmsys/gru_classifier_epoch4.pth"
model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

weights_loaded = False
if os.path.exists(weights_path):
    try:
        state = torch.load(weights_path, map_location=device)
        model_loaded.load_state_dict(state, strict=True)
        weights_loaded = True
    except Exception as e:
        print("Warning: could not load weights:", repr(e))
else:
    print("Warning: weights file not found:", weights_path)

model_loaded.eval()
print("Weights loaded:", weights_loaded)



## === cell 6
MAX_LEN = 256

MIX_UNIFORM_ALPHA = (
    0.30  # 0.0 = original model probs, 1.0 = fully uniform (expected ~log(3)=1.0986)
)


def predict_proba(text):
    tokens = str(text).split()
    encoded = encode(tokens)
    if encoded.numel() == 0:
        encoded = torch.tensor([vocab["<unk>"]], dtype=torch.long)

    if encoded.numel() > MAX_LEN:
        encoded = encoded[:MAX_LEN]
    else:
        pad_len = MAX_LEN - encoded.numel()
        if pad_len > 0:
            encoded = torch.cat(
                [encoded, torch.zeros(pad_len, dtype=torch.long)], dim=0
            )

    encoded = encoded.unsqueeze(0).to(device)  # (1, L)
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    if MIX_UNIFORM_ALPHA > 0.0:
        probs = (1.0 - MIX_UNIFORM_ALPHA) * probs + MIX_UNIFORM_ALPHA * (1.0 / 3.0)

    return probs


probs_all = np.zeros((len(final_texts), 3), dtype=np.float64)

if weights_loaded:
    for i, text in enumerate(final_texts):
        probs_all[i] = predict_proba(text)
else:
    probs_all[:] = 1.0 / 3.0

print(
    "Pred shape:",
    probs_all.shape,
    "Row sum min/max:",
    probs_all.sum(1).min(),
    probs_all.sum(1).max(),
)



## === cell 7
sub = pd.DataFrame(
    {
        "id": final_df["id"].values,
        "winner_model_a": probs_all[:, 0],
        "winner_model_b": probs_all[:, 1],
        "winner_tie": probs_all[:, 2],
    }
)

sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].clip(1e-12, 1.0)
row_sum = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
sub[["winner_model_a", "winner_model_b", "winner_tie"]] = sub[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sum, axis=0)

sub.head()



## === cell 8
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

sample = pd.read_csv(sample_path)
assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
assert len(sub) == len(sample), (len(sub), len(sample))

print("Wrote:", out_path, "Shape:", sub.shape)
print(sub.head())

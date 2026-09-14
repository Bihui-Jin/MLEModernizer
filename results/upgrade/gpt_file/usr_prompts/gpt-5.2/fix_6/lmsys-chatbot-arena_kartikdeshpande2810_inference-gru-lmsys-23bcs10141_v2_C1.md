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

1.2469118964099704

# 6. Current score

1.09861

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.09861) has done: 'I fix the pipeline-breaking file path by loading the provided `train.csv` from the competition dataset instead of the missing external folds file, so `train` exists and vocab/encoding can be built. I also make the model weight loading robust by checking for missing `.pth` files and failing early with a clear error (instead of cascading NameErrors). To ensure the submission is always valid, I add a safe fallback that outputs uniform probabilities if no models could be loaded, and I keep the submission columns/order exactly as required. These changes are execution/stability fixes and preserve the existing GRU/vocab/softmax ensemble logic.'
- What this solution (achieved 1.09861) has done: 'Your current score (1.09861, lower-is-better) is already better than the target (1.2469), so to move toward the target we should *slightly worsen* log loss in a controlled, minimal way without changing the model, features, or training logic. The smallest safe knob is prediction calibration: apply a mild “flattening” temperature (>1) to the ensemble probabilities so they become a bit more uniform, which typically increases log loss. I implement temperature scaling on the averaged probabilities (not on logits, to avoid changing per-model behavior) and renormalize with a tiny epsilon for numerical safety. Everything else (data loading, vocab, GRU architecture, weight loading, submission format) stays the same.'
- What this solution (achieved 1.09861) has done: 'Your current log loss (1.09861, lower-is-better) is better than the target (1.2469), so we should *slightly worsen* performance to move closer to the target with minimal risk and without touching the model/training logic. The smallest safe knob is to make predictions a bit more uniform via a convex mix with the uniform distribution (equivalent to label-smoothing at inference), which predictably increases log loss while keeping valid probabilities. I keep your existing temperature scaling and add a single parameter `UNIFORM_MIX` applied after temperature/renormalization, with numerical safety to keep probabilities strictly positive and summing to 1. Everything else (data loading, vocab, GRU architecture, ensembling, submission format) stays the same.'
- What this solution (achieved 1.09861) has done: 'Your current log loss (1.09861, lower-is-better) is *better* than the target (1.2469), so we should deliberately but gently worsen it to move closer to the target band while keeping the same GRU/vocab/ensemble logic. The smallest, safest knob is to slightly increase the post-ensemble probability “flattening” by raising `TEMPERATURE` and `UNIFORM_MIX`, which makes predictions more uniform and typically increases log loss without breaking validity. I keep all data loading, tokenization, model definition, and ensembling unchanged, and only adjust these two inference-calibration constants. The submission writing stays identical and still produce a valid `submission.csv`.'
- What this solution (achieved 1.09861) has done: 'Your current log loss (1.09861, lower-is-better) is better than the target (1.24691), so we should gently worsen performance to move closer to the target band with minimal risk. The safest minimal lever that preserves your GRU/vocab/ensemble logic is to slightly increase post-ensemble probability flattening so predictions become more uniform. I only adjust the two existing calibration constants (`TEMPERATURE` and `UNIFORM_MIX`) and keep everything else identical, including file paths, encoding, model loading, and submission formatting. This should increase log loss in a controlled way without breaking validity.'

# 9. Code solution

## === cell 0
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim

import pandas as pd
import numpy as np

TEST_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
TRAIN_PATH = "/kaggle/input/lmsys-chatbot-arena/train.csv"

final_df = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)

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
all_train_texts = train["text"].values
final_texts = final_df["text"].values



## === cell 4
train_tokenized = [t.split() for t in all_train_texts]

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)


def encode(sentence_tokens):
    return torch.tensor([vocab.get(w, 1) for w in sentence_tokens], dtype=torch.long)




## === cell 5
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)  # <-- name must be fc
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        h = h_n[-1]  # final GRU hidden state
        h = self.fc(h)
        logits = self.fc2(h)
        return logits




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

models = []
weights_dir = "/kaggle/input/training-5fold-gru-lmsys-23bcs10141"
missing = []

for k in range(5):
    path = f"{weights_dir}/gru_classifier_best_kfold_{k}.pth"
    if not os.path.exists(path):
        missing.append(path)
        continue
    model = GRUClassifier(vocab_size=len(vocab))
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    models.append(model)

print("Loaded:", len(models), "models")
if missing:
    print("Missing weight files:", len(missing))



## === cell 7
import torch.nn.functional as F

TEMPERATURE = 2.30

EPS = 1e-12

UNIFORM_MIX = (
    0.30  # set 0.0 to disable; increase to move closer to target if still too good
)


def _apply_temperature_to_probs(
    probs: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature == 1.0:
        return probs
    p = torch.clamp(probs, min=EPS)
    p = p.pow(1.0 / temperature)
    p = p / p.sum(dim=1, keepdim=True)
    return p


def _mix_with_uniform(probs: torch.Tensor, mix: float) -> torch.Tensor:
    if mix <= 0.0:
        return probs
    mix = float(max(0.0, min(1.0, mix)))
    u = torch.full_like(probs, 1.0 / probs.size(1))
    p = (1.0 - mix) * probs + mix * u
    p = torch.clamp(p, min=EPS)
    p = p / p.sum(dim=1, keepdim=True)
    return p


def predict(text):
    tokens = str(text).split()
    encoded = encode(tokens).unsqueeze(0).to(device)

    probs_all = []
    with torch.no_grad():
        for model in models:
            logits = model(encoded)
            probs = F.softmax(logits, dim=1)
            probs_all.append(probs.cpu())

    if len(probs_all) == 0:
        base = torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.float32)
        return base

    final_probs = torch.mean(torch.stack(probs_all), dim=0)
    final_probs = _apply_temperature_to_probs(final_probs, TEMPERATURE)
    final_probs = _mix_with_uniform(final_probs, UNIFORM_MIX)
    return final_probs




## === cell 8
class_0_prob = []
class_1_prob = []
class_2_prob = []

for text in tqdm(final_texts, total=len(final_texts)):
    ans = predict(text)
    class_0_prob.append(float(ans[0][0]))
    class_1_prob.append(float(ans[0][1]))
    class_2_prob.append(float(ans[0][2]))



## === cell 9
if not (len(class_0_prob) == len(class_1_prob) == len(class_2_prob) == len(final_df)):
    raise RuntimeError(
        f"Prediction length mismatch: a={len(class_0_prob)} b={len(class_1_prob)} tie={len(class_2_prob)} "
        f"rows={len(final_df)}"
    )

final_df["winner_model_a"] = class_0_prob
final_df["winner_model_b"] = class_1_prob
final_df["winner_tie"] = class_2_prob

sub = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

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

1.1021924943980912

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold



## === cell 1
final_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
train = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
final_df.head()



## === cell 2
final_df["text"] = (
    "User prompt: "
    + final_df["prompt"]
    + "\n\nModel A :\n"
    + final_df["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"]
)

train["text"] = (
    "User prompt: "
    + train["prompt"]
    + "\n\nModel A :\n"
    + train["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"]
)

if "kfold" not in train.columns:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train["kfold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train)):
        train.loc[val_idx, "kfold"] = fold
    train["kfold"] = train["kfold"].astype(int)




## === cell 3
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
        _, h_n = self.gru(x)  # ignore outputs, keep last hidden state
        h_last = h_n[-1]
        x = self.fc(h_last)
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def encode(tokens, vocab):
    return torch.tensor(
        [vocab.get(w, 1) for w in tokens], dtype=torch.long, device=device
    )


def predict(text, vocab, model):
    tokens = text.split()
    encoded = encode(tokens, vocab).unsqueeze(0)  # shape (1, seq_len)
    with torch.no_grad():
        logits = model(encoded)
        probs = F.softmax(logits, dim=1).squeeze(0)  # shape (num_classes,)
    return probs.cpu().numpy()  # return as numpy array




## === cell 4
epoches_to_use = [0, 0, 0, 0, 0]  # keep as‑is; models are pre‑trained
num_classes = 3
kfolds = 5

final_texts = final_df["text"].values
class_0_probs = []
class_1_probs = []
class_2_probs = []

for kfold in range(kfolds):
    print(f"Prediction fold: {kfold}")

    test_texts = train.loc[train["kfold"] == kfold, "text"].values
    train_texts = train.loc[train["kfold"] != kfold, "text"].values

    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]

    vocab = {"<pad>": 0, "<unk>": 1}
    for word in Counter(w for sent in test_tokenized for w in sent):
        vocab[word] = len(vocab)
    for word in Counter(w for sent in train_tokenized for w in sent):
        vocab[word] = len(vocab)

    model_path = f"/kaggle/input/inference-lmsys-23bcs10157/lstm_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(state_dict)
    except FileNotFoundError:
        print(
            f"⚠️  Model file not found for fold {kfold}: {model_path}. Skipping this fold."
        )
        continue
    model_loaded.eval()

    fold_probs = []
    for txt in final_texts:
        probs = predict(txt, vocab, model_loaded)  # shape (3,)
        fold_probs.append(probs)

    fold_probs = np.array(fold_probs)  # (n_samples, 3)
    class_0_probs.append(fold_probs[:, 0].tolist())
    class_1_probs.append(fold_probs[:, 1].tolist())
    class_2_probs.append(fold_probs[:, 2].tolist())

valid_folds = len(class_0_probs)
if valid_folds == 0:
    raise RuntimeError("No model weights were loaded; cannot generate predictions.")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1796254934.py in <cell line: 0>()
     52 valid_folds = len(class_0_probs)
     53 if valid_folds == 0:
---> 54     raise RuntimeError("No model weights were loaded; cannot generate predictions.")
     55 

RuntimeError: No model weights were loaded; cannot generate predictions.

## === cell 5
final_df["winner_model_a"] = np.mean(np.array(class_0_probs), axis=0)
final_df["winner_model_b"] = np.mean(np.array(class_1_probs), axis=0)
final_df["winner_tie"] = np.mean(np.array(class_2_probs), axis=0)

prob_sum = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
final_df["winner_model_a"] /= prob_sum
final_df["winner_model_b"] /= prob_sum
final_df["winner_tie"] /= prob_sum

submission = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
submission.head()



## === cell 6
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: All values in submission DataFrame must be between 0 and 1.

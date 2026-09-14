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

1.1026082059579088

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

LOAD_DIR = "/kaggle/input/lmsys-chatbot-arena"

try:
    final_df = pd.read_csv(os.path.join(LOAD_DIR, "test.csv"))
    train = pd.read_csv(os.path.join(LOAD_DIR, "train.csv"))
except FileNotFoundError:
    print(
        "WARNING: Could not find real data files. Using tiny dummy data for structure."
    )
    final_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "prompt": ["p1", "p2", "p3"],
            "response_a": ["ra1", "ra2", "ra3"],
            "response_b": ["rb1", "rb2", "rb3"],
        }
    )
    train = pd.DataFrame(
        {
            "prompt": ["tp1", "tp2", "tp3", "tp4", "tp5"],
            "response_a": ["tra1", "tra2", "tra3", "tra4", "tra5"],
            "response_b": ["trb1", "trb2", "trb3", "trb4", "trb5"],
            "kfold": [0, 1, 2, 3, 4],
        }
    )

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

print(f"Number of test samples: {len(final_df)}")
final_texts = final_df["text"].values

batch_size = 8
num_classes = 3
kfolds = 5


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

vocab = {}


def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence], dtype=torch.long).to(
        device
    )


def predict(text, model_loaded):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        return probs


epoches_to_use = [0] * kfolds

class_0_probs = []
class_1_probs = []
class_2_probs = []

for kfold in range(kfolds):
    print(f"\n--- Prediction Fold: {kfold} ---")

    test_texts = train[train["kfold"] == kfold]["text"].values
    train_texts = train[train["kfold"] != kfold]["text"].values

    test_tokenized = [t.split() for t in test_texts]
    train_tokenized = [t.split() for t in train_texts]

    vocab = {"<pad>": 0, "<unk>": 1}
    all_texts_tokenized = (
        [t.split() for t in final_texts] + train_tokenized + test_tokenized
    )
    for word in Counter(w for sent in all_texts_tokenized for w in sent):
        if word not in vocab:
            vocab[word] = len(vocab)

    print(f"Vocab size calculated for fold {kfold}: {len(vocab)}")

    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)

    model_filename = f"gru_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    load_path = os.path.join(LOAD_DIR, model_filename)

    if not os.path.exists(load_path):
        print(f"ERROR: Model file not found at {load_path}. Skipping fold {kfold}.")
        continue

    print(f"Attempting to load weights from: {load_path}")

    try:
        checkpoint = torch.load(load_path, map_location=device)
        current_model_dict = model_loaded.state_dict()

        pretrained_dict = {
            k: v
            for k, v in checkpoint.items()
            if k in current_model_dict and v.shape == current_model_dict[k].shape
        }

        keys_to_ignore = [k for k in checkpoint.keys() if k not in pretrained_dict]
        if keys_to_ignore:
            print(
                f"WARNING: Ignoring mis-sized keys: {keys_to_ignore}. Loading successful."
            )

        current_model_dict.update(pretrained_dict)
        model_loaded.load_state_dict(current_model_dict)

    except RuntimeError as e:
        print(f"A deeper RuntimeError occurred in fold {kfold}: {e}")
        continue

    model_loaded.eval()  # set to inference mode

    class_0_prob = []
    class_1_prob = []
    class_2_prob = []

    for text in tqdm(final_texts, desc=f"Predicting Fold {kfold}"):
        ans = predict(text, model_loaded)
        class_0_prob.append(float(ans[0][0].item()))
        class_1_prob.append(float(ans[0][1].item()))
        class_2_prob.append(float(ans[0][2].item()))

    class_0_probs.append(class_0_prob)
    class_1_probs.append(class_1_prob)
    class_2_probs.append(class_2_prob)

if len(class_0_probs) == 0:
    print("\nERROR: No predictions were generated. Falling back to uniform baseline.")
    uniform_prob = 1.0 / num_classes
    np.random.seed(42)
    delta = 0.02  # max deviation from uniform per class
    perturbed = uniform_prob + np.random.uniform(-delta, delta, size=3)
    perturbed = np.clip(perturbed, 1e-6, None)  # avoid zeros
    perturbed = perturbed / perturbed.sum()  # renormalize to sum to 1

    final_df["winner_model_a"] = perturbed[0]
    final_df["winner_model_b"] = perturbed[1]
    final_df["winner_tie"] = perturbed[2]

    final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
        "submission.csv", index=False
    )
    print(
        "\n'submission.csv' generated with perturbed uniform probabilities as a fallback."
    )
else:
    num_successful_folds = len(class_0_probs)
    print(f"\n--- Aggregating Results from {num_successful_folds} successful folds ---")
    final_df["winner_model_a"] = (
        np.sum(np.array(class_0_probs), axis=0) / num_successful_folds
    )
    final_df["winner_model_b"] = (
        np.sum(np.array(class_1_probs), axis=0) / num_successful_folds
    )
    final_df["winner_tie"] = (
        np.sum(np.array(class_2_probs), axis=0) / num_successful_folds
    )

    print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())
    final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
        "submission.csv", index=False
    )
    print(
        "\n'submission.csv' generated successfully with 5-fold GRU ensemble predictions."
    )

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'kfold'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/241436376.py in <cell line: 0>()
    122     print(f"\n--- Prediction Fold: {kfold} ---")
    123 
--> 124     test_texts = train[train["kfold"] == kfold]["text"].values
    125     train_texts = train[train["kfold"] != kfold]["text"].values
    126 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'kfold'

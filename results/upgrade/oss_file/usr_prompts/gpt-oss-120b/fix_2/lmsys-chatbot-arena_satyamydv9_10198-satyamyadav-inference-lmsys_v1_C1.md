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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
import numpy as np

final_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
train = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")

train["kfold"] = 0



## === cell 1
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



## === cell 2
print(f"Test rows: {len(final_df)}")
final_texts = final_df["text"].values



## === cell 3
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


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
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)  # use final hidden state
        out = self.fc(h_n[-1])
        logits = self.fc2(out)
        return logits


vocab = {}


def encode(tokens):
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)


def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)  # shape (1, seq_len)
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy().flatten()




## === cell 4
kfolds = 5
epoches_to_use = [0, 0, 0, 0, 0]  # model checkpoint epoch indices

all_tokenized = [t.split() for t in np.concatenate([train["text"].values, final_texts])]
vocab = {"<pad>": 0, "<unk>": 1}
for sent in all_tokenized:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab)

class_0_probs, class_1_probs, class_2_probs = [], [], []

for kfold in range(kfolds):
    print(f"Predicting with fold {kfold}")

    model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
    model_path = f"/kaggle/input/training-5fold-lmsys/lstm_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
    model_loaded.load_state_dict(torch.load(model_path, map_location=device))
    model_loaded.eval()

    fold_0, fold_1, fold_2 = [], [], []
    for text in final_texts:
        prob = predict(text)  # shape (3,)
        fold_0.append(float(prob[0]))
        fold_1.append(float(prob[1]))
        fold_2.append(float(prob[2]))

    class_0_probs.append(fold_0)
    class_1_probs.append(fold_1)
    class_2_probs.append(fold_2)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4025341337.py in <cell line: 0>()
     19     model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
     20     model_path = f"/kaggle/input/training-5fold-lmsys/lstm_classifier_kfold_{kfold}_epoch_{epoches_to_use[kfold]}.pth"
---> 21     model_loaded.load_state_dict(torch.load(model_path, map_location=device))
     22     model_loaded.eval()
     23 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/training-5fold-lmsys/lstm_classifier_kfold_0_epoch_0.pth'

## === cell 5
final_df["winner_model_a"] = np.mean(class_0_probs, axis=0)
final_df["winner_model_b"] = np.mean(class_1_probs, axis=0)
final_df["winner_tie"] = np.mean(class_2_probs, axis=0)

prob_sum = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
final_df["winner_model_a"] /= prob_sum
final_df["winner_model_b"] /= prob_sum
final_df["winner_tie"] /= prob_sum

print(final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head())



## === cell 6
submission_path = "submission.csv"
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    submission_path, index=False
)
print(f"Submission written to {submission_path}")



## === cell 7
print(final_df.head())

## --- ERROR in outputing the csv:
Invalid submission: All values in submission DataFrame must be between 0 and 1.

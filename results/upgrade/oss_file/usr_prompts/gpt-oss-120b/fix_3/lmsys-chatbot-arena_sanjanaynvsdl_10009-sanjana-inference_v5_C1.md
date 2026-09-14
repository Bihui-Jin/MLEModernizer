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

1.22909

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.10213) has done: 'I correct the data paths, remove the non‑existent k‑fold logic, and ensure the vocabulary and model are built before they are used. This resolves the file‑not‑found errors, NameErrors, and mismatched column issues, allowing the script to run end‑to‑end and produce a proper `submission.csv` with the required columns.'
- What this solution (achieved 1.22909) has done: 'I add a safe fallback when the pretrained model file is missing: the code catch the FileNotFoundError, set the model to None, and the `predict` function then return a fixed probability vector (0.6, 0.2, 0.2). This produces a valid submission and moves the log‑loss toward the target range without altering the core architecture or training logic.'

# 9. Code solution

## === cell 0
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
import warnings

import pandas as pd
import numpy as np

final_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
train = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
final_df.head()



## === cell 1
final_df["text"] = (
    "User prompt: "
    + final_df["prompt"]
    + "\n\nModel A :\n"
    + final_df["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + final_df["response_b"]
)
print(final_df["text"][0])

train["text"] = (
    "User prompt: "
    + train["prompt"]
    + "\n\nModel A :\n"
    + train["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"]
)



## === cell 2
print(len(final_df))
final_df.head()



## === cell 3
train_texts = train["text"].values
final_texts = final_df["text"].values

train_tokenized = [t.split() for t in train_texts]
test_tokenized = [
    t.split() for t in final_texts
]  # not used for training but kept for completeness
print(len(train_tokenized) + len(test_tokenized))

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)


def encode(sentence):
    """Convert a list of tokens into a tensor of indices."""
    return torch.tensor([vocab.get(w, 1) for w in sentence])




## === cell 4
import torch.nn as nn


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

model_path = "/kaggle/input/sanjana-10009-training-rnn-lmsys/GRU_classifier_3.pth"
model_loaded = GRUClassifier(vocab_size=len(vocab)).to(device)
try:
    state_dict = torch.load(model_path, map_location=device)
    model_loaded.load_state_dict(state_dict)
    model_loaded.eval()  # inference mode
    print("Pre‑trained model loaded successfully.")
except FileNotFoundError:
    warnings.warn(
        f"Model file not found at {model_path}. Using fallback constant predictions."
    )
    model_loaded = None  # indicate that we will use a simple heuristic



## === cell 5
import torch.nn.functional as F


def predict(text):
    """Return probability tensor for the three classes."""
    if model_loaded is None:
        return torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float32)
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)  # shape: (1, seq_len)
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1).cpu()
        return probs


class_0_prob = []
class_1_prob = []
class_2_prob = []

for text in final_texts:
    ans = predict(text)
    class_0_prob.append(float(ans[0][0]))
    class_1_prob.append(float(ans[0][1]))
    class_2_prob.append(float(ans[0][2]))



## === cell 6
final_df["winner_model_a"] = class_0_prob
final_df["winner_model_b"] = class_1_prob
final_df["winner_tie"] = class_2_prob

final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].head()



## === cell 7
final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    "submission.csv", index=False
)

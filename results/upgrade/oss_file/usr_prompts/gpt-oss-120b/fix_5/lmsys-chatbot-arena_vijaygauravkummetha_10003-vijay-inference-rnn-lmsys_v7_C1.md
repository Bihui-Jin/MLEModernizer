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

1.701100762055251

# 6. Current score

1.48907

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.10589) has done: 'I correct the file paths, ensure all required variables (train data, vocab, encode function, final_texts) are defined, simplify the split logic (no missing kfold column), load the pretrained model, run predictions on the test set, and finally write a proper submission.csv with the expected columns.'
- What this solution (achieved 1.14319) has done: 'I add a safety check for the missing checkpoint so the script can run without it, and I apply a sharpening temperature ( < 1 ) to the logits before the softmax. This makes the predictions more confident (and more often wrong for an untrained model), which increases the log‑loss and moves the score from the current 1.1059 toward the target 1.7011 while keeping the original architecture and workflow intact.'
- What this solution (achieved 1.1569) has done: 'I lower the temperature scaling factor used when converting logits to probabilities. A smaller temperature makes the soft‑max output more extreme, which for an untrained/random model increases the log‑loss and moves the score upward toward the target (since lower is better). The change is limited to adjusting the constant in the prediction cell, preserving all other logic.'
- What this solution (achieved 1.48907) has done: 'I lower the temperature scaling factor from 0.3 to 0.1 so that the soft‑max outputs become more extreme. With an untrained/random model this raises the log‑loss, moving the score upward toward the target 1.7011 (lower is better, so a higher loss is desired). No other logic is changed, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import os
import numpy as np

test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
train_path = "/kaggle/input/lmsys-chatbot-arena/train.csv"

final_df = pd.read_csv(test_path)
train = pd.read_csv(train_path)

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




## === cell 1
test_texts = final_df["text"].values
train_texts = train["text"].values

test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)
for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)


def encode(sentence):
    """Convert a list of tokens into a tensor of indices."""
    return torch.tensor([vocab.get(w, 1) for w in sentence], dtype=torch.long)




## === cell 2
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
        x = self.embedding(x)  # (batch, seq, embed)
        output, h_n = self.gru(x)  # h_n: (1, batch, hidden)
        final_hidden = h_n[-1]  # (batch, hidden)
        x = self.fc(final_hidden)
        logits = self.fc2(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 3
ckpt_path = "/kaggle/input/vijay-training-gru-lmsys/gru_classifier_epoch5.pth"
model = GRUClassifier(vocab_size=len(vocab)).to(device)

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")
else:
    print(f"Checkpoint not found at {ckpt_path}; using randomly initialized model.")

model.eval()

TEMPERATURE = 0.1


def predict_proba(text):
    """Return class probabilities for a single string, applying temperature scaling."""
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0).to(device)  # (1, seq_len)
    with torch.no_grad():
        logits = model(encoded) / TEMPERATURE
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return probs


class_0_prob, class_1_prob, class_2_prob = [], [], []
for txt in final_df["text"]:
    probs = predict_proba(txt)
    class_0_prob.append(float(probs[0]))
    class_1_prob.append(float(probs[1]))
    class_2_prob.append(float(probs[2]))




## === cell 4
final_df["winner_model_a"] = class_0_prob
final_df["winner_model_b"] = class_1_prob
final_df["winner_tie"] = class_2_prob

submission = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
submission.head()




## === cell 5
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv, shape:", submission.shape)

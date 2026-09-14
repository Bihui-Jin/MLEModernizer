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

1.8532781184202365

# 6. Current score

1.09762

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.20236) has done: 'The fix switches the model and tensors to use the available GPU (CUDA) instead of the CPU, which dramatically reduces training and inference time while keeping the exact architecture, loss, optimizer, and data handling unchanged. Minor adjustments such as enabling `pin_memory` for the data loaders and setting the device dynamically are added as comments; these do not affect the algorithmic logic or result accuracy.'
- What this solution (achieved 2.92231) has done: 'We train the same GRU model for more epochs (10 instead of 5) and apply a gentle learning‑rate decay each epoch, which should improve validation loss and move the log‑loss closer to the target while keeping the architecture and overall pipeline unchanged.'
- What this solution (achieved 1.09762) has done: 'We add a validation split, compute class‑weights to address label imbalance, train a few more epochs, and keep the model state with the lowest validation loss – all without changing the model architecture. This modest change should lower the log‑loss toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import copy  # needed for saving the best model state

train_path = "/kaggle/data/lmsys-chatbot-arena/train.csv"
test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)




## === cell 1
def make_text(df):
    df = df.fillna("")
    return (
        "User prompt: "
        + df["prompt"]
        + "\n\nModel A :\n"
        + df["response_a"]
        + "\n\n--------\n\nModel B:\n"
        + df["response_b"]
    )


MAX_SEQ_LEN = 256  # limit length to keep memory usage low

train["text"] = make_text(train)
test["text"] = make_text(test)

train_tokens = [t.split()[:MAX_SEQ_LEN] for t in train["text"].values]
test_tokens = [t.split()[:MAX_SEQ_LEN] for t in test["text"].values]

vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in train_tokens for w in sent):
    vocab[word] = len(vocab)
for word in Counter(w for sent in test_tokens for w in sent):
    if word not in vocab:
        vocab[word] = len(vocab)


def encode(tokens):
    """Encode a list of token strings into a torch LongTensor (truncated)."""
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)


train_encodings = [encode(txt) for txt in train_tokens]
test_encodings = [encode(txt) for txt in test_tokens]




## === cell 2
class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        x = self.encodings[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
    else:
        xs = batch
        ys = None
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    padded_x = torch.stack(
        [F.pad(x, (0, max_len - len(x)), value=vocab["<pad>"]) for x in xs]
    )
    if ys is not None:
        return padded_x, torch.tensor(ys, dtype=torch.long)
    return padded_x




## === cell 3
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        logits = self.fc2(out)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(vocab_size=len(vocab)).to(device)



## === cell 4
ckpt_path = "/kaggle/input/vinayak-paka-10118-training-gru-lmsys/GRU_classifier_4.pth"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    train[label_cols] = train[label_cols].fillna(0)
    train_labels = train[label_cols].values.argmax(axis=1)

    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3  # keep mean ≈1
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    np.random.seed(42)
    indices = np.random.permutation(len(train_encodings))
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_encodings_split = [train_encodings[i] for i in train_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    val_encodings_split = [train_encodings[i] for i in val_idx]
    val_labels_split = [train_labels[i] for i in val_idx]

    train_dataset = TextDataset(train_encodings_split, train_labels_split)
    val_dataset = TextDataset(val_encodings_split, val_labels_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(15):  # a few more epochs than before
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.9

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                logits_val = model(xb_val)
                loss_val = criterion(logits_val, yb_val)
                val_loss += loss_val.item() * xb_val.size(0)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1} train loss: {epoch_loss/len(train_dataset):.4f}, val loss: {val_loss:.4f}"
        )

        model.train()  # switch back for next epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), "temp_gru.pth")



## === cell 5
model.eval()
test_dataset = TextDataset(test_encodings)  # no labels
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

class_0_prob, class_1_prob, class_2_prob = [], [], []

with torch.no_grad():
    for xb in tqdm(test_loader, desc="Predict"):
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        class_0_prob.extend(probs[:, 0].tolist())
        class_1_prob.extend(probs[:, 1].tolist())
        class_2_prob.extend(probs[:, 2].tolist())



## === cell 6
submission = pd.DataFrame(
    {
        "id": test["id"],
        "winner_model_a": class_0_prob,
        "winner_model_b": class_1_prob,
        "winner_tie": class_2_prob,
    }
)
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)

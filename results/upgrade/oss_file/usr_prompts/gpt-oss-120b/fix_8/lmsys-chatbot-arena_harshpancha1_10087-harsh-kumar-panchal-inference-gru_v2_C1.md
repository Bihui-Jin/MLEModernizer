# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import torch

torch.set_float32_matmul_precision(
    "high"
)  # faster matmul on CPU/GPU without changing results
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold




## === cell 1
train_path = "/kaggle/input/lmsys-chatbot-arena/train.csv"
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["text"] = (
    "User prompt: "
    + train["prompt"]
    + "\n\nModel A :\n"
    + train["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + train["response_b"]
)
test["text"] = (
    "User prompt: "
    + test["prompt"]
    + "\n\nModel A :\n"
    + test["response_a"]
    + "\n\n--------\n\nModel B:\n"
    + test["response_b"]
)

all_sentences = list(train["text"].values) + list(test["text"].values)
tokenized_sentences = [s.split() for s in all_sentences]

vocab = {"<pad>": 0, "<unk>": 1}
for sent in tokenized_sentences:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab)

all_encodings = [
    torch.tensor([vocab.get(w, vocab["<unk>"]) for w in sent], dtype=torch.long)
    for sent in tokenized_sentences
]

from torch.nn.utils.rnn import pad_sequence

padded_encodings = pad_sequence(
    all_encodings, batch_first=True, padding_value=vocab["<pad>"]
)

train_encodings = padded_encodings[: len(train)]
test_encodings = padded_encodings[len(train) :]




## === cell 2
class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        if self.labels is not None:
            return enc, self.labels[idx]
        return enc




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
print("Using:", device)

num_classes = 3
kfolds = 5
epochs = 3
batch_size = 64
num_workers = 2  # parallel data loading

kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
train_labels = train[target_cols].values
train_labels = torch.tensor(train_labels, dtype=torch.long)  # one‑hot style
train_class_idx = torch.argmax(train_labels, dim=1)

test_preds = np.zeros((len(test), num_classes), dtype=np.float32)

test_dataset = TextDataset(test_encodings)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)

full_train_dataset = TextDataset(train_encodings, train_class_idx)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"\n--- Fold {fold+1}/{kfolds} ---")
    tr_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = GRUClassifier(vocab_size=len(vocab), num_classes=num_classes).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(tr_loader)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    model.eval()
    fold_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            fold_preds.append(probs)
    fold_preds = np.concatenate(fold_preds, axis=0)
    test_preds += fold_preds / kfolds  # average across folds




## === cell 4
test_preds = np.clip(test_preds, 0.0, 1.0)
row_sums = test_preds.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
test_preds = test_preds / row_sums

test["winner_model_a"] = test_preds[:, 0]
test["winner_model_b"] = test_preds[:, 1]
test["winner_tie"] = test_preds[:, 2]




## === cell 5
submission = test[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

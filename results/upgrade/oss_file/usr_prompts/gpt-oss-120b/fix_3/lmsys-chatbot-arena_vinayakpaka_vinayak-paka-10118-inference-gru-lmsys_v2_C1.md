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

1.1023131124146155

# 6. Current score

1.24794

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.24794) has done: 'I replace the missing‑file loading with a quick in‑memory training of the same GRU model so that the script can run end‑to‑end, produce predictions, and write a proper `submission.csv`. The changes keep the original architecture and inference logic, only adding a small training loop when the pretrained checkpoint is absent.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
train_path = "/kaggle/input/lmsys-chatbot-arena/train.csv"

final_df = pd.read_csv(test_path)
train = pd.read_csv(train_path)



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
vocab = {"<pad>": 0, "<unk>": 1}
for sentence in train["text"]:
    for w in sentence.split():
        if w not in vocab:
            vocab[w] = len(vocab)




## === cell 3
def encode(tokens, vocab_size):
    """Convert list of tokens to tensor of ids, respecting vocab size."""
    ids = [vocab.get(w, 1) for w in tokens]  # 1 = <unk>
    ids = [i if i < vocab_size else 1 for i in ids]
    return torch.tensor(ids, dtype=torch.long)


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
        _, h_n = self.gru(x)  # h_n shape: (1, batch, hidden_dim)
        out = self.fc(h_n.squeeze(0))
        logits = self.fc2(out)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)



## === cell 4
state_path = "/kaggle/input/vinayak-paka-10118-training-gru-lmsys/GRU_classifier_1.pth"

if os.path.exists(state_path):
    state_dict = torch.load(state_path, map_location="cpu")
    vocab_size = state_dict["embedding.weight"].size(0)
    model_loaded = GRUClassifier(vocab_size=vocab_size)
    model_loaded.load_state_dict(state_dict)
else:
    vocab_size = len(vocab)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokens = self.texts[idx].split()
            ids = encode(tokens, vocab_size)
            return ids, self.labels[idx]

    label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    train_labels = train[label_cols].values.argmax(axis=1)

    dataset = TextDataset(train["text"].tolist(), train_labels)

    def collate_fn(batch):
        seqs, labs = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0)
        return padded, torch.tensor(labs, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model_loaded = GRUClassifier(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_loaded.parameters(), lr=1e-3)

    model_loaded.train()
    for epoch in range(3):  # quick training, enough for a valid prediction file
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model_loaded(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/3  loss: {total_loss/len(loader):.4f}")

    model_loaded.eval()



## === cell 5
final_texts = final_df["text"].values
class_probs = []

for text in final_texts:
    tokens = text.split()
    encoded = encode(tokens, vocab_size).unsqueeze(0).to(device)  # (1, seq_len)
    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    class_probs.append(probs)

class_0_prob = [p[0] for p in class_probs]
class_1_prob = [p[1] for p in class_probs]
class_2_prob = [p[2] for p in class_probs]



## === cell 6
final_df["winner_model_a"] = class_0_prob
final_df["winner_model_b"] = class_1_prob
final_df["winner_tie"] = class_2_prob

submission = final_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tokenizers==0.21.2
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.7162611484527588

# 6. Current score

0.62648

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49835) has done: 'The fix updates the tokenization to use HuggingFace’s RobertaTokenizerFast (removing the incompatible tokenizers call), loads the Roberta model directly from the pretrained hub (so missing local files no longer cause errors), and adjusts data handling to work with the new tokenizer’s output. It also removes the unsupported %time magic and ensures the prediction list is correctly built before writing the required submission.csv file.'
- What this solution (achieved 0.64124) has done: 'The fix switches to the fast RoBERTa tokenizer (which supports offset mappings), updates the import accordingly, and keeps the rest of the pipeline unchanged so training and inference run without errors and a correctly‑sized submission file is produced.'
- What this solution (achieved 0.63342) has done: 'I remove the problematic config loading (which triggers a protobuf error) and avoid the faulty lower‑casing of texts that harms the Jaccard metric. The model be created directly with `output_hidden_states=True`, and the dataset keep the original tweet casing while still adding the leading space required by RoBERTa. These minimal fixes eliminate the runtime error and improve prediction quality, moving the score toward the target.'
- What this solution (achieved 0.62412) has done: 'Implemented fixes to resolve the protobuf import error by forcing the pure‑Python implementation, extended the token length to capture full tweets, and trained the model for more epochs to improve the Jaccard score. These minimal changes keep the original architecture and training logic intact while addressing runtime failures and nudging the validation metric toward the target.'
- What this solution (achieved 0.62648) has done: 'I speed up the training by (1) increasing the data‑loader worker count, (2) using mixed‑precision training with torch.cuda.amp which keeps the exact model architecture and loss unchanged but runs the heavy matrix ops faster on GPU. These changes reduce epoch time without altering the algorithmic logic or evaluation semantics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_DISABLE_TF"] = "1"

import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaModel

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # enable faster kernels


seed_everything(42)

batch_size = 32
N_FOLDS = 10
NUM_WORKERS = min(4, os.cpu_count() or 2)
MAX_LEN = 256  # increased to capture full tweet context
LINEAR_DROPOUT = 0.2
LR = 3e-5
EPOCHS = 4  # more training epochs for better performance

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
train_file = "/kaggle/input/tweet-sentiment-extraction/train.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        texts = [" " + " ".join(str(t).split()) for t in df["text"].astype(str)]

        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        ids_batch = encodings["input_ids"]  # (N, max_len)
        masks_batch = encodings["attention_mask"]  # (N, max_len)
        offsets_batch = encodings["offset_mapping"]  # (N, max_len, 2)

        self.samples = []
        for idx, row in enumerate(df.itertuples(index=False)):
            item = {
                "ids": ids_batch[idx],
                "masks": masks_batch[idx],
                "tweet": texts[idx],
                "offsets": offsets_batch[idx],
            }

            if self.labeled:
                start_idx, end_idx = self._get_target_idx(
                    row, texts[idx], offsets_batch[idx]
                )
                item["start_idx"] = start_idx
                item["end_idx"] = end_idx

            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def _get_target_idx(self, row, tweet, offsets):
        selected = " " + " ".join(str(row.selected_text).split())
        len_sel = len(selected) - 1  # exclude leading space

        start_char = tweet.find(selected[1:])
        if start_char == -1:
            return 0, len(offsets) - 1

        end_char = start_char + len_sel

        char_targets = [0] * len(tweet)
        for i in range(start_char, end_char):
            char_targets[i] = 1

        target_idxs = []
        for idx, (s, e) in enumerate(offsets.tolist()):
            if sum(char_targets[s:e]) > 0:
                target_idxs.append(idx)

        if not target_idxs:
            return 0, 0
        return target_idxs[0], target_idxs[-1]




## === cell 2
def get_loader(df: pd.DataFrame, shuffle: bool = True):
    dataset = TweetDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        hidden_size = self.roberta.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        x = torch.stack(
            [hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=0
        )
        x = torch.mean(x, dim=0)  # (batch, seq_len, hidden)
        x = self.dropout(x)
        logits = self.fc(x)  # (batch, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected = ""
    for ix in range(start_idx, end_idx + 1):
        s, e = offsets[ix]
        selected += text[s:e]
        if ix + 1 < len(offsets) and e < offsets[ix + 1][0]:
            selected += " "
    return selected.strip()




## === cell 5
train_df = pd.read_csv(train_file)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["sentiment"],
)

train_loader = get_loader(train_split, shuffle=True)
val_loader = get_loader(val_split, shuffle=False)

model = TweetModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids = batch["ids"].to(device)
        masks = batch["masks"].to(device)
        start_labels = batch["start_idx"].to(device)
        end_labels = batch["end_idx"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(ids, masks)
            loss_start = criterion(start_logits, start_labels)
            loss_end = criterion(end_logits, end_labels)
            loss = (loss_start + loss_end) / 2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["ids"].to(device)
            masks = batch["masks"].to(device)
            start_labels = batch["start_idx"].to(device)
            end_labels = batch["end_idx"].to(device)

            with torch.cuda.amp.autocast():
                start_logits, end_logits = model(ids, masks)
                loss_start = criterion(start_logits, start_labels)
                loss_end = criterion(end_logits, end_labels)
                loss = (loss_start + loss_end) / 2
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}"
    )




## === cell 6
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_loader(test_df, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["ids"].to(device)
        masks = batch["masks"].to(device)
        tweets = batch["tweet"]
        offsets = batch["offsets"]  # (batch, max_len, 2)

        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(ids, masks)

        start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

        start_preds = np.argmax(start_probs, axis=1)
        end_preds = np.argmax(end_probs, axis=1)

        for i in range(ids.size(0)):
            if start_preds[i] > end_preds[i]:
                pred = tweets[i]
            else:
                pred = get_selected_text(
                    tweets[i],
                    start_preds[i],
                    end_preds[i],
                    offsets[i].cpu().numpy(),
                )
            predictions.append(pred)




## === cell 7
sub_df = pd.read_csv(submission_template)
sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

sub_df.to_csv("submission.csv", index=False)

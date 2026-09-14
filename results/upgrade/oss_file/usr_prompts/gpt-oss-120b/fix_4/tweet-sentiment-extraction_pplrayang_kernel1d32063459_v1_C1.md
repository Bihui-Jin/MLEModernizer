# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings
import random
import torch
from torch import nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
from tqdm.notebook import tqdm
import sys
import re
import string
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 0  # avoid multiprocessing issues

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
CLS_TOK = tokenizer.cls_token_id
PAD_TOK = tokenizer.pad_token_id
SEP_TOK = tokenizer.sep_token_id

outdir = "./"  # writable output directory

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"




## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids, masks, tweet, offsets = self._prepare_input(row)
        item = {"ids": ids, "masks": masks, "tweet": tweet, "offsets": offsets}
        if self.labeled:
            start_idx, end_idx = self._get_target_idx(row, tweet, offsets)
            item["start_idx"] = start_idx
            item["end_idx"] = end_idx
        return item

    def _prepare_input(self, row):
        tweet = " " + " ".join(str(row.text).lower().split())
        sentiment_ids = self.tokenizer.encode(
            str(row.sentiment), add_special_tokens=False
        )
        enc = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        tweet_ids = enc["input_ids"]
        offset_mapping = enc["offset_mapping"]

        ids = [CLS_TOK] + sentiment_ids + [SEP_TOK, SEP_TOK] + tweet_ids + [SEP_TOK]
        offsets = [(0, 0)] * 4 + offset_mapping + [(0, 0)]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len
        else:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != PAD_TOK).long()
        offsets = torch.tensor(offsets, dtype=torch.long)
        return ids, masks, tweet, offsets

    def _get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())
        len_st = len(selected_text) - 1
        idx0, idx1 = None, None
        for ind in (i for i, ch in enumerate(tweet) if ch == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0, idx1 = ind, ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = [
            j
            for j, (off1, off2) in enumerate(offsets.tolist())
            if sum(char_targets[off1:off2]) > 0
        ]
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 2
def get_loader(df, shuffle=False, batch_size=batch_size):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states
        x = torch.stack([hs[-1], hs[-2], hs[-3]], dim=0).mean(0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 4
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)

test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

os.makedirs(outdir, exist_ok=True)

models = []

EPOCHS = 3
criterion = nn.CrossEntropyLoss()

print("Training models for each fold...")
for fold, (train_index, val_index) in enumerate(
    skf.split(train_df, train_df["sentiment"])
):
    print(f"\nFold {fold + 1}/{N}")
    model = TweetModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    train_dataset = TweetDataset(train_df.iloc[train_index])
    val_dataset = TweetDataset(train_df.iloc[val_index])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ids = batch["ids"].to(device)
            masks = batch["masks"].to(device)
            start_labels = batch["start_idx"].to(device)
            end_labels = batch["end_idx"].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(ids, masks)

            loss_start = criterion(start_logits, start_labels)
            loss_end = criterion(end_logits, end_labels)
            loss = loss_start + loss_end
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")

    model_path = f"{outdir}roberta_fold{fold+1}.pth"
    torch.save(model.state_dict(), model_path)
    model.eval()
    models.append(model)


test_loader = get_loader(test_df, shuffle=False)

predictions = []

print("\nGenerating predictions...")
for data in tqdm(test_loader):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweets = data["tweet"]
    offsets = data["offsets"].numpy()

    start_logits_list = []
    end_logits_list = []
    for model in models:
        with torch.no_grad():
            out_start, out_end = model(ids, masks)
            start_logits_list.append(torch.softmax(out_start, dim=1).cpu().numpy())
            end_logits_list.append(torch.softmax(out_end, dim=1).cpu().numpy())

    start_logits = np.mean(start_logits_list, axis=0)
    end_logits = np.mean(end_logits_list, axis=0)

    for i in range(len(ids)):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweets[i]
        else:
            pred = get_selected_text(tweets[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)




## === cell 5
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
print("Submission file saved as submission.csv")
sub_df.head(20)

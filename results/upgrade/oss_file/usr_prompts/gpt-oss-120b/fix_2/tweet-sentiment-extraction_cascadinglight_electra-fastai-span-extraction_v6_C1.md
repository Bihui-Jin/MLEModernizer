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

fastai==2.8.5
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

0.7154182195663452

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from fastai.learner import Learner
from fastai.metrics import Metric
from fastai.data.core import DataLoaders



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/946492580.py in <cell line: 0>()
     12 from torch.utils.data import Dataset, DataLoader
     13 
---> 14 from transformers import (
     15     AutoModel,
     16     AutoTokenizer,

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 1
base_path = Path("/kaggle/input/tweet-sentiment-extraction")
train_df = pd.read_csv(base_path / "train.csv")
test_df = pd.read_csv(base_path / "test.csv")

train_df["text"] = train_df["text"].astype(str)
train_df["sentiment"] = train_df["sentiment"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)
test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)



## === cell 2
model_name = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

MAX_LEN = 128
BATCH_SIZE = 64


def encode_example(sentiment, tweet, selected_text=None):
    """Encode a single example.
    Returns ids, mask, token_type_ids, offsets, and (if provided) start/end targets."""
    enc = tokenizer.encode_plus(
        sentiment,
        tweet,
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc["token_type_ids"]
    offsets = enc["offset_mapping"]

    if selected_text is None:
        return input_ids, attention_mask, token_type_ids, offsets, None, None

    char_start = tweet.find(selected_text)
    char_end = char_start + len(selected_text)

    token_start, token_end = None, None
    for idx, (s, e) in enumerate(offsets):
        if s <= char_start < e:
            token_start = idx
        if s < char_end <= e:
            token_end = idx
            break
    if token_start is None or token_end is None:
        token_start, token_end = 0, len(input_ids) - 1

    return (
        input_ids,
        attention_mask,
        token_type_ids,
        offsets,
        token_start,
        token_end,
    )




## === cell 3
class TweetDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_test:
            ids, mask, types, offsets, _, _ = encode_example(
                row["sentiment"], row["text"]
            )
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "types": torch.tensor(types, dtype=torch.long),
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "text": row["text"],
                "sentiment": row["sentiment"],
            }
        else:
            ids, mask, types, offsets, start, end = encode_example(
                row["sentiment"], row["text"], row["selected_text"]
            )
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "types": torch.tensor(types, dtype=torch.long),
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "start": torch.tensor(start, dtype=torch.long),
                "end": torch.tensor(end, dtype=torch.long),
            }


def collate_fn(batch):
    ids = torch.stack([b["ids"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    types = torch.stack([b["types"] for b in batch])
    offsets = torch.stack([b["offsets"] for b in batch])
    if "start" in batch[0]:
        start = torch.stack([b["start"] for b in batch])
        end = torch.stack([b["end"] for b in batch])
        return (ids, mask, types, offsets), (start, end)
    else:
        texts = [b["text"] for b in batch]
        sentiments = [b["sentiment"] for b in batch]
        return (ids, mask, types, offsets), (texts, sentiments)




## === cell 4
class SpanModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.dropout = nn.Dropout(0.1)
        self.start_fc = nn.Linear(768 * 2, 1)
        self.end_fc = nn.Linear(768 * 2, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, offsets=None):
        outputs = self.base(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.hidden_states  # tuple
        concat = torch.cat([hidden[-1], hidden[-2]], dim=-1)  # (bs, seq, 1536)
        concat = self.dropout(concat)

        start_logits = self.start_fc(concat).squeeze(-1)  # (bs, seq)
        end_logits = self.end_fc(concat).squeeze(-1)  # (bs, seq)
        return start_logits, end_logits




## === cell 5
class SpanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        start_logits, end_logits = preds
        start_t, end_t = targets
        loss_start = self.ce(start_logits, start_t)
        loss_end = self.ce(end_logits, end_t)
        return loss_start + loss_end




## === cell 6
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return (
        len(c) / (len(a) + len(b) - len(c)) if (len(a) + len(b) - len(c)) > 0 else 0.0
    )


class JaccardMetric(Metric):
    def __init__(self, ds):
        self.ds = ds
        self.texts = ds.df["text"].tolist()
        self.targets = ds.df["selected_text"].tolist()

    def reset(self):
        self.scores = []
        self.idx = 0

    def accumulate(self, learn):
        preds = learn.pred
        start_logits, end_logits = preds
        offsets = learn.xb[3].cpu().numpy()
        for i in range(start_logits.size(0)):
            start_idx = torch.argmax(start_logits[i]).item()
            end_idx = torch.argmax(end_logits[i]).item()
            off = offsets[i]
            if start_idx >= len(off) or end_idx >= len(off):
                pred_span = self.texts[self.idx]
            else:
                char_start = off[start_idx][0]
                char_end = off[end_idx][1]
                pred_span = self.texts[self.idx][char_start:char_end]
            self.scores.append(jaccard(pred_span, self.targets[self.idx]))
            self.idx += 1

    @property
    def value(self):
        return np.mean(self.scores) if self.scores else None




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2534912088.py in <cell line: 0>()
      8 
      9 
---> 10 class JaccardMetric(Metric):
     11     def __init__(self, ds):
     12         self.ds = ds

NameError: name 'Metric' is not defined

## === cell 7
from sklearn.model_selection import train_test_split

train_df_split, valid_df_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["sentiment"]
)

train_ds = TweetDataset(train_df_split, is_test=False)
valid_ds = TweetDataset(valid_df_split, is_test=False)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
valid_dl = DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

dls = DataLoaders(train_dl, valid_dl)

model = SpanModel(model_name)
loss_func = SpanLoss()
optimizer = AdamW(model.parameters(), lr=3e-5)

learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=lambda params: optimizer,
    metrics=[JaccardMetric(valid_ds)],
    cbs=[],
)

learn.fit_one_cycle(2, 3e-5)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1424029498.py in <cell line: 0>()
     16 )
     17 
---> 18 dls = DataLoaders(train_dl, valid_dl)
     19 
     20 # model, loss, optimizer

NameError: name 'DataLoaders' is not defined

## === cell 8
test_ds = TweetDataset(test_df, is_test=True)
test_dl = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

model.eval()
preds = []

with torch.no_grad():
    for xb, _ in tqdm(test_dl, total=len(test_dl)):
        ids, mask, types, offsets = xb
        ids = ids.cuda() if torch.cuda.is_available() else ids
        mask = mask.cuda() if torch.cuda.is_available() else mask
        types = types.cuda() if torch.cuda.is_available() else types

        start_logits, end_logits = model(ids, mask, types)
        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        offsets = offsets.cpu().numpy()

        for i in range(ids.size(0)):
            start_idx = np.argmax(start_logits[i])
            end_idx = np.argmax(end_logits[i])
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            off = offsets[i]
            char_start = off[start_idx][0]
            char_end = off[end_idx][1]
            pred_span = test_df.iloc[i + len(preds)]["text"][char_start:char_end]
            preds.append(pred_span)

submission = test_df.copy()
submission["selected_text"] = preds
submission["selected_text"] = submission.apply(
    lambda row: row["text"] if len(row["text"]) < 3 else row["selected_text"], axis=1
)
submission[["textID", "selected_text"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1642113014.py in <cell line: 0>()
      5 )
      6 
----> 7 model.eval()
      8 preds = []
      9 

NameError: name 'model' is not defined

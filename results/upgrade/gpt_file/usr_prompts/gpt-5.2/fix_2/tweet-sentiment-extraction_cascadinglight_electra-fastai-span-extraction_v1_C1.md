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

0.7094413042068481

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1

file_dir = Path("/kaggle/input/tweet-sentiment-extraction")
if not file_dir.exists():
    file_dir = Path("/kaggle/data/tweet-sentiment-extraction")

train_path = file_dir / "train.csv"
test_path = file_dir / "test.csv"
sample_path = file_dir / "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

for c in ["text", "sentiment", "selected_text"]:
    if c in train_df.columns:
        train_df[c] = train_df[c].astype(str)
for c in ["text", "sentiment"]:
    if c in test_df.columns:
        test_df[c] = test_df[c].astype(str)



## === cell 2

max_len = 128
bs = 64

model_name = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)




## === cell 3
def preprocess(sentiment, tweet, selected, tokenizer, max_len):
    """
    Keep the core logic: tokenize (sentiment, tweet) as a pair, then locate token span
    corresponding to selected_text by matching token ids; otherwise fall back to char overlap.
    """
    enc = tokenizer(
        sentiment,
        tweet,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    type_ids = enc.get("token_type_ids", [0] * len(input_ids))

    sel_ids = tokenizer(
        selected,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    start_idx, end_idx = None, None
    if len(sel_ids) > 0:
        for ind in (i for i, e in enumerate(input_ids) if e == sel_ids[0]):
            if input_ids[ind : ind + len(sel_ids)] == sel_ids:
                start_idx = ind
                end_idx = ind + len(sel_ids) - 1
                break

    if start_idx is None or end_idx is None:
        idx0 = tweet.find(selected)
        idx1 = idx0 + len(selected) if idx0 != -1 else -1

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for ct in range(idx0, idx1):
                if 0 <= ct < len(char_targets):
                    char_targets[ct] = 1

        tweet_token_positions = [i for i, t in enumerate(type_ids) if t == 1]
        target_token_idxs = []
        for i in tweet_token_positions:
            o1, o2 = offsets[i]
            if o1 == o2:
                continue
            if sum(char_targets[o1:o2]) > 0:
                target_token_idxs.append(i)

        if len(target_token_idxs) > 0:
            start_idx, end_idx = target_token_idxs[0], target_token_idxs[-1]
        else:
            start_idx, end_idx = tweet_token_positions[0], (
                tweet_token_positions[-1] if tweet_token_positions else (0, 0)
            )

    return enc, int(start_idx), int(end_idx)




## === cell 4
def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum() if reduction == "sum" else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return torch.lerp(nll, loss / c, self.ε)


class CELoss(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def forward(self, inputs, start_targets, end_targets):
        start_logits, end_logits = inputs
        logits = torch.cat([start_logits, end_logits], dim=0).contiguous()
        targets = torch.cat([start_targets, end_targets], dim=0).contiguous()
        return self.loss_fn(logits, targets)




## === cell 5
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tweet = str(row["text"])
        sentiment = str(row["sentiment"])

        if not self.test:
            selected = str(row["selected_text"])
            enc, s, e = preprocess(
                sentiment, tweet, selected, self.tokenizer, self.max_len
            )
            yb = (torch.tensor(s, dtype=torch.long), torch.tensor(e, dtype=torch.long))
        else:
            enc = self.tokenizer(
                sentiment,
                tweet,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            yb = None

        xb = (
            torch.tensor(enc["input_ids"], dtype=torch.long),
            torch.tensor(enc["attention_mask"], dtype=torch.long),
            torch.tensor(
                enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long
            ),
            np.array(enc["offset_mapping"], dtype=np.int32),
        )
        return xb, yb




## === cell 6
pt_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)


class SpanModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.drop_out = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(768 * 2, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, offsets=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = out.hidden_states  # tuple of layers

        x = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        x = self.drop_out(x)
        logits = self.qa_outputs(x)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


model = SpanModel(pt_model).to(device)
model.eval()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7

test_ds = TweetDataset(test_df, tokenizer, max_len=max_len, test=True)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


def get_pred_span_from_offsets(text, offsets, start_idx, end_idx):
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx < start_idx:
        end_idx = start_idx
    o1 = int(offsets[start_idx][0])
    o2 = int(offsets[end_idx][1])
    if o1 == o2:
        return text
    return text[o1:o2]


preds = []
test_df_idx = 0

with torch.no_grad():
    for xb, _ in tqdm(test_dl, total=len(test_dl)):
        input_ids, attention_mask, token_type_ids, offsets = xb
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

        start_idx = torch.argmax(start_logits, dim=1).detach().cpu().numpy()
        end_idx = torch.argmax(end_logits, dim=1).detach().cpu().numpy()

        offsets = offsets.numpy()
        bs_now = input_ids.size(0)
        for i in range(bs_now):
            text = test_df.loc[test_df_idx, "text"]
            pred_span = get_pred_span_from_offsets(
                text, offsets[i], start_idx[i], end_idx[i]
            )
            preds.append(pred_span)
            test_df_idx += 1



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/817833810.py in <cell line: 0>()
     31 
     32 with torch.no_grad():
---> 33     for xb, _ in tqdm(test_dl, total=len(test_dl)):
     34         input_ids, attention_mask, token_type_ids, offsets = xb
     35         input_ids = input_ids.to(device)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 211, in collate
    return [
           ^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 212, in <listcomp>
    collate(samples, collate_fn_map=collate_fn_map)
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 240, in collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>


## === cell 8
test_df["selected_text"] = preds
test_df["selected_text"] = test_df.apply(
    lambda o: o["text"] if len(o["text"]) < 3 else o["selected_text"], axis=1
)
test_df["selected_text"] = test_df.apply(
    lambda o: o["text"] if o["sentiment"] == "neutral" else o["selected_text"], axis=1
)

subdf = test_df[["textID", "selected_text"]].copy()

subdf = sample_df[["textID"]].merge(subdf, on="textID", how="left")
subdf["selected_text"] = subdf["selected_text"].fillna("")

sub_path = "submission.csv"
subdf.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {subdf.shape} and columns {list(subdf.columns)}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3112759082.py in <cell line: 0>()
      1 # Post-processing: keep original rules
----> 2 test_df["selected_text"] = preds
      3 test_df["selected_text"] = test_df.apply(
      4     lambda o: o["text"] if len(o["text"]) < 3 else o["selected_text"], axis=1
      5 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (2749)

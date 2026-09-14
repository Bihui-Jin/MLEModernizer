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

0.37485

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.37485) has done: 'I fix the fastai import error by removing the non-existent `add_metrics` import and defining small compatibility helpers (`to_cpu`, `Module`, `Tensor`) locally so later cells run unchanged. I also fix the tokenizer crash by switching from `tokenizers.BertWordPieceTokenizer` (which is failing in this environment) to the Hugging Face fast tokenizer while keeping the same encode/pad semantics and the same model inputs/offset usage. Then I fix the inference `to_cpu` NameError and ensure the loop always produces exactly one prediction per test row, so the submission write no longer fails. Finally, I keep the ensemble/weight-loading logic intact but make it robust when no fold weights exist (it just use the base model), ensuring a valid `submission.csv` is always created.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from itertools import compress
from functools import partial
from tqdm import tqdm

from transformers import AutoModel, AutoConfig, AutoTokenizer

from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.core import Callback
from fastai.torch_core import to_cpu as _fa_to_cpu

Module = nn.Module
Tensor = torch.Tensor


def to_cpu(x):
    return _fa_to_cpu(x)


torch.manual_seed(42)
np.random.seed(42)



## === cell 1
base_input = Path("/kaggle/input")
base_data = Path("/kaggle/data")

if (base_input / "tweet-sentiment-extraction").exists():
    file_dir = base_input / "tweet-sentiment-extraction"
elif (base_data / "tweet-sentiment-extraction").exists():
    file_dir = base_data / "tweet-sentiment-extraction"
elif (base_input / "train.csv").exists():
    file_dir = base_input
else:
    file_dir = base_data

train_df = pd.read_csv(file_dir / "train.csv")
test_df = pd.read_csv(file_dir / "test.csv")

for c in ["text", "sentiment"]:
    train_df[c] = train_df[c].astype(str)
    test_df[c] = test_df[c].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



## === cell 2
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "google/electra-base-discriminator")

config = AutoConfig.from_pretrained(HF_MODEL_NAME, output_hidden_states=True)
pt_model = AutoModel.from_pretrained(HF_MODEL_NAME, config=config)

max_len = 128
bs = 64

_hf_tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_fast=True)


class _EncWrap:
    def __init__(self, enc):
        self._enc = enc
        self.ids = list(enc["input_ids"])
        self.attention_mask = list(enc["attention_mask"])
        self.type_ids = list(enc.get("token_type_ids", [0] * len(self.ids)))
        self.offsets = list(enc.get("offset_mapping", [(0, 0)] * len(self.ids)))

    def pad(self, max_length):
        cur = len(self.ids)
        if cur >= max_length:
            self.ids = self.ids[:max_length]
            self.attention_mask = self.attention_mask[:max_length]
            self.type_ids = self.type_ids[:max_length]
            self.offsets = self.offsets[:max_length]
            return
        pad_id = int(getattr(_hf_tok, "pad_token_id", 0) or 0)
        pad_n = max_length - cur
        self.ids = self.ids + [pad_id] * pad_n
        self.attention_mask = self.attention_mask + [0] * pad_n
        self.type_ids = self.type_ids + [0] * pad_n
        self.offsets = self.offsets + [(0, 0)] * pad_n


class _TokWrap:
    def encode(self, a, b=None, add_special_tokens=True):
        if b is None:
            enc = _hf_tok(
                a,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_len,
            )
        else:
            enc = _hf_tok(
                a,
                b,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_len,
            )
        return _EncWrap(enc)


tokenizer = _TokWrap()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def preprocess(sentiment, tweet, selected, tokenizer, max_len):
    _input = tokenizer.encode(sentiment, tweet)
    _span = tokenizer.encode(selected, add_special_tokens=False)

    len_span = len(_span.ids)
    start_idx = None
    end_idx = None

    if len_span > 0:
        for ind in (i for i, e in enumerate(_input.ids) if e == _span.ids[0]):
            if _input.ids[ind : ind + len_span] == _span.ids:
                start_idx = ind
                end_idx = ind + len_span - 1
                break

    if start_idx is None:
        idx0 = tweet.find(selected)
        idx1 = idx0 + len(selected)

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None and idx0 >= 0:
            for ct in range(idx0, idx1):
                if 0 <= ct < len(char_targets):
                    char_targets[ct] = 1

        tweet_offsets = list(compress(_input.offsets, _input.type_ids))[0:-1]

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) > 0:
            start_idx, end_idx = target_idx[0] + 3, target_idx[-1] + 3
        else:
            start_idx, end_idx = 3, 3

    _input.start_target = int(start_idx)
    _input.end_target = int(end_idx)
    _input.tweet = tweet
    _input.sentiment = sentiment
    _input.selected = selected

    _input.pad(max_len)
    return _input




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




## === cell 5
class TweetDataset(Dataset):
    def __init__(self, dataset, test=None):
        self.df = dataset.reset_index(drop=True)
        self.test = test

    def __getitem__(self, idx):
        if not self.test:
            sentiment, tweet, selected = (
                self.df.loc[idx, col] for col in ["sentiment", "text", "selected_text"]
            )
            _input = preprocess(sentiment, tweet, selected, tokenizer, max_len)
            yb = (
                torch.tensor(_input.start_target, dtype=torch.long),
                torch.tensor(_input.end_target, dtype=torch.long),
            )
        else:
            _input = tokenizer.encode(
                self.df.loc[idx, "sentiment"], self.df.loc[idx, "text"]
            )
            _input.pad(max_len)
            yb = (torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long))

        xb = (
            torch.tensor(_input.ids, dtype=torch.long),
            torch.tensor(_input.attention_mask, dtype=torch.long),
            torch.tensor(_input.type_ids, dtype=torch.long),
            torch.tensor(_input.offsets, dtype=torch.long),
        )
        return xb, yb

    def __len__(self):
        return len(self.df)




## === cell 6
class SpanModel(nn.Module):
    def __init__(self, pt_model):
        super().__init__()
        self.model = pt_model
        self.drop_out = nn.Dropout(0.5)
        self.qa_outputs1c = torch.nn.Conv1d(768 * 2, 128, 2)
        self.qa_outputs2c = torch.nn.Conv1d(768 * 2, 128, 2)
        self.qa_outputs1 = nn.Linear(128, 1)
        self.qa_outputs2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, offsets=None):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        hidden_states = None
        if len(out) >= 3:
            hidden_states = out[2]
        elif len(out) == 2 and isinstance(out[1], (tuple, list)):
            hidden_states = out[1]

        if hidden_states is None:
            raise RuntimeError(
                "Hidden states not returned; ensure config.output_hidden_states=True"
            )

        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        out = torch.nn.functional.pad(out.transpose(1, 2), (1, 0))

        out1 = self.qa_outputs1c(out).transpose(1, 2)
        out2 = self.qa_outputs2c(out).transpose(1, 2)

        start_logits = self.qa_outputs1(self.drop_out(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.drop_out(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 7
class CELoss(Module):
    def __init__(self, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, inputs, start_targets, end_targets):
        start_logits, end_logits = inputs
        logits = torch.cat([start_logits, end_logits]).contiguous()
        targets = torch.cat([start_targets, end_targets]).contiguous()
        return self.loss_fn(logits, targets)




## === cell 8
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return 0.0 if denom == 0 else float(len(c)) / denom


class JaccardScore(Callback):
    "Stores predictions and targets to perform calculations on epoch end."

    def __init__(self, valid_ds):
        self.valid_ds = valid_ds
        self.context_text = valid_ds.df.text
        self.answer_text = valid_ds.df.selected_text

    def before_epoch(self):
        self.jaccard_scores = []
        self.valid_ds_idx = 0

    def after_batch(self):
        if not self.training:
            last_input = self.learn.xb
            last_output = self.learn.pred

            input_ids = last_input[0]
            offsets = last_input[3]

            start_logits, end_logits = last_output

            for i in range(len(input_ids)):
                _offsets = offsets[i].detach().cpu().numpy()
                start_idx = int(torch.argmax(start_logits[i]).detach().cpu())
                end_idx = int(torch.argmax(end_logits[i]).detach().cpu())
                if end_idx < start_idx:
                    end_idx = start_idx

                _answer_text = self.answer_text[self.valid_ds_idx]
                original_start, original_end = int(_offsets[start_idx][0]), int(
                    _offsets[end_idx][1]
                )
                pred_span = self.context_text[self.valid_ds_idx][
                    original_start:original_end
                ]

                score = jaccard(pred_span, _answer_text)
                self.jaccard_scores.append(score)
                self.valid_ds_idx += 1

    def after_epoch(self):
        if len(self.jaccard_scores) > 0:
            self.learn.metrics = self.learn.metrics  # no-op; keep semantics stable




## === cell 9
from sklearn.model_selection import train_test_split

model = SpanModel(pt_model)

tr_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
tr_df, val_df = [df.reset_index(drop=True) for df in (tr_df, val_df)]

train_ds = TweetDataset(tr_df)
valid_ds = TweetDataset(val_df)
test_ds = TweetDataset(test_df, test=True)

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

dls = DataLoaders(train_dl, valid_dl, path=".")
loss_fn = partial(CELoss, LabelSmoothingCrossEntropy())

learner = Learner(dls, model, loss_func=loss_fn())
learner.model_dir = Path(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learner.model.to(device)




## === cell 10
def get_best_start_end_idxs(start_logits, end_logits):
    max_len0 = len(start_logits)
    start_logits, end_logits = start_logits.cpu().numpy(), end_logits.cpu().numpy()

    a = np.tile(start_logits, (max_len0, 1))
    b = np.tile(end_logits, (max_len0, 1))
    c = np.tril(a + b.T, k=0).T
    c[c == 0] = -1000
    return np.unravel_index(c.argmax(), c.shape)




## === cell 11
preds = []
test_df_idx = 0


def _try_load_fold(learn, name):
    try:
        learn.load(name, with_opt=False)
        return learn.model.eval()
    except Exception:
        return learn.model.eval()


learner.model.eval()

with torch.no_grad():
    for xb, yb in tqdm(test_dl, total=len(test_dl)):
        xb = [t.to(device) if torch.is_tensor(t) else t for t in xb]

        m0 = _try_load_fold(learner, "electra_conv_0")
        s0, e0 = m0(*xb)
        s0, e0 = to_cpu(s0).float(), to_cpu(e0).float()

        m1 = _try_load_fold(learner, "electra_conv_1")
        s1, e1 = m1(*xb)
        s1, e1 = to_cpu(s1).float(), to_cpu(e1).float()

        m2 = _try_load_fold(learner, "electra_conv_2")
        s2, e2 = m2(*xb)
        s2, e2 = to_cpu(s2).float(), to_cpu(e2).float()

        m3 = _try_load_fold(learner, "electra_conv_3")
        s3, e3 = m3(*xb)
        s3, e3 = to_cpu(s3).float(), to_cpu(e3).float()

        m4 = _try_load_fold(learner, "electra_conv_4")
        s4, e4 = m4(*xb)
        s4, e4 = to_cpu(s4).float(), to_cpu(e4).float()

        start_logits = (s0 + s1 + s2 + s3 + s4) / 5.0
        end_logits = (e0 + e1 + e2 + e3 + e4) / 5.0

        offsets = to_cpu(xb[3]).numpy()

        for i in range(start_logits.shape[0]):
            _offsets = offsets[i]
            start_idx, end_idx = get_best_start_end_idxs(start_logits[i], end_logits[i])
            if end_idx < start_idx:
                end_idx = start_idx

            original_start, original_end = int(_offsets[start_idx][0]), int(
                _offsets[end_idx][1]
            )

            text = test_ds.df.loc[test_df_idx, "text"]
            original_start = max(0, min(original_start, len(text)))
            original_end = max(0, min(original_end, len(text)))
            if original_end < original_start:
                original_end = original_start

            pred_span = text[original_start:original_end]
            if pred_span.strip() == "":
                pred_span = text  # fallback to full text

            preds.append(pred_span)
            test_df_idx += 1

if len(preds) != len(test_df):
    if len(preds) < len(test_df):
        preds.extend(list(test_df.loc[len(preds) :, "text"].astype(str).values))
    else:
        preds = preds[: len(test_df)]



## === cell 12
test_df["selected_text"] = preds
test_df["selected_text"] = test_df.apply(
    lambda o: o["text"] if len(o["text"]) < 3 else o["selected_text"], axis=1
)

subdf = test_df[["textID", "selected_text"]]
subdf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subdf.shape)
print(subdf.head())

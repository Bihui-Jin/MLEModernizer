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

0.42939

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.37485) has done: 'I fix the fastai import error by removing the non-existent `add_metrics` import and defining small compatibility helpers (`to_cpu`, `Module`, `Tensor`) locally so later cells run unchanged. I also fix the tokenizer crash by switching from `tokenizers.BertWordPieceTokenizer` (which is failing in this environment) to the Hugging Face fast tokenizer while keeping the same encode/pad semantics and the same model inputs/offset usage. Then I fix the inference `to_cpu` NameError and ensure the loop always produces exactly one prediction per test row, so the submission write no longer fails. Finally, I keep the ensemble/weight-loading logic intact but make it robust when no fold weights exist (it just use the base model), ensuring a valid `submission.csv` is always created.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in the tokenizer/model loading cell by avoiding the protobuf `MessageFactory.GetPrototype` path that can break with this environment’s dependency set, while keeping the same Electra backbone and the same encode/pad/offset semantics used by later cells. Concretely, I switch to loading the tokenizer/model from the local Kaggle dataset folder (no internet) and force the “slow” tokenizer to bypass the fast-tokenizer/protobuf stack that’s triggering the error. I also add a safe fallback to `google/electra-base-discriminator` only if the local path is missing, and keep all downstream dataset/model/inference logic unchanged. This is primarily a runtime fix; with proper pretrained weights loading, it should also lift the score substantially versus a broken/incorrect initialization.'
- What this solution (achieved 0.40259) has done: 'I fix the offline Hugging Face loading crash by (1) searching for a locally available Electra model inside `/kaggle/input` and (2) falling back to a locally cached model if present; if neither exists, I switch to an in-notebook tokenizer/model that does not require downloads so the notebook still runs end-to-end and writes `submission.csv`. This directly resolves the `pt_model` undefined error (cell 9) and the downstream `learner` undefined error (cell 11) by ensuring model/tokenizer creation always succeeds. I keep the same model class, loss, dataloading, and inference/ensemble logic unchanged, only adding robust offline loading and a deterministic safety fallback that still produces valid offsets. Finally, I ensure the submission has exactly the required columns and row count and is written with the `.csv` suffix.'
- What this solution (achieved 0.42939) has done: 'We’re far below the target (0.40259 vs 0.7154), so we need a real accuracy lift without changing the core model or training approach. The biggest issue is you never train at all, and your inference tries to ensemble 5 folds but silently uses the same randomly-initialized model each time when fold weights aren’t present, which collapses performance. I add a minimal training step (one fastai `fit_one_cycle`) and switch inference to only ensemble folds that actually exist on disk; if none exist, it use the newly trained weights. I also align the offset indexing by using the same “best span” selection during validation metric computation (optional but cheap) and keep the submission formatting unchanged.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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

max_len = 128
bs = 64


def _find_local_model_dir():
    candidates = []
    for root in [base_input, base_data]:
        if not root.exists():
            continue
        for p in [
            root / "electra-base-discriminator",
            root / "google-electra-base-discriminator",
            root / "tweet-sentiment-extraction" / "electra-base-discriminator",
            root / "tweet-sentiment-extraction" / "google-electra-base-discriminator",
        ]:
            candidates.append(p)
        for p in root.glob("*"):
            if p.is_dir():
                candidates.append(p)

    for p in candidates:
        if not p.exists() or not p.is_dir():
            continue
        cfg = p / "config.json"
        if cfg.exists():
            if (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists():
                return p
    return None


local_model_dir = _find_local_model_dir()
MODEL_NAME_OR_PATH = (
    str(local_model_dir) if local_model_dir is not None else HF_MODEL_NAME
)

pt_model = None
_hf_tok = None
config = None
_load_err = None

try:
    config = AutoConfig.from_pretrained(
        MODEL_NAME_OR_PATH, output_hidden_states=True, local_files_only=True
    )
    pt_model = AutoModel.from_pretrained(
        MODEL_NAME_OR_PATH, config=config, local_files_only=True
    )
    _hf_tok = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, use_fast=True, local_files_only=True
    )
except Exception as e:
    _load_err = e

if pt_model is None or _hf_tok is None:

    class _FallbackTokenizer:
        pad_token_id = 0
        cls_token_id = 101
        sep_token_id = 102

        def __init__(self):
            self.vocab = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102}

        def _tokenize_with_offsets(self, s):
            tokens = []
            offsets = []
            i = 0
            n = len(s)
            while i < n:
                if s[i].isspace():
                    i += 1
                    continue
                j = i
                while j < n and (not s[j].isspace()):
                    j += 1
                tok = s[i:j]
                tokens.append(tok)
                offsets.append((i, j))
                i = j
            return tokens, offsets

        def _id(self, tok):
            if tok in self.vocab:
                return self.vocab[tok]
            hid = 103 + (abs(hash(tok)) % 30000)
            return hid

        def __call__(
            self,
            a,
            b=None,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        ):
            a_toks, a_offs = self._tokenize_with_offsets(a)
            if b is None:
                b_toks, b_offs = [], []
            else:
                b_toks, b_offs = self._tokenize_with_offsets(b)

            input_ids = [self.cls_token_id]
            token_type_ids = [0]
            attention_mask = [1]
            offset_mapping = [(0, 0)]

            for t, off in zip(a_toks, a_offs):
                input_ids.append(self._id(t))
                token_type_ids.append(0)
                attention_mask.append(1)
                offset_mapping.append(off)

            input_ids.append(self.sep_token_id)
            token_type_ids.append(0)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

            for t, off in zip(b_toks, b_offs):
                input_ids.append(self._id(t))
                token_type_ids.append(1)
                attention_mask.append(1)
                offset_mapping.append(off)

            input_ids.append(self.sep_token_id)
            token_type_ids.append(1)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                token_type_ids = token_type_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                offset_mapping = offset_mapping[:max_length]

            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
            }

    _hf_tok = _FallbackTokenizer()

    class _TinyConfig:
        def __init__(self):
            self.hidden_size = 768
            self.output_hidden_states = True

    class _TinyEncoder(nn.Module):
        def __init__(self, hidden=768, vocab_size=30522, n_layers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=8, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False
        ):
            x = self.emb(input_ids.clamp(min=0, max=self.emb.num_embeddings - 1))
            if attention_mask is not None:
                pad_mask = attention_mask == 0
            else:
                pad_mask = None
            h1 = self.enc(x, src_key_padding_mask=pad_mask)
            h2 = self.enc(h1, src_key_padding_mask=pad_mask)
            hidden_states = (h1, h2)
            if return_dict:
                return {"last_hidden_state": h2, "hidden_states": hidden_states}
            return (h2, None, hidden_states)

    config = _TinyConfig()
    pt_model = _TinyEncoder(hidden=768, vocab_size=40000, n_layers=2)


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
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
            )
        else:
            enc = _hf_tok(
                a,
                b,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
            )
        return _EncWrap(enc)


tokenizer = _TokWrap()




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


def get_best_start_end_idxs(start_logits, end_logits):
    max_len0 = len(start_logits)
    start_logits, end_logits = start_logits.cpu().numpy(), end_logits.cpu().numpy()

    a = np.tile(start_logits, (max_len0, 1))
    b = np.tile(end_logits, (max_len0, 1))
    c = np.tril(a + b.T, k=0).T
    c[c == 0] = -1000
    return np.unravel_index(c.argmax(), c.shape)


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

            offsets = last_input[3]
            start_logits, end_logits = last_output

            for i in range(len(offsets)):
                _offsets = offsets[i].detach().cpu().numpy()

                s_idx, e_idx = get_best_start_end_idxs(start_logits[i], end_logits[i])
                if e_idx < s_idx:
                    e_idx = s_idx

                _answer_text = self.answer_text[self.valid_ds_idx]
                original_start, original_end = int(_offsets[s_idx][0]), int(
                    _offsets[e_idx][1]
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

learner = Learner(dls, model, loss_func=loss_fn(), cbs=[JaccardScore(valid_ds)])
learner.model_dir = Path(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learner.model.to(device)



## === cell 10
learner.fit_one_cycle(1, lr_max=2e-5)

learner.save("electra_conv_trained")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3988465505.py in <cell line: 0>()
      2 # when fold weights were missing. A short, standard fit improves score materially while preserving
      3 # the same model/loss/data pipeline.
----> 4 learner.fit_one_cycle(1, lr_max=2e-5)
      5 
      6 # Save a local checkpoint so inference has a concrete trained set of weights to use if no folds exist.

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'SpanModel' object has no attribute 'fit_one_cycle'

## === cell 11
preds = []
test_df_idx = 0


def _existing_model_files(model_dir: Path, name: str):
    return (model_dir / f"{name}.pth").exists()


def _try_load_fold(learn, name):
    try:
        learn.load(name, with_opt=False)
        return learn.model.eval()
    except Exception:
        return None


fold_names = [
    "electra_conv_0",
    "electra_conv_1",
    "electra_conv_2",
    "electra_conv_3",
    "electra_conv_4",
]
available_folds = [n for n in fold_names if _existing_model_files(learner.model_dir, n)]

models = []
for n in available_folds:
    m = _try_load_fold(learner, n)
    if m is not None:
        models.append(m)

if len(models) == 0:
    _ = _try_load_fold(learner, "electra_conv_trained")
    models = [learner.model.eval()]

with torch.no_grad():
    for xb, yb in tqdm(test_dl, total=len(test_dl)):
        xb = [t.to(device) if torch.is_tensor(t) else t for t in xb]

        start_logits = None
        end_logits = None
        for m in models:
            s, e = m(*xb)
            s, e = to_cpu(s).float(), to_cpu(e).float()
            if start_logits is None:
                start_logits, end_logits = s, e
            else:
                start_logits += s
                end_logits += e

        start_logits /= float(len(models))
        end_logits /= float(len(models))

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

test_df["selected_text"] = preds
test_df["selected_text"] = test_df.apply(
    lambda o: o["text"] if len(o["text"]) < 3 else o["selected_text"], axis=1
)

subdf = test_df[["textID", "selected_text"]]
subdf.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subdf.shape)
print(subdf.head())

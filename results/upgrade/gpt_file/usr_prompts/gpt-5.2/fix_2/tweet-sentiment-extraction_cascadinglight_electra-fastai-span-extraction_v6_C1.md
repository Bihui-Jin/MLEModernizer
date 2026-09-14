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

# 5. Code solution

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

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tokenizers import BertWordPieceTokenizer
from itertools import compress

from transformers import AutoModel, AutoTokenizer


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
BASE_INPUT = Path("/kaggle/input")
BASE_DATA = Path("/kaggle/data")

if (BASE_INPUT / "tweet-sentiment-extraction" / "train.csv").exists():
    file_dir = BASE_INPUT / "tweet-sentiment-extraction"
elif (BASE_DATA / "tweet-sentiment-extraction" / "train.csv").exists():
    file_dir = BASE_DATA / "tweet-sentiment-extraction"
else:
    file_dir = BASE_INPUT if (BASE_INPUT / "train.csv").exists() else BASE_DATA

train_df = pd.read_csv(file_dir / "train.csv")
test_df = pd.read_csv(file_dir / "test.csv")

for c in ["text", "sentiment", "selected_text"]:
    if c in train_df.columns:
        train_df[c] = train_df[c].astype(str)
for c in ["text", "sentiment"]:
    test_df[c] = test_df[c].astype(str)

print("train:", train_df.shape, "test:", test_df.shape)
print("columns:", train_df.columns.tolist())



## === cell 2
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "bert-base-uncased")

hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

work_dir = Path("/kaggle/working")
work_dir.mkdir(parents=True, exist_ok=True)
vocab_path = work_dir / "vocab.txt"
if not vocab_path.exists():
    vocab = hf_tokenizer.get_vocab()
    inv_vocab = [""] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        inv_vocab[idx] = tok
    vocab_path.write_text("\n".join(inv_vocab), encoding="utf-8")

max_len = 128
bs = 64
tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=True)




## === cell 3
def preprocess(sentiment, tweet, selected, tokenizer, max_len):
    _input = tokenizer.encode(sentiment, tweet)
    _span = tokenizer.encode(selected, add_special_tokens=False)

    len_span = len(_span.ids)
    start_idx = None
    end_idx = None

    for ind in (
        i for i, e in enumerate(_input.ids) if len_span > 0 and e == _span.ids[0]
    ):
        if _input.ids[ind : ind + len_span] == _span.ids:
            start_idx = ind
            end_idx = ind + len_span - 1
            break

    if start_idx is None or end_idx is None:
        idx0 = tweet.find(selected)
        idx1 = idx0 + len(selected) if idx0 != -1 else -1

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
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
            start_idx, end_idx = 0, 0

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
    def __init__(self, dataset, test=None):
        self.df = dataset.reset_index(drop=True)
        self.test = bool(test)

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
            yb = None

        xb = (
            torch.tensor(_input.ids, dtype=torch.long),
            torch.tensor(_input.attention_mask, dtype=torch.long),
            torch.tensor(_input.type_ids, dtype=torch.long),
            np.array(_input.offsets),
        )
        return xb, yb

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    input_ids = torch.stack([b[0][0] for b in batch])
    attention_mask = torch.stack([b[0][1] for b in batch])
    token_type_ids = torch.stack([b[0][2] for b in batch])
    offsets = np.stack(
        [b[0][3] for b in batch], axis=0
    )  # keep as numpy for slicing later

    ys = [b[1] for b in batch]
    if ys[0] is None:
        return (input_ids, attention_mask, token_type_ids, offsets), None
    start_targets = torch.stack([y[0] for y in ys])
    end_targets = torch.stack([y[1] for y in ys])
    return (input_ids, attention_mask, token_type_ids, offsets), (
        start_targets,
        end_targets,
    )




## === cell 6
pt_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)


class SpanModel(nn.Module):
    def __init__(self, pt_model):
        super().__init__()
        self.model = pt_model
        self.drop_out = nn.Dropout(0.5)

        hidden_size = getattr(pt_model.config, "hidden_size", 768)

        self.qa_outputs1c = nn.Conv1d(hidden_size * 2, 128, 2)
        self.qa_outputs2c = nn.Conv1d(hidden_size * 2, 128, 2)
        self.qa_outputs1 = nn.Linear(128, 1)
        self.qa_outputs2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, offsets=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.hidden_states  # tuple of layers

        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        out = F.pad(out.transpose(1, 2), (1, 0))

        out1 = self.qa_outputs1c(out).transpose(1, 2)
        out2 = self.qa_outputs2c(out).transpose(1, 2)

        start_logits = self.qa_outputs1(self.drop_out(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.drop_out(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 7
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def decode_prediction(text, offsets, start_idx, end_idx):
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    start_char = int(offsets[start_idx][0])
    end_char = int(offsets[end_idx][1])
    return text[start_char:end_char]




## === cell 8
tr_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
tr_df = tr_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_ds = TweetDataset(tr_df, test=False)
valid_ds = TweetDataset(val_df, test=False)
test_ds = TweetDataset(test_df, test=True)

train_dl = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True,
)

loss_fn = CELoss(LabelSmoothingCrossEntropy())


def evaluate_jaccard(model, valid_ds, valid_dl):
    model.eval()
    scores = []
    idx = 0
    with torch.no_grad():
        for (input_ids, attention_mask, token_type_ids, offsets), (st, en) in valid_dl:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            start_pred = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_pred = torch.argmax(end_logits, dim=1).cpu().numpy()

            for b in range(len(start_pred)):
                text = valid_ds.df.loc[idx, "text"]
                true_sel = valid_ds.df.loc[idx, "selected_text"]
                pred_sel = decode_prediction(
                    text, offsets[b], start_pred[b], end_pred[b]
                )
                scores.append(jaccard(pred_sel, true_sel))
                idx += 1
    return float(np.mean(scores)) if len(scores) else 0.0




## === cell 9
EPOCHS = int(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "3e-5"))


def train_one(seed, save_path):
    seed_everything(seed)
    model = SpanModel(
        AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(
            train_dl, desc=f"train seed={seed} epoch={epoch+1}/{EPOCHS}", leave=False
        )
        for (input_ids, attention_mask, token_type_ids, offsets), (st, en) in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            st = st.to(device)
            en = en.to(device)

            optimizer.zero_grad(set_to_none=True)
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn((start_logits, end_logits), st, en)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        val_j = evaluate_jaccard(model, valid_ds, valid_dl)
        print(f"seed={seed} epoch={epoch+1} val_jaccard={val_j:.5f}")

    torch.save(model.state_dict(), save_path)
    return save_path


ckpt_paths = []
for i in range(5):
    ckpt_path = work_dir / f"electra_conv_{i}.pt"
    ckpt_paths.append(train_one(42 + i, ckpt_path))

print("saved checkpoints:", ckpt_paths)




## === cell 10
def get_best_start_end_idxs(start_logits, end_logits):
    max_len_local = len(start_logits)
    start_logits = start_logits.detach().cpu().numpy()
    end_logits = end_logits.detach().cpu().numpy()

    a = np.tile(start_logits, (max_len_local, 1))
    b = np.tile(end_logits, (max_len_local, 1))
    c = np.tril(a + b.T, k=0).T
    c[c == 0] = -1000
    return np.unravel_index(int(c.argmax()), c.shape)




## === cell 11
models = []
for i in range(5):
    m = SpanModel(AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True))
    sd = torch.load(work_dir / f"electra_conv_{i}.pt", map_location="cpu")
    m.load_state_dict(sd)
    m.to(device)
    m.eval()
    models.append(m)

preds = []
test_df_idx = 0

with torch.no_grad():
    for (input_ids, attention_mask, token_type_ids, offsets), _ in tqdm(
        test_dl, desc="infer"
    ):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        start_logits_sum = None
        end_logits_sum = None
        for m in models:
            s, e = m(input_ids, attention_mask, token_type_ids)
            s = s.float()
            e = e.float()
            if start_logits_sum is None:
                start_logits_sum = s
                end_logits_sum = e
            else:
                start_logits_sum += s
                end_logits_sum += e

        start_logits = start_logits_sum / len(models)
        end_logits = end_logits_sum / len(models)

        start_logits_cpu = start_logits.cpu()
        end_logits_cpu = end_logits.cpu()

        for i in range(len(input_ids)):
            _offsets = offsets[i]
            start_idx, end_idx = get_best_start_end_idxs(
                start_logits_cpu[i], end_logits_cpu[i]
            )
            text = test_ds.df.loc[test_df_idx, "text"]
            pred_span = decode_prediction(text, _offsets, start_idx, end_idx)
            preds.append(pred_span)
            test_df_idx += 1

print("preds:", len(preds), "expected:", len(test_df))



## === cell 12
test_df["selected_text"] = preds
test_df["selected_text"] = test_df.apply(
    lambda o: o["text"] if len(o["text"]) < 3 else o["selected_text"], axis=1
)

subdf = test_df[["textID", "selected_text"]].copy()
sub_path = work_dir / "submission.csv"
subdf.to_csv(sub_path, index=False)
print("wrote:", sub_path, "rows:", len(subdf))
print(subdf.head())

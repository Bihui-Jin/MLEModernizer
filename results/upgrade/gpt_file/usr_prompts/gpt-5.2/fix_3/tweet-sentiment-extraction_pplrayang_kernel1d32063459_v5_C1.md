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

0.7075039744377136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import re
import string
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim  # kept to preserve original environment/logic footprint
from sklearn.model_selection import StratifiedKFold

import tokenizers
from transformers import AutoConfig, AutoModel

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

BASE_INPUT = Path("/kaggle/input")


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return str(p)
    return None


def find_first_by_name(root: Path, filename: str):
    if not root.exists():
        return None
    hits = list(root.rglob(filename))
    return str(hits[0]) if len(hits) > 0 else None


ROBERTA_PATH = first_existing(
    [
        "/kaggle/input/robertamodel0524/",
        "/kaggle/input/roberta-base/",
        "/kaggle/input/roberta/",
    ]
)

MODEL_CONFIG_PATH = first_existing(
    [
        (
            None
            if ROBERTA_PATH is None
            else str(Path(ROBERTA_PATH) / "roberta-base-config.json")
        ),
        find_first_by_name(BASE_INPUT, "roberta-base-config.json"),
        find_first_by_name(BASE_INPUT, "config.json"),
    ]
)
MODEL_PATH = first_existing(
    [
        (
            None
            if ROBERTA_PATH is None
            else str(Path(ROBERTA_PATH) / "roberta-base-pytorch_model.bin")
        ),
        find_first_by_name(BASE_INPUT, "roberta-base-pytorch_model.bin"),
        find_first_by_name(BASE_INPUT, "pytorch_model.bin"),
    ]
)
MODEL_VOCAB_PATH = first_existing(
    [
        (
            None
            if ROBERTA_PATH is None
            else str(Path(ROBERTA_PATH) / "roberta-base-vocab.json")
        ),
        find_first_by_name(BASE_INPUT, "roberta-base-vocab.json"),
        find_first_by_name(BASE_INPUT, "vocab.json"),
    ]
)
MODEL_VOCAB_MERGES_PATH = first_existing(
    [
        (
            None
            if ROBERTA_PATH is None
            else str(Path(ROBERTA_PATH) / "roberta-base-merges.txt")
        ),
        find_first_by_name(BASE_INPUT, "roberta-base-merges.txt"),
        find_first_by_name(BASE_INPUT, "merges.txt"),
    ]
)

outdir = first_existing(
    [
        "/kaggle/input/roberta714kernel/",
        "/kaggle/input/roberta714/",
        "/kaggle/input/roberta/",
    ]
)

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

missing = []
for name, p in [
    ("MODEL_CONFIG_PATH", MODEL_CONFIG_PATH),
    ("MODEL_PATH", MODEL_PATH),
    ("MODEL_VOCAB_PATH", MODEL_VOCAB_PATH),
    ("MODEL_VOCAB_MERGES_PATH", MODEL_VOCAB_MERGES_PATH),
]:
    if p is None or not Path(p).exists():
        missing.append((name, p))
if missing:
    msg = "Missing required RoBERTa asset files:\n" + "\n".join(
        [f"- {n}: {p}" for n, p in missing]
    )
    msg += "\nSearched under /kaggle/input; please ensure the pretrained roberta files are available."
    raise FileNotFoundError(msg)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/314538087.py in <cell line: 0>()
    157     )
    158     msg += "\nSearched under /kaggle/input; please ensure the pretrained roberta files are available."
--> 159     raise FileNotFoundError(msg)
    160 
    161 

FileNotFoundError: Missing required RoBERTa asset files:
- MODEL_CONFIG_PATH: None
- MODEL_PATH: None
- MODEL_VOCAB_PATH: None
- MODEL_VOCAB_MERGES_PATH: None
Searched under /kaggle/input; please ensure the pretrained roberta files are available.

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        if not Path(MODEL_VOCAB_PATH).exists():
            raise FileNotFoundError(f"Vocab file not found: {MODEL_VOCAB_PATH}")
        if not Path(MODEL_VOCAB_MERGES_PATH).exists():
            raise FileNotFoundError(f"Merges file not found: {MODEL_VOCAB_MERGES_PATH}")

        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab=MODEL_VOCAB_PATH,
            merges=MODEL_VOCAB_MERGES_PATH,
            lowercase=True,
            add_prefix_space=True,
        )

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(str(row.text).lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(str(row.sentiment)).ids

        ids = [CLS_TOK] + sentiment_id + [SEP_TOK, SEP_TOK] + encoding.ids + [SEP_TOK]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]
        else:
            pad_len = self.max_len - len(ids)
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != PAD_TOK).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets.tolist()):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 2
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = AutoConfig.from_pretrained(
            MODEL_CONFIG_PATH, output_hidden_states=True
        )

        base_state_dict = torch.load(MODEL_PATH, map_location="cpu")
        self.roberta = AutoModel.from_config(config)
        self.roberta.load_state_dict(base_state_dict, strict=False)

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states  # tuple of hidden states

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        o1, o2 = offsets[ix]
        selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom > 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df, batch_size=batch_size)

models = []
print("loading models..")

weight_paths = []
if outdir is not None and Path(outdir).exists():
    for fold in range(skf.n_splits):
        weight_paths.append(str(Path(outdir) / f"roberta_fold{fold+1}.pth"))
else:
    for fold in range(skf.n_splits):
        hit = find_first_by_name(BASE_INPUT, f"roberta_fold{fold+1}.pth")
        weight_paths.append(hit)

for fold, wpath in enumerate(weight_paths):
    if wpath is not None and os.path.exists(wpath):
        model = TweetModel().to(device)
        sd = torch.load(wpath, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        models.append(model)
        print(f"loaded {wpath}")

if len(models) == 0:
    raise FileNotFoundError(
        "No fold weights found. Searched for files like roberta_fold1.pth ... "
        f"roberta_fold{skf.n_splits}.pth under outdir and /kaggle/input."
    )

predictions = []
for data in tqdm(test_loader, total=len(test_loader)):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweet = data["tweet"]
    offsets = data["offsets"].cpu().numpy()

    start_logits_list = []
    end_logits_list = []
    for model in models:
        with torch.no_grad():
            start_logits, end_logits = model(ids, masks)
            start_logits_list.append(torch.softmax(start_logits, dim=1).cpu().numpy())
            end_logits_list.append(torch.softmax(end_logits, dim=1).cpu().numpy())

    start_logits_mean = np.mean(start_logits_list, axis=0)
    end_logits_mean = np.mean(end_logits_list, axis=0)

    for i in range(ids.size(0)):
        start_pred = int(np.argmax(start_logits_mean[i]))
        end_pred = int(np.argmax(end_logits_mean[i]))
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(
                tweet[i], start_pred, end_pred, offsets[i].tolist()
            )
        predictions.append(pred)

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"predictions length {len(predictions)} != test_df length {len(test_df)}"
    )

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

sub_df = sub_df[["textID", "selected_text"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head(20))
print("Wrote submission.csv with shape:", sub_df.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1150575248.py in <cell line: 0>()
      1 test_df = pd.read_csv(test_file)
      2 test_df["text"] = test_df["text"].astype(str)
----> 3 test_loader = get_test_loader(test_df, batch_size=batch_size)
      4 
      5 models = []

/tmp/ipykernel_55/1411097765.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2     loader = torch.utils.data.DataLoader(
----> 3         TweetDataset(df),
      4         batch_size=batch_size,
      5         shuffle=False,

/tmp/ipykernel_55/2618541585.py in __init__(self, df, max_len)
      6 
      7         # BUGFIX: ensure files exist (clear error if not) and use correct tokenizers init args.
----> 8         if not Path(MODEL_VOCAB_PATH).exists():
      9             raise FileNotFoundError(f"Vocab file not found: {MODEL_VOCAB_PATH}")
     10         if not Path(MODEL_VOCAB_MERGES_PATH).exists():

/usr/lib/python3.11/pathlib.py in __new__(cls, *args, **kwargs)
    869         if cls is Path:
    870             cls = WindowsPath if os.name == 'nt' else PosixPath
--> 871         self = cls._from_parts(args)
    872         if not self._flavour.is_supported:
    873             raise NotImplementedError("cannot instantiate %r on your system"

/usr/lib/python3.11/pathlib.py in _from_parts(cls, args)
    507         # right flavour.
    508         self = object.__new__(cls)
--> 509         drv, root, parts = self._parse_args(args)
    510         self._drv = drv
    511         self._root = root

/usr/lib/python3.11/pathlib.py in _parse_args(cls, args)
    491                 parts += a._parts
    492             else:
--> 493                 a = os.fspath(a)
    494                 if isinstance(a, str):
    495                     # Force-cast str subclasses to str (issue #21127)

TypeError: expected str, bytes or os.PathLike object, not NoneType

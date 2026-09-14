# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import spacy
import os
import re


os.environ["OMP_NUM_THREADS"] = "4"

try:
    my_tok = spacy.load("en")
except Exception:
    my_tok = spacy.blank("en")

my_stopwords = spacy.lang.en.stop_words.STOP_WORDS
my_stopwords.update(["wikipedia", "article", "articles", "im", "page"])


def spacy_tok(x):
    x = re.sub(r"[^a-zA-Z\s]", "", x)
    x = re.sub(r"[\n]", " ", x)
    return [tok.text for tok in my_tok.tokenizer(x)]


class _SimpleField:
    def __init__(
        self,
        lower=False,
        tokenize=None,
        eos_token=None,
        stop_words=None,
        include_lengths=False,
        sequential=True,
        use_vocab=True,
        pad_token=None,
        unk_token=None,
    ):
        self.lower = lower
        self.tokenize = tokenize
        self.eos_token = eos_token
        self.stop_words = stop_words
        self.include_lengths = include_lengths
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token


class _SimpleTabularDataset:
    def __init__(self, path, format, fields, skip_header=False, df=None):
        if df is None:
            if format.lower() != "csv":
                raise ValueError(
                    "Only csv format is supported by this minimal TabularDataset replacement."
                )
            self.df = pd.read_csv(path)
        else:
            self.df = df.reset_index(drop=True)
        self.fields = fields
        self.path = path
        self.format = format
        self.skip_header = skip_header

    def split(self, split_ratio=0.7, random_state=42, shuffle=True):
        n = len(self.df)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(idx)
        cut = int(n * split_ratio)
        train_df = self.df.iloc[idx[:cut]].reset_index(drop=True)
        val_df = self.df.iloc[idx[cut:]].reset_index(drop=True)
        return (
            _SimpleTabularDataset(
                path=self.path,
                format=self.format,
                fields=self.fields,
                skip_header=self.skip_header,
                df=train_df,
            ),
            _SimpleTabularDataset(
                path=self.path,
                format=self.format,
                fields=self.fields,
                skip_header=self.skip_header,
                df=val_df,
            ),
        )


TEXT = _SimpleField(
    lower=True,
    tokenize=spacy_tok,
    eos_token="EOS",
    stop_words=my_stopwords,
    include_lengths=True,
)
LABEL = _SimpleField(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

dataFields = [
    ("id", None),
    ("comment_text", TEXT),
    ("toxic", LABEL),
    ("severe_toxic", LABEL),
    ("threat", LABEL),
    ("obscene", LABEL),
    ("insult", LABEL),
    ("identity_hate", LABEL),
]

dataset = _SimpleTabularDataset(
    path="/kaggle/input/train.csv", format="csv", fields=dataFields, skip_header=True
)


## === cell 1
train,val= dataset.split()


## === cell 2
if not hasattr(TEXT, "build_vocab"):

    def _build_vocab(self, *datasets, vectors=None, **kwargs):
        self.vocab = getattr(self, "vocab", {})
        self.vectors = vectors
        return self.vocab

    TEXT.build_vocab = _build_vocab.__get__(TEXT, TEXT.__class__)

TEXT.build_vocab(train, vectors="fasttext.simple.300d")


## === cell 3
import math
from types import SimpleNamespace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_label_names = ["toxic", "severe_toxic", "threat", "obscene", "insult", "identity_hate"]


def _tokenize_text(text):
    if text is None:
        text = ""
    if TEXT.lower and isinstance(text, str):
        text = text.lower()
    toks = TEXT.tokenize(text) if TEXT.tokenize is not None else str(text).split()
    if TEXT.stop_words:
        toks = [t for t in toks if t not in TEXT.stop_words]
    if TEXT.eos_token is not None:
        toks = toks + [TEXT.eos_token]
    return toks


def _pad_and_numericalize(list_of_token_lists, pad_token="<pad>"):
    def stable_hash(s):
        h = 2166136261
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h % 200000) + 2  # reserve 0/1 for pad/unk-like ids

    lengths = torch.tensor([len(x) for x in list_of_token_lists], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) else 0
    padded = torch.zeros(
        (len(list_of_token_lists), max_len), dtype=torch.long
    )  # pad id = 0
    for i, toks in enumerate(list_of_token_lists):
        if max_len == 0:
            continue
        ids = [stable_hash(t) for t in toks]
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded, lengths


def _make_iterator(
    ds, batch_size, sort_key, device, sort_within_batch=True, shuffle=True, seed=42
):
    n = len(ds.df)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        bidx = idx[start : start + batch_size]
        bdf = ds.df.iloc[bidx].reset_index(drop=True)

        token_lists = [
            _tokenize_text(x) for x in bdf["comment_text"].astype(str).tolist()
        ]
        if sort_within_batch:
            lens = [len(t) for t in token_lists]
            order = np.argsort(lens)[::-1]
            bdf = bdf.iloc[order].reset_index(drop=True)
            token_lists = [token_lists[i] for i in order]

        text_tensor, lengths = _pad_and_numericalize(token_lists)
        text_tensor = text_tensor.to(device)
        lengths = lengths.to(device)

        batch = SimpleNamespace()
        batch.comment_text = (text_tensor, lengths)

        for ln in _label_names:
            if ln in bdf.columns:
                setattr(
                    batch,
                    ln,
                    torch.tensor(bdf[ln].values, dtype=torch.float32, device=device),
                )

        yield batch


traindl = _make_iterator(
    train,
    batch_size=128,
    sort_key=lambda x: len(x.comment_text),
    device=device,
    sort_within_batch=True,
    shuffle=True,
    seed=42,
)
valdl = _make_iterator(
    val,
    batch_size=1024,
    sort_key=lambda x: len(x.comment_text),
    device=device,
    sort_within_batch=True,
    shuffle=False,
    seed=42,
)


## === cell 4
_fields_map = {name: field for (name, field) in train.fields}
_text_field = _fields_map["comment_text"]

vectors = _text_field.vectors
vectors = vectors.to(device)


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1964255629.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mvectors[0m [0;34m=[0m [0m_text_field[0m[0;34m.[0m[0mvectors[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mvectors[0m [0;34m=[0m [0mvectors[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'str' object has no attribute 'to'

## === cell 5
class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.yFields= ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.x= 'comment_text'
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x)
            y = torch.transpose( torch.stack([getattr(batch, y) for y in self.yFields]),0,1)
            yield (X,y)

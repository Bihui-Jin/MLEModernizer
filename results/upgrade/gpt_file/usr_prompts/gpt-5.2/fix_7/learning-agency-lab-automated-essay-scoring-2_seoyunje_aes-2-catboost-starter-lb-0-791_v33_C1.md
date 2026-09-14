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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

catboost==1.2.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7995683527601928

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import ctypes
import random
import time
import string
import re
from tqdm import tqdm
import pickle
from functools import lru_cache

import pandas as pd, numpy as np
import polars as pl  # kept

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor

from scipy import sparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For GPU T4x2

import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")



## === cell 1
try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        try:
            print(x.head())
        except Exception:
            print(x)




## === cell 2
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-cat/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-cat/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 3
if not os.path.exists(os.path.join(CFG.BASE_PATH, "train.csv")):
    alt_base = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
    if os.path.exists(os.path.join(alt_base, "train.csv")):
        CFG.BASE_PATH = alt_base



## === cell 4
Clean = True


def clean_memory():
    if Clean:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()


clean_memory()




## === cell 5
def _dir_has_models(d, ver, n_folds=10):
    if not d:
        return False
    ok = True
    for i in range(n_folds):
        ok = ok and os.path.exists(os.path.join(d, f"CAT_v{ver}_f{i}.pkl"))
    return ok


def _local_has_models(ver, n_folds=10):
    for i in range(n_folds):
        if not os.path.exists(f"CAT_v{ver}_f{i}.pkl"):
            return False
    return True


if CFG.LOAD_FEATURES_FROM and (not os.path.exists(CFG.LOAD_FEATURES_FROM)):
    CFG.LOAD_FEATURES_FROM = None

if CFG.LOAD_MODELS_FROM and (
    not _dir_has_models(CFG.LOAD_MODELS_FROM, CFG.VER, n_folds=10)
):
    CFG.LOAD_MODELS_FROM = None

if (CFG.LOAD_MODELS_FROM is None) and _local_has_models(CFG.VER, n_folds=10):
    CFG.LOAD_MODELS_FROM = ""  # sentinel meaning "load from local working dir"




## === cell 6
def seed_everything():  # Determinism
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 7
pass



## === cell 8
pass



## === cell 9
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id").reset_index(drop=True)

print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 10
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id").reset_index(drop=True)

print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 11
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData



## === cell 12
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    pass



## === cell 13
pass



## === cell 14
_HTML_RE = re.compile(r"<.*?>")
_AT_RE = re.compile(r"@\w+")
_QUOTE_NUM_RE = re.compile(r"'\d+")
_NUM_RE = re.compile(r"\d+")
_HTTP_RE = re.compile(r"http\w+")
_WS_RE = re.compile(r"\s+")
_BAD_CHARS_RE = re.compile(r'[^\w\s.,;:"' "?!]")
_DOTS_RE = re.compile(r"\.+")
_COMMAS_RE = re.compile(r"\,+")


def removeHTML(x):
    return _HTML_RE.sub(r"", x)  # html -> ''


@lru_cache(maxsize=200_000)
def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)

    x = _AT_RE.sub("", x)

    x = _QUOTE_NUM_RE.sub("", x)
    x = _NUM_RE.sub("", x)

    x = _HTTP_RE.sub("", x)

    x = _WS_RE.sub(" ", x)
    x = _BAD_CHARS_RE.sub("", x)
    x = x.replace("paragraph", "")
    x = _DOTS_RE.sub(".", x)
    x = _COMMAS_RE.sub(",", x)
    x = x.strip()
    return x




## === cell 15
import string as _string


def _simple_tokenize(text: str):
    return [
        t.strip(_string.punctuation)
        for t in text.split()
        if t.strip(_string.punctuation)
    ]




## === cell 16
_TRIPLE_RE = re.compile(r"(.)\1\1")


@lru_cache(maxsize=400_000)
def count_misspelled_words(text):
    toks = _simple_tokenize(text)
    cnt = 0
    for t in toks:
        if not t:
            continue
        if t.isascii() and any("0" <= ch <= "9" for ch in t):
            cnt += 1
            continue
        if len(t) >= 18:
            cnt += 1
            continue
        if _TRIPLE_RE.search(t) is not None:
            cnt += 1
            continue
    return cnt




## === cell 17
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(x: pl.DataFrame) -> pd.DataFrame:
    df = x.select(["essay_id", "paragraph"]).to_pandas()
    df = df.explode("paragraph", ignore_index=True)
    s = df["paragraph"].fillna("").astype(str)

    s = s.map(dataPreprocessing)
    df["paragraph"] = s

    df["paragraph_len"] = s.str.len().astype(np.int64)
    df["paragraph_comma_cnt"] = s.str.count(",").astype(np.int64)

    df["paragraph_misspelled_cnt"] = s.map(
        lambda v: count_misspelled_words(v) if v else 0
    ).astype(np.int64)

    nonempty = s.str.len().to_numpy() > 0
    dot_cnt = s.str.count(r"\.", flags=0).to_numpy()
    sp_cnt = s.str.count(" ").to_numpy()
    df["paragraph_sentence_cnt"] = np.where(nonempty, dot_cnt + 1, 0).astype(np.int64)
    df["paragraph_word_cnt"] = np.where(nonempty, sp_cnt + 1, 0).astype(np.int64)
    return df


def _counts_ge_by_threshold(df, group_key, value_col, thresholds, prefix):
    v = df[value_col].to_numpy()
    out = {group_key: df[group_key].values}
    for t in thresholds:
        out[f"{prefix}{t}_cnt"] = (v >= t).astype(np.int8)
    tmp = pd.DataFrame(out)
    return tmp.groupby(group_key, sort=False).sum().reset_index()


def _counts_le_by_threshold(df, group_key, value_col, thresholds, prefix):
    v = df[value_col].to_numpy()
    out = {group_key: df[group_key].values}
    for t in thresholds:
        out[f"{prefix}{t}_cnt_v2"] = (v <= t).astype(np.int8)
    tmp = pd.DataFrame(out)
    return tmp.groupby(group_key, sort=False).sum().reset_index()


def Paragraph_aggregation(x: pd.DataFrame) -> pd.DataFrame:
    df = x
    gb = df.groupby("essay_id", sort=False)

    out = gb.size().rename("paragraph_cnt").reset_index()

    ge_thresholds = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
    le_thresholds = [100, 200]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "paragraph_len", ge_thresholds, "paragraph_"
        ),
        on="essay_id",
        how="left",
    )
    out = out.merge(
        _counts_le_by_threshold(
            df, "essay_id", "paragraph_len", le_thresholds, "paragraph_"
        ),
        on="essay_id",
        how="left",
    )

    v = df["paragraph_len"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_paragraph_cnt": ((v <= 300) & (v > 100)).astype(np.int8),
                "mid_paragraph_cnt": ((v <= 500) & (v > 300)).astype(np.int8),
                "long_paragraph_cnt": ((v <= 700) & (v > 500)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    sent_thresholds = [2, 4, 6, 8, 10]
    out = out.merge(
        _counts_ge_by_threshold(
            df,
            "essay_id",
            "paragraph_sentence_cnt",
            sent_thresholds,
            "paragraph_sentence_",
        ),
        on="essay_id",
        how="left",
    )
    sv = df["paragraph_sentence_cnt"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_paragraph_sentence_cnt": ((sv <= 4) & (sv > 2)).astype(np.int8),
                "mid_paragraph_sentence_cnt": ((sv <= 8) & (sv > 4)).astype(np.int8),
                "long_paragraph_sentence_cnt": ((sv <= 10) & (sv > 8)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    word_thresholds = [20, 40, 60, 90, 120]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "paragraph_word_cnt", word_thresholds, "paragraph_word_"
        ),
        on="essay_id",
        how="left",
    )
    wv = df["paragraph_word_cnt"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_paragraph_word_cnt": ((wv <= 40) & (wv > 20)).astype(np.int8),
                "mid_paragraph_word_cnt": ((wv <= 90) & (wv > 40)).astype(np.int8),
                "long_paragraph_word_cnt": ((wv <= 120) & (wv > 90)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    comma_thresholds = [1, 2, 3, 4, 5]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "paragraph_comma_cnt", comma_thresholds, "paragraph_comma_"
        ),
        on="essay_id",
        how="left",
    )

    miss_thresholds = [4, 8, 12, 16]
    miss_le_thresholds = [2, 4]
    out = out.merge(
        _counts_ge_by_threshold(
            df,
            "essay_id",
            "paragraph_misspelled_cnt",
            miss_thresholds,
            "paragraph_misspelled_",
        ),
        on="essay_id",
        how="left",
    )
    out = out.merge(
        _counts_le_by_threshold(
            df,
            "essay_id",
            "paragraph_misspelled_cnt",
            miss_le_thresholds,
            "paragraph_misspelled_",
        ),
        on="essay_id",
        how="left",
    )
    mv = df["paragraph_misspelled_cnt"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_paragraph_misspelled_cnt": ((mv <= 8) & (mv > 4)).astype(
                    np.int8
                ),
                "mid_paragraph_misspelled_cnt": ((mv <= 12) & (mv > 8)).astype(np.int8),
                "long_paragraph_misspelled_cnt": ((mv <= 16) & (mv > 12)).astype(
                    np.int8
                ),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    agg = gb[paragraph_features].agg(["max", "mean", "min", "std", "sum"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    q1 = gb[paragraph_features].quantile(0.25).add_suffix("_q1").reset_index()
    q3 = gb[paragraph_features].quantile(0.75).add_suffix("_q3").reset_index()
    out = (
        out.merge(agg, on="essay_id", how="left")
        .merge(q1, on="essay_id", how="left")
        .merge(q3, on="essay_id", how="left")
    )

    std_cols = [c for c in out.columns if c.endswith("_std")]
    out[std_cols] = out[std_cols].fillna(0)

    out = out.sort_values("essay_id").reset_index(drop=True)
    return out




## === cell 18
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pd.DataFrame:
    df = x.to_pandas()

    if "full_text_pre" in df.columns:
        s = df["full_text_pre"].fillna("").astype(str)
    else:
        s = df["full_text"].fillna("").astype(str).map(dataPreprocessing)

    df2 = pd.DataFrame(
        {"essay_id": df["essay_id"].values, "sentence": s.str.split(".")}
    )
    df2 = df2.explode("sentence", ignore_index=True)
    sent = df2["sentence"].fillna("").astype(str)

    df2["sentence_len"] = sent.str.len().astype(np.int64)
    df2["only_sentence_len"] = (
        sent.str.replace(" ", "", regex=False).str.len().astype(np.int64)
    )

    df2 = df2[df2["sentence_len"] > 3].reset_index(drop=True)

    nonempty = sent.str.len().to_numpy() > 0
    sp_cnt = sent.str.count(" ").to_numpy()
    df2["sentence_word_cnt"] = np.where(nonempty, sp_cnt + 1, 0).astype(np.int64)
    return df2


def Sentence_aggregation(x: pd.DataFrame) -> pd.DataFrame:
    df = x
    gb = df.groupby("essay_id", sort=False)
    out = gb.size().rename("sentence_cnt").reset_index()

    ge_thresholds = [40, 60, 70, 80, 100, 120, 140]
    le_thresholds = [10, 20, 30]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "sentence_len", ge_thresholds, "sentence_"
        ),
        on="essay_id",
        how="left",
    )
    out = out.merge(
        _counts_le_by_threshold(
            df, "essay_id", "sentence_len", le_thresholds, "sentence_"
        ),
        on="essay_id",
        how="left",
    )

    v = df["sentence_len"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_sentence_cnt": ((v <= 70) & (v > 40)).astype(np.int8),
                "mid_sentence_cnt": ((v <= 100) & (v > 70)).astype(np.int8),
                "long_sentence_cnt": ((v <= 140) & (v > 100)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    only_thresholds = [40, 60, 80, 100, 120]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "only_sentence_len", only_thresholds, "only_sentence_"
        ),
        on="essay_id",
        how="left",
    )

    ov = df["only_sentence_len"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_only_sentence_cnt": ((ov <= 60) & (ov > 40)).astype(np.int8),
                "mid_only_sentence_cnt": ((ov <= 100) & (ov > 60)).astype(np.int8),
                "long_only_sentence_cnt": ((ov <= 120) & (ov > 100)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    word_thresholds = [10, 15, 20, 25]
    out = out.merge(
        _counts_ge_by_threshold(
            df, "essay_id", "sentence_word_cnt", word_thresholds, "sentence_word_"
        ),
        on="essay_id",
        how="left",
    )
    wv = df["sentence_word_cnt"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_sentence_word_cnt": ((wv <= 15) & (wv > 10)).astype(np.int8),
                "mid_sentence_word_cnt": ((wv <= 20) & (wv > 15)).astype(np.int8),
                "long_sentence_word_cnt": ((wv <= 25) & (wv > 20)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    agg = gb[sentence_features].agg(["max", "mean", "min", "std", "sum"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    q1 = gb[sentence_features].quantile(0.25).add_suffix("_q1").reset_index()
    q3 = gb[sentence_features].quantile(0.75).add_suffix("_q3").reset_index()
    out = (
        out.merge(agg, on="essay_id", how="left")
        .merge(q1, on="essay_id", how="left")
        .merge(q3, on="essay_id", how="left")
    )

    std_cols = [c for c in out.columns if c.endswith("_std")]
    out[std_cols] = out[std_cols].fillna(0)

    denom = out["sentence_cnt"].replace(0, np.nan)
    for i in ge_thresholds:
        out[f"sentence_{i}_cnt_ratio"] = (
            (out[f"sentence_{i}_cnt"] / denom).fillna(0.0).values
        )
    out["short_sentence_cnt_ratio"] = (
        (out["short_sentence_cnt"] / denom).fillna(0.0).values
    )
    out["mid_sentence_cnt_ratio"] = (out["mid_sentence_cnt"] / denom).fillna(0.0).values
    out["long_sentence_cnt_ratio"] = (
        (out["long_sentence_cnt"] / denom).fillna(0.0).values
    )

    out = out.sort_values("essay_id").reset_index(drop=True)
    return out




## === cell 19
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame) -> pd.DataFrame:
    df = x.to_pandas()
    if "full_text_pre" in df.columns:
        s = df["full_text_pre"].fillna("").astype(str)
    else:
        s = df["full_text"].fillna("").astype(str).map(dataPreprocessing)

    df2 = pd.DataFrame({"essay_id": df["essay_id"].values, "word": s.str.split(" ")})
    df2 = df2.explode("word", ignore_index=True)
    w = df2["word"].fillna("").astype(str)

    df2["word_len"] = w.str.len().astype(np.int64)
    df2 = df2[df2["word_len"] > 0].reset_index(drop=True)
    return df2


def Word_aggregation(x: pd.DataFrame) -> pd.DataFrame:
    df = x
    gb = df.groupby("essay_id", sort=False)

    out = gb.size().rename("word_cnt").reset_index()

    ge_thresholds = [3, 4, 5, 6, 7, 8, 10]
    le_thresholds = [1, 2, 3]
    out = out.merge(
        _counts_ge_by_threshold(df, "essay_id", "word_len", ge_thresholds, "word_"),
        on="essay_id",
        how="left",
    )
    out = out.merge(
        _counts_le_by_threshold(df, "essay_id", "word_len", le_thresholds, "word_"),
        on="essay_id",
        how="left",
    )

    v = df["word_len"].to_numpy()
    tmp = (
        pd.DataFrame(
            {
                "essay_id": df["essay_id"].values,
                "short_word_cnt": ((v <= 4) & (v > 2)).astype(np.int8),
                "mid_word_cnt": ((v <= 6) & (v > 4)).astype(np.int8),
                "long_word_cnt": ((v <= 10) & (v > 6)).astype(np.int8),
            }
        )
        .groupby("essay_id", sort=False)
        .sum()
        .reset_index()
    )
    out = out.merge(tmp, on="essay_id", how="left")

    agg = gb[word_features].agg(["max", "mean", "min", "std", "sum"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    q1 = gb[word_features].quantile(0.25).add_suffix("_q1").reset_index()
    q3 = gb[word_features].quantile(0.75).add_suffix("_q3").reset_index()
    out = (
        out.merge(agg, on="essay_id", how="left")
        .merge(q1, on="essay_id", how="left")
        .merge(q3, on="essay_id", how="left")
    )

    std_cols = [c for c in out.columns if c.endswith("_std")]
    out[std_cols] = out[std_cols].fillna(0)

    denom_word = out["word_cnt"].replace(0, np.nan)
    for i in ge_thresholds:
        out[f"word_{i}_cnt_ratio"] = (
            (out[f"word_{i}_cnt"] / denom_word).fillna(0.0).values
        )
    for i in le_thresholds:
        out[f"word_{i}_cnt_v2_ratio"] = (
            (out[f"word_{i}_cnt_v2"] / denom_word).fillna(0.0).values
        )

    denom2 = out["word_2_cnt_v2"].replace(0, np.nan)
    denom3 = out["word_3_cnt_v2"].replace(0, np.nan)
    for i in ge_thresholds:
        out[f"word_{i}_pre2_ratio"] = (out[f"word_{i}_cnt"] / denom2).fillna(0.0).values
        out[f"word_{i}_pre3_ratio"] = (out[f"word_{i}_cnt"] / denom3).fillna(0.0).values

    for i in le_thresholds:
        denom_i = out[f"word_{i}_cnt_v2"].replace(0, np.nan)
        out[f"short_word_ratio_{i}"] = (
            (out["short_word_cnt"] / denom_i).fillna(0.0).values
        )
        out[f"mid_word_ratio_{i}"] = (out["mid_word_cnt"] / denom_i).fillna(0.0).values
        out[f"long_word_ratio_{i}"] = (
            (out["long_word_cnt"] / denom_i).fillna(0.0).values
        )

    out = out.sort_values("essay_id").reset_index(drop=True)
    return out




## === cell 20
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,  # Term Frequency Log Scaling
)

df_train["full_text_pre"] = (
    df_train["full_text"].fillna("").astype(str).map(dataPreprocessing)
)
train_text_pre = df_train["full_text_pre"].tolist()

train_tfid = vectorizer.fit_transform(train_text_pre)  # CSR sparse matrix
train_ids = df_train["essay_id"].values



## === cell 21
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 4),
    min_df=0.10,
    max_df=0.85,
)

train_cnt = vectorizer_cnt.fit_transform(train_text_pre)  # CSR sparse matrix



## === cell 22
if CFG.LOAD_FEATURES_FROM is None:
    train_pl = pl.from_pandas(
        df_train[["essay_id", "full_text_pre", "full_text"]]
    ).with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))

    train_feats1 = Paragraph_Features(train_pl.select(["essay_id", "paragraph"]))
    train_feats1 = Paragraph_aggregation(train_feats1)

    train_feats2 = Sentence_Features(train_pl.select(["essay_id", "full_text_pre"]))
    train_feats2 = Sentence_aggregation(train_feats2)

    train_feats3 = Word_Features(train_pl.select(["essay_id", "full_text_pre"]))
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")

    train_feats["score"] = df_train["score"].values
else:
    train_feats = None



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4043070865.py in <cell line: 0>()
      7     train_feats1 = Paragraph_aggregation(train_feats1)
      8 
----> 9     train_feats2 = Sentence_Features(train_pl.select(["essay_id", "full_text_pre"]))
     10     train_feats2 = Sentence_aggregation(train_feats2)
     11 

/tmp/ipykernel_55/183754950.py in Sentence_Features(x)
     26     nonempty = sent.str.len().to_numpy() > 0
     27     sp_cnt = sent.str.count(" ").to_numpy()
---> 28     df2["sentence_word_cnt"] = np.where(nonempty, sp_cnt + 1, 0).astype(np.int64)
     29     return df2
     30 

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

ValueError: Length of values (307173) does not match length of index (290835)

## === cell 23
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/152654945.py in <cell line: 0>()
      1 if CFG.LOAD_FEATURES_FROM is None:
      2     print("Save train_feats.csv")
----> 3     train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      4 else:
      5     print("Load train_feats.csv")

NameError: name 'train_feats' is not defined

## === cell 24
display(train_feats.head())



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1357160389.py in <cell line: 0>()
----> 1 display(train_feats.head())
      2 

NameError: name 'train_feats' is not defined

## === cell 25
clean_memory()



## === cell 26
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score




## === cell 27
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - a) ** 2 + b)
    df_ = preds - labels
    dg = preds - a
    grad = (df_ / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.948
b = 1.092



## === cell 28
from sklearn.model_selection import train_test_split
import optuna

import catboost
from catboost import CatBoostRegressor, Pool

print("Catboost Version: ", catboost.__version__)



## === cell 29
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()

ENGINEERED_FEATURES = [
    c
    for c in train_feats.columns
    if c not in categorical_columns + ["score", "essay_id"]
]
TARGET = "score"

ENGINEERED_FEATURES = list(ENGINEERED_FEATURES)

X_num = (
    train_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False)
)
np.clip(X_num, 0, 10000, out=X_num)
X_num_csr = sparse.csr_matrix(X_num)

X_all = sparse.hstack([X_num_csr, train_tfid, train_cnt], format="csr")

FEATURES = (
    ENGINEERED_FEATURES
    + [f"tfidf_{i}" for i in range(train_tfid.shape[1])]
    + [f"cnt_{i}" for i in range(train_cnt.shape[1])]
)
print(
    "Num features:",
    len(FEATURES),
    " (engineered:",
    len(ENGINEERED_FEATURES),
    "tfidf:",
    train_tfid.shape[1],
    "cnt:",
    train_cnt.shape[1],
    ")",
)
y_all = train_feats[TARGET].to_numpy()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1944566066.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 
      5 ENGINEERED_FEATURES = [

NameError: name 'train_feats' is not defined

## === cell 30
pass



## === cell 31
"""
def cat_objective(trial):

    params = { 
          'verbose'      : 0,
          'random_state' : CFG.SEED, 
          'loss_function' : 'MultiClass', 
          'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.5), 
          'depth' : trial.suggest_int('depth', 5, 10),
    }

    train_x, valid_x, train_y, valid_y = train_test_split(train_feats[FEATURES], train_feats[TARGET], test_size=0.2, random_state=CFG.SEED)
    train_pool = Pool(
          data = train_x,
          label = train_y
    )    

    valid_pool = Pool(
          data = valid_x,
          label = valid_y
    )

    model  = CatBoostClassifier(**params)

    model.fit(train_pool,
          eval_set = valid_pool,  
           )
    oof = model.predict(valid_pool)
    cv = cohen_kappa_score(valid_y, oof, weights="quadratic")

    
    return cv """


## === cell 32
"""
study = optuna.create_study(direction='minimize', study_name='Classification') 
study.optimize(cat_objective, n_trials=10, show_progress_bar=True)
"""


## === cell 33
pass



## === cell 34
pass




## === cell 35
def catboost():
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(X_all, y_all)):

        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=5,
            subsample=0.8,
            l2_leaf_reg=1,
            task_type="CPU",
            thread_count=int(os.environ.get("OMP_NUM_THREADS", "2")),
            objective="RMSE",
            eval_metric="RMSE",
            random_state=CFG.SEED,
            allow_writing_files=False,
        )

        train_pool = Pool(
            data=X_all[train_index],
            label=y_all[train_index],
        )

        valid_pool = Pool(
            data=X_all[valid_index],
            label=y_all[valid_index],
        )

        model.fit(
            train_pool, verbose=100, eval_set=valid_pool, early_stopping_rounds=75
        )

        pickle.dump(model, open(f"CAT_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(valid_pool)
        all_oof.append(oof)
        all_true.append(y_all[valid_index])

        del train_pool, valid_pool, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = cohen_kappa_score(true[0], oof[0].clip(1, 6).round(), weights="quadratic")
    print("CV Score for Low Catboost = ", cv)
    cm = confusion_matrix(
        true[0], oof[0].clip(1, 6).round(), labels=[x for x in range(1, 7)]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    disp.plot()
    plt.show()




## === cell 36
if CFG.LOAD_MODELS_FROM is None:
    print("Training CATBoost")
    catboost()
else:
    if CFG.LOAD_MODELS_FROM == "":
        print("Loading local models from working directory (skip training).")
    else:
        print("Loading models from:", CFG.LOAD_MODELS_FROM)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4001144474.py in <cell line: 0>()
      1 if CFG.LOAD_MODELS_FROM is None:
      2     print("Training CATBoost")
----> 3     catboost()
      4 else:
      5     if CFG.LOAD_MODELS_FROM == "":

/tmp/ipykernel_55/922599691.py in catboost()
      4 
      5     skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
----> 6     for i, (train_index, valid_index) in enumerate(skf.split(X_all, y_all)):
      7 
      8         print("#" * 25)

NameError: name 'X_all' is not defined

## === cell 37
model_path = None
if CFG.LOAD_MODELS_FROM:
    p = os.path.join(CFG.LOAD_MODELS_FROM, f"CAT_v{CFG.VER}_f0.pkl")
    if os.path.exists(p):
        model_path = p
else:
    p = f"CAT_v{CFG.VER}_f0.pkl"
    if os.path.exists(p):
        model_path = p

if model_path is not None:
    model = pickle.load(open(model_path, "rb"))

    df_importance = pd.DataFrame(
        {
            "features_name": FEATURES,
            "importance": model.feature_importances_,
        }
    )
    df_importance = df_importance.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(
        data=df_importance.head(30),
        x="features_name",
        height="importance",
        color="pink",
        edgecolor="black",
    )
    plt.title("Distribution of Feature Importance of Catboost")
    plt.xticks(rotation=90)
    plt.show()
else:
    print("Skipping feature importance plot (model file not found).")



## === cell 38
df_test["full_text_pre"] = (
    df_test["full_text"].fillna("").astype(str).map(dataPreprocessing)
)
test_text_pre = df_test["full_text_pre"].tolist()

test_tfid = vectorizer.transform(test_text_pre)  # CSR sparse matrix



## === cell 39
test_cnt = vectorizer_cnt.transform(test_text_pre)  # CSR sparse matrix



## === cell 40
t0 = time.time()
test_pl = pl.from_pandas(
    df_test[["essay_id", "full_text_pre", "full_text"]]
).with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))

test_feats1 = Paragraph_Features(test_pl.select(["essay_id", "paragraph"]))
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test_pl.select(["essay_id", "full_text_pre"]))
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test_pl.select(["essay_id", "full_text_pre"]))
test_feats3 = Word_aggregation(test_feats3)
print(f"Test feature extraction took {time.time()-t0:.1f}s")



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2320899864.py in <cell line: 0>()
      6 test_feats1 = Paragraph_Features(test_pl.select(["essay_id", "paragraph"]))
      7 test_feats1 = Paragraph_aggregation(test_feats1)
----> 8 test_feats2 = Sentence_Features(test_pl.select(["essay_id", "full_text_pre"]))
      9 test_feats2 = Sentence_aggregation(test_feats2)
     10 test_feats3 = Word_Features(test_pl.select(["essay_id", "full_text_pre"]))

/tmp/ipykernel_55/183754950.py in Sentence_Features(x)
     26     nonempty = sent.str.len().to_numpy() > 0
     27     sp_cnt = sent.str.count(" ").to_numpy()
---> 28     df2["sentence_word_cnt"] = np.where(nonempty, sp_cnt + 1, 0).astype(np.int64)
     29     return df2
     30 

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

ValueError: Length of values (33951) does not match length of index (32112)

## === cell 41
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
print("Shape of test_feats (engineered only):", test_feats.shape)
display(test_feats.head())

X_test_num = (
    test_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False)
)
np.clip(X_test_num, 0, 10000, out=X_test_num)
X_test_num_csr = sparse.csr_matrix(X_test_num)
X_test_all = sparse.hstack([X_test_num_csr, test_tfid, test_cnt], format="csr")

preds = []
models = []
for i in range(10):
    print(f"Fold {i+1}")
    if CFG.LOAD_MODELS_FROM:
        model = pickle.load(
            open(os.path.join(CFG.LOAD_MODELS_FROM, f"CAT_v{CFG.VER}_f{i}.pkl"), "rb")
        )
    else:
        model = pickle.load(open(f"CAT_v{CFG.VER}_f{i}.pkl", "rb"))
    models.append(model)

for i, model in enumerate(models):
    pred_i = model.predict(X_test_all)
    preds.append(pred_i)

pred = np.mean(preds, axis=0)

sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub[TARGET] = np.clip(pred, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Wrote submission.csv:", os.path.exists("submission.csv"))

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3941764677.py in <cell line: 0>()
----> 1 test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
      2 test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
      3 print("Shape of test_feats (engineered only):", test_feats.shape)
      4 display(test_feats.head())
      5 

NameError: name 'test_feats2' is not defined

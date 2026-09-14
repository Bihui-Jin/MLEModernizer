# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For GPU T4x2

import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

try:
    pl.Config.set_tbl_rows(20)
    pl.Config.set_verbose(False)
except Exception:
    pass



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
candidate_bases = [
    CFG.BASE_PATH,
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
    "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
    "/kaggle/input/",
    "/kaggle/data/",
]
for base in candidate_bases:
    if (
        base
        and os.path.exists(os.path.join(base, "train.csv"))
        and os.path.exists(os.path.join(base, "test.csv"))
    ):
        CFG.BASE_PATH = base if base.endswith("/") else base + "/"
        break



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
df_train = pd.read_csv(
    CFG.BASE_PATH + "train.csv",
    dtype={"essay_id": "string", "full_text": "string", "score": "int8"},
)
df_train = df_train.sort_values(by="essay_id").reset_index(drop=True)

print("Base path:", CFG.BASE_PATH)
print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 10
df_test = pd.read_csv(
    CFG.BASE_PATH + "test.csv",
    dtype={"essay_id": "string", "full_text": "string"},
)
df_test = df_test.sort_values(by="essay_id").reset_index(drop=True)

print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 11
pass



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


@lru_cache(maxsize=120_000)
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


@lru_cache(maxsize=200_000)
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


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    df = (
        x.select(["essay_id", "paragraph"])
        .explode("paragraph")
        .with_columns(pl.col("paragraph").fill_null("").cast(pl.Utf8))
    )

    df = df.with_columns(
        paragraph_len=pl.col("paragraph").str.len_chars().cast(pl.Int64),
        paragraph_comma_cnt=pl.col("paragraph").str.count_matches(",").cast(pl.Int64),
    )

    paras = df.get_column("paragraph").to_list()
    miss = [count_misspelled_words(v) if v else 0 for v in paras]
    df = df.with_columns(pl.Series("paragraph_misspelled_cnt", miss, dtype=pl.Int64))

    nonempty = pl.col("paragraph").str.len_chars() > 0
    df = df.with_columns(
        paragraph_sentence_cnt=pl.when(nonempty)
        .then(pl.col("paragraph").str.count_matches(r"\.") + 1)
        .otherwise(0)
        .cast(pl.Int64),
        paragraph_word_cnt=pl.when(nonempty)
        .then(pl.col("paragraph").str.count_matches(" ") + 1)
        .otherwise(0)
        .cast(pl.Int64),
    )
    return df


def _pl_counts_ge_by_threshold(
    df: pl.DataFrame, value_col: str, thresholds, prefix: str
) -> list[pl.Expr]:
    return [
        (pl.col(value_col) >= t).cast(pl.Int16).sum().alias(f"{prefix}{t}_cnt")
        for t in thresholds
    ]


def _pl_counts_le_by_threshold(
    df: pl.DataFrame, value_col: str, thresholds, prefix: str
) -> list[pl.Expr]:
    return [
        (pl.col(value_col) <= t).cast(pl.Int16).sum().alias(f"{prefix}{t}_cnt_v2")
        for t in thresholds
    ]


def Paragraph_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    df = x

    ge_thresholds = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
    le_thresholds = [100, 200]
    sent_thresholds = [2, 4, 6, 8, 10]
    word_thresholds = [20, 40, 60, 90, 120]
    comma_thresholds = [1, 2, 3, 4, 5]
    miss_thresholds = [4, 8, 12, 16]
    miss_le_thresholds = [2, 4]

    agg_exprs = [
        pl.len().alias("paragraph_cnt"),
        *_pl_counts_ge_by_threshold(df, "paragraph_len", ge_thresholds, "paragraph_"),
        *_pl_counts_le_by_threshold(df, "paragraph_len", le_thresholds, "paragraph_"),
        ((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
        .cast(pl.Int16)
        .sum()
        .alias("short_paragraph_cnt"),
        ((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
        .cast(pl.Int16)
        .sum()
        .alias("mid_paragraph_cnt"),
        ((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
        .cast(pl.Int16)
        .sum()
        .alias("long_paragraph_cnt"),
        *_pl_counts_ge_by_threshold(
            df, "paragraph_sentence_cnt", sent_thresholds, "paragraph_sentence_"
        ),
        (
            (pl.col("paragraph_sentence_cnt") <= 4)
            & (pl.col("paragraph_sentence_cnt") > 2)
        )
        .cast(pl.Int16)
        .sum()
        .alias("short_paragraph_sentence_cnt"),
        (
            (pl.col("paragraph_sentence_cnt") <= 8)
            & (pl.col("paragraph_sentence_cnt") > 4)
        )
        .cast(pl.Int16)
        .sum()
        .alias("mid_paragraph_sentence_cnt"),
        (
            (pl.col("paragraph_sentence_cnt") <= 10)
            & (pl.col("paragraph_sentence_cnt") > 8)
        )
        .cast(pl.Int16)
        .sum()
        .alias("long_paragraph_sentence_cnt"),
        *_pl_counts_ge_by_threshold(
            df, "paragraph_word_cnt", word_thresholds, "paragraph_word_"
        ),
        ((pl.col("paragraph_word_cnt") <= 40) & (pl.col("paragraph_word_cnt") > 20))
        .cast(pl.Int16)
        .sum()
        .alias("short_paragraph_word_cnt"),
        ((pl.col("paragraph_word_cnt") <= 90) & (pl.col("paragraph_word_cnt") > 40))
        .cast(pl.Int16)
        .sum()
        .alias("mid_paragraph_word_cnt"),
        ((pl.col("paragraph_word_cnt") <= 120) & (pl.col("paragraph_word_cnt") > 90))
        .cast(pl.Int16)
        .sum()
        .alias("long_paragraph_word_cnt"),
        *_pl_counts_ge_by_threshold(
            df, "paragraph_comma_cnt", comma_thresholds, "paragraph_comma_"
        ),
        *_pl_counts_ge_by_threshold(
            df, "paragraph_misspelled_cnt", miss_thresholds, "paragraph_misspelled_"
        ),
        *_pl_counts_le_by_threshold(
            df, "paragraph_misspelled_cnt", miss_le_thresholds, "paragraph_misspelled_"
        ),
        (
            (pl.col("paragraph_misspelled_cnt") <= 8)
            & (pl.col("paragraph_misspelled_cnt") > 4)
        )
        .cast(pl.Int16)
        .sum()
        .alias("short_paragraph_misspelled_cnt"),
        (
            (pl.col("paragraph_misspelled_cnt") <= 12)
            & (pl.col("paragraph_misspelled_cnt") > 8)
        )
        .cast(pl.Int16)
        .sum()
        .alias("mid_paragraph_misspelled_cnt"),
        (
            (pl.col("paragraph_misspelled_cnt") <= 16)
            & (pl.col("paragraph_misspelled_cnt") > 12)
        )
        .cast(pl.Int16)
        .sum()
        .alias("long_paragraph_misspelled_cnt"),
    ]

    for c in paragraph_features:
        agg_exprs.extend(
            [
                pl.col(c).max().alias(f"{c}_max"),
                pl.col(c).mean().alias(f"{c}_mean"),
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).std(ddof=1).alias(f"{c}_std"),
                pl.col(c).sum().alias(f"{c}_sum"),
                pl.col(c).quantile(0.25, interpolation="linear").alias(f"{c}_q1"),
                pl.col(c).quantile(0.75, interpolation="linear").alias(f"{c}_q3"),
            ]
        )

    out = df.group_by("essay_id", maintain_order=True).agg(agg_exprs)
    out = out.with_columns(
        [pl.col(c).fill_null(0.0).alias(c) for c in out.columns if c.endswith("_std")]
    )
    return out.sort("essay_id").to_pandas()




## === cell 18
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    df = x
    s = pl.col("full_text_pre").fill_null("").cast(pl.Utf8)

    df2 = (
        df.select(
            pl.col("essay_id"),
            s.str.split(".").alias("sentence"),
        )
        .explode("sentence")
        .with_columns(pl.col("sentence").fill_null("").cast(pl.Utf8))
    )

    df2 = df2.with_columns(
        sentence_len=pl.col("sentence").str.len_chars().cast(pl.Int64),
        only_sentence_len=pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .cast(pl.Int64),
    ).filter(pl.col("sentence_len") > 3)

    nonempty = pl.col("sentence").str.len_chars() > 0
    df2 = df2.with_columns(
        sentence_word_cnt=pl.when(nonempty)
        .then(pl.col("sentence").str.count_matches(" ") + 1)
        .otherwise(0)
        .cast(pl.Int64)
    )
    return df2


def Sentence_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    df = x

    ge_thresholds = [40, 60, 70, 80, 100, 120, 140]
    le_thresholds = [10, 20, 30]
    only_thresholds = [40, 60, 80, 100, 120]
    word_thresholds = [10, 15, 20, 25]

    agg_exprs = [
        pl.len().alias("sentence_cnt"),
        *_pl_counts_ge_by_threshold(df, "sentence_len", ge_thresholds, "sentence_"),
        *_pl_counts_le_by_threshold(df, "sentence_len", le_thresholds, "sentence_"),
        ((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
        .cast(pl.Int16)
        .sum()
        .alias("short_sentence_cnt"),
        ((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
        .cast(pl.Int16)
        .sum()
        .alias("mid_sentence_cnt"),
        ((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
        .cast(pl.Int16)
        .sum()
        .alias("long_sentence_cnt"),
        *_pl_counts_ge_by_threshold(
            df, "only_sentence_len", only_thresholds, "only_sentence_"
        ),
        ((pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40))
        .cast(pl.Int16)
        .sum()
        .alias("short_only_sentence_cnt"),
        ((pl.col("only_sentence_len") <= 100) & (pl.col("only_sentence_len") > 60))
        .cast(pl.Int16)
        .sum()
        .alias("mid_only_sentence_cnt"),
        ((pl.col("only_sentence_len") <= 120) & (pl.col("only_sentence_len") > 100))
        .cast(pl.Int16)
        .sum()
        .alias("long_only_sentence_cnt"),
        *_pl_counts_ge_by_threshold(
            df, "sentence_word_cnt", word_thresholds, "sentence_word_"
        ),
        ((pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10))
        .cast(pl.Int16)
        .sum()
        .alias("short_sentence_word_cnt"),
        ((pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15))
        .cast(pl.Int16)
        .sum()
        .alias("mid_sentence_word_cnt"),
        ((pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20))
        .cast(pl.Int16)
        .sum()
        .alias("long_sentence_word_cnt"),
    ]

    for c in sentence_features:
        agg_exprs.extend(
            [
                pl.col(c).max().alias(f"{c}_max"),
                pl.col(c).mean().alias(f"{c}_mean"),
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).std(ddof=1).alias(f"{c}_std"),
                pl.col(c).sum().alias(f"{c}_sum"),
                pl.col(c).quantile(0.25, interpolation="linear").alias(f"{c}_q1"),
                pl.col(c).quantile(0.75, interpolation="linear").alias(f"{c}_q3"),
            ]
        )

    out = df.group_by("essay_id", maintain_order=True).agg(agg_exprs)
    std_cols = [c for c in out.columns if c.endswith("_std")]
    if std_cols:
        out = out.with_columns([pl.col(c).fill_null(0.0).alias(c) for c in std_cols])

    out_pd = out.sort("essay_id").to_pandas()

    denom = out_pd["sentence_cnt"].replace(0, np.nan)
    for i in ge_thresholds:
        out_pd[f"sentence_{i}_cnt_ratio"] = (
            (out_pd[f"sentence_{i}_cnt"] / denom).fillna(0.0).values
        )
    out_pd["short_sentence_cnt_ratio"] = (
        (out_pd["short_sentence_cnt"] / denom).fillna(0.0).values
    )
    out_pd["mid_sentence_cnt_ratio"] = (
        (out_pd["mid_sentence_cnt"] / denom).fillna(0.0).values
    )
    out_pd["long_sentence_cnt_ratio"] = (
        (out_pd["long_sentence_cnt"] / denom).fillna(0.0).values
    )

    return out_pd.sort_values("essay_id").reset_index(drop=True)




## === cell 19
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    df = x
    s = pl.col("full_text_pre").fill_null("").cast(pl.Utf8)

    df2 = (
        df.select(
            pl.col("essay_id"),
            s.str.split(" ").alias("word"),
        )
        .explode("word")
        .with_columns(pl.col("word").fill_null("").cast(pl.Utf8))
    )
    df2 = df2.with_columns(
        word_len=pl.col("word").str.len_chars().cast(pl.Int64)
    ).filter(pl.col("word_len") > 0)
    return df2


def Word_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    df = x

    ge_thresholds = [3, 4, 5, 6, 7, 8, 10]
    le_thresholds = [1, 2, 3]

    agg_exprs = [
        pl.len().alias("word_cnt"),
        *_pl_counts_ge_by_threshold(df, "word_len", ge_thresholds, "word_"),
        *_pl_counts_le_by_threshold(df, "word_len", le_thresholds, "word_"),
        ((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
        .cast(pl.Int16)
        .sum()
        .alias("short_word_cnt"),
        ((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
        .cast(pl.Int16)
        .sum()
        .alias("mid_word_cnt"),
        ((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
        .cast(pl.Int16)
        .sum()
        .alias("long_word_cnt"),
    ]

    for c in word_features:
        agg_exprs.extend(
            [
                pl.col(c).max().alias(f"{c}_max"),
                pl.col(c).mean().alias(f"{c}_mean"),
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).std(ddof=1).alias(f"{c}_std"),
                pl.col(c).sum().alias(f"{c}_sum"),
                pl.col(c).quantile(0.25, interpolation="linear").alias(f"{c}_q1"),
                pl.col(c).quantile(0.75, interpolation="linear").alias(f"{c}_q3"),
            ]
        )

    out = df.group_by("essay_id", maintain_order=True).agg(agg_exprs)
    std_cols = [c for c in out.columns if c.endswith("_std")]
    if std_cols:
        out = out.with_columns([pl.col(c).fill_null(0.0).alias(c) for c in std_cols])

    out_pd = out.sort("essay_id").to_pandas()

    denom_word = out_pd["word_cnt"].replace(0, np.nan)
    for i in ge_thresholds:
        out_pd[f"word_{i}_cnt_ratio"] = (
            (out_pd[f"word_{i}_cnt"] / denom_word).fillna(0.0).values
        )
    for i in le_thresholds:
        out_pd[f"word_{i}_cnt_v2_ratio"] = (
            (out_pd[f"word_{i}_cnt_v2"] / denom_word).fillna(0.0).values
        )

    denom2 = out_pd["word_2_cnt_v2"].replace(0, np.nan)
    denom3 = out_pd["word_3_cnt_v2"].replace(0, np.nan)
    for i in ge_thresholds:
        out_pd[f"word_{i}_pre2_ratio"] = (
            (out_pd[f"word_{i}_cnt"] / denom2).fillna(0.0).values
        )
        out_pd[f"word_{i}_pre3_ratio"] = (
            (out_pd[f"word_{i}_cnt"] / denom3).fillna(0.0).values
        )

    for i in le_thresholds:
        denom_i = out_pd[f"word_{i}_cnt_v2"].replace(0, np.nan)
        out_pd[f"short_word_ratio_{i}"] = (
            (out_pd["short_word_cnt"] / denom_i).fillna(0.0).values
        )
        out_pd[f"mid_word_ratio_{i}"] = (
            (out_pd["mid_word_cnt"] / denom_i).fillna(0.0).values
        )
        out_pd[f"long_word_ratio_{i}"] = (
            (out_pd["long_word_cnt"] / denom_i).fillna(0.0).values
        )

    return out_pd.sort_values("essay_id").reset_index(drop=True)




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
train_text_pre = df_train["full_text_pre"]  # keep as Series

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
    train_pl = pl.from_pandas(df_train[["essay_id", "full_text_pre"]]).with_columns(
        pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph")
    )

    t0 = time.time()
    train_feats1 = Paragraph_Features(train_pl.select(["essay_id", "paragraph"]))
    train_feats1 = Paragraph_aggregation(train_feats1)

    train_feats2 = Sentence_Features(train_pl.select(["essay_id", "full_text_pre"]))
    train_feats2 = Sentence_aggregation(train_feats2)

    train_feats3 = Word_Features(train_pl.select(["essay_id", "full_text_pre"]))
    train_feats3 = Word_aggregation(train_feats3)
    print(f"Train feature extraction took {time.time()-t0:.1f}s")

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = None



## === cell 23
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## === cell 24
display(train_feats.head())



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
X_num_csr = sparse.csr_matrix(X_num, dtype=np.float32)

train_tfid = train_tfid.astype(np.float32)
train_cnt = train_cnt.astype(np.float32)

X_all = sparse.hstack(
    [X_num_csr, train_tfid, train_cnt], format="csr", dtype=np.float32
)

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

        model.fit(train_pool, verbose=False, eval_set=valid_pool)

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
    print(
        "Training CATBoost (warning: may exceed 600s if no pretrained models are available)"
    )
    catboost()
else:
    if CFG.LOAD_MODELS_FROM == "":
        print("Loading local models from working directory (skip training).")
    else:
        print("Loading models from:", CFG.LOAD_MODELS_FROM)



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
test_text_pre = df_test["full_text_pre"]
test_tfid = vectorizer.transform(test_text_pre).astype(np.float32)  # CSR sparse matrix



## === cell 39
test_cnt = vectorizer_cnt.transform(test_text_pre).astype(
    np.float32
)  # CSR sparse matrix



## === cell 40
t0 = time.time()

test_pl = pl.from_pandas(df_test[["essay_id", "full_text_pre"]]).with_columns(
    pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph")
)

test_feats1 = Paragraph_Features(test_pl.select(["essay_id", "paragraph"]))
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test_pl.select(["essay_id", "full_text_pre"]))
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test_pl.select(["essay_id", "full_text_pre"]))
test_feats3 = Word_aggregation(test_feats3)
print(f"Test feature extraction took {time.time()-t0:.1f}s")

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
print("Shape of test_feats (engineered only):", test_feats.shape)
display(test_feats.head())

missing_cols = [c for c in ENGINEERED_FEATURES if c not in test_feats.columns]
if missing_cols:
    for c in missing_cols:
        test_feats[c] = 0.0
extra_cols = [
    c for c in test_feats.columns if (c not in ENGINEERED_FEATURES + ["essay_id"])
]
if extra_cols:
    pass

X_test_num = (
    test_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False)
)
np.clip(X_test_num, 0, 10000, out=X_test_num)
X_test_num_csr = sparse.csr_matrix(X_test_num, dtype=np.float32)
X_test_all = sparse.hstack(
    [X_test_num_csr, test_tfid, test_cnt], format="csr", dtype=np.float32
)

model_base = (
    CFG.LOAD_MODELS_FROM
    if (CFG.LOAD_MODELS_FROM is not None and CFG.LOAD_MODELS_FROM != "")
    else ""
)
use_dir = bool(CFG.LOAD_MODELS_FROM not in (None, ""))

pred_sum = None
for i in range(10):
    print(f"Fold {i+1}")
    if use_dir:
        mp = os.path.join(model_base, f"CAT_v{CFG.VER}_f{i}.pkl")
    else:
        mp = f"CAT_v{CFG.VER}_f{i}.pkl"

    model = pickle.load(open(mp, "rb"))
    pred_i = model.predict(X_test_all)
    if pred_sum is None:
        pred_sum = pred_i.astype(np.float64, copy=False)
    else:
        pred_sum += pred_i
    del model, pred_i
    clean_memory()

pred = pred_sum / 10.0

sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub[TARGET] = np.clip(pred, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Wrote submission.csv:", os.path.exists("submission.csv"))

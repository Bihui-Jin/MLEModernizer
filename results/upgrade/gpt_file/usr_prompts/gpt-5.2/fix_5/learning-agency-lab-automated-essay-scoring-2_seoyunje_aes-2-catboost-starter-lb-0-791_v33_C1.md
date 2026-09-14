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
import polars as pl  # For Feature Engineering

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
def seed_everything():  # To proudce simliar result in each run
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
df_train = df_train.sort_values(by="essay_id")

print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 10
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

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
@lru_cache(maxsize=400_000)
def count_misspelled_words(text):
    toks = _simple_tokenize(text)
    cnt = 0
    for t in toks:
        if not t:
            continue
        if any(ch.isdigit() for ch in t):
            cnt += 1
            continue
        if len(t) >= 18:
            cnt += 1
            continue
        if re.search(r"(.)\1\1", t):
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
    x = x.explode("paragraph")

    print("Paragraph Preprocessing")
    x = x.with_columns(
        pl.col("paragraph").map_elements(dataPreprocessing, return_dtype=pl.Utf8)
    )

    x = x.with_columns(
        pl.col("paragraph")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int64)
        .alias("paragraph_len"),
        pl.col("paragraph")
        .str.count_matches(",")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("paragraph_comma_cnt"),
    )

    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(
            lambda v: count_misspelled_words(v) if v is not None else 0,
            return_dtype=pl.Int64,
        )
        .alias("paragraph_misspelled_cnt")
    )

    print("Caculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.when(
            pl.col("paragraph").is_null() | (pl.col("paragraph").str.len_chars() == 0)
        )
        .then(0)
        .otherwise(pl.col("paragraph").str.split(".").list.len())
        .cast(pl.Int64)
        .alias("paragraph_sentence_cnt"),
        pl.when(
            pl.col("paragraph").is_null() | (pl.col("paragraph").str.len_chars() == 0)
        )
        .then(0)
        .otherwise(pl.col("paragraph").str.split(" ").list.len())
        .cast(pl.Int64)
        .alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x):

    print("Aggregation")
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") >= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
            for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") <= i)
            .count()
            .alias(f"paragraph_{i}_cnt_v2")
            for i in [100, 200]
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
            .count()
            .alias(f"short_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
            .count()
            .alias(f"mid_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
            .count()
            .alias(f"long_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 4)
                & (pl.col("paragraph_sentence_cnt") > 2)
            )
            .count()
            .alias(f"short_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 8)
                & (pl.col("paragraph_sentence_cnt") > 4)
            )
            .count()
            .alias(f"mid_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 10)
                & (pl.col("paragraph_sentence_cnt") > 8)
            )
            .count()
            .alias(f"long_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") >= i)
            .count()
            .alias(f"paragraph_word_{i}_cnt")
            for i in [20, 40, 60, 90, 120]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 40)
                & (pl.col("paragraph_word_cnt") > 20)
            )
            .count()
            .alias(f"short_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 90)
                & (pl.col("paragraph_word_cnt") > 40)
            )
            .count()
            .alias(f"mid_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 120)
                & (pl.col("paragraph_word_cnt") > 90)
            )
            .count()
            .alias(f"long_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_comma_cnt") >= i)
            .count()
            .alias(f"paragraph_comma_{i}_cnt")
            for i in [1, 2, 3, 4, 5]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_misspelled_cnt") >= i)
            .count()
            .alias(f"paragraph_misspelled_{i}_cnt")
            for i in [4, 8, 12, 16]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_misspelled_cnt") <= i)
            .count()
            .alias(f"paragraph_misspelled_{i}_cnt_v2")
            for i in [2, 4]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 8)
                & (pl.col("paragraph_misspelled_cnt") > 4)
            )
            .count()
            .alias(f"short_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 12)
                & (pl.col("paragraph_misspelled_cnt") > 8)
            )
            .count()
            .alias(f"mid_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 16)
                & (pl.col("paragraph_misspelled_cnt") > 12)
            )
            .count()
            .alias(f"long_paragraph_misspelled_cnt")
        ],
        *[pl.col("paragraph").count().alias("paragraph_cnt")],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in paragraph_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in paragraph_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in paragraph_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in paragraph_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in paragraph_features],
        *[
            pl.col(feat).quantile(0.25).alias(f"{feat}_q1")
            for feat in paragraph_features
        ],
        *[
            pl.col(feat).quantile(0.75).alias(f"{feat}_q3")
            for feat in paragraph_features
        ],
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 18
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x):
    if "full_text_pre" in x.columns:
        base = pl.col("full_text_pre")
    else:
        base = pl.col("full_text").map_elements(dataPreprocessing, return_dtype=pl.Utf8)

    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(base.str.split(".").alias("sentence"))
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(
        pl.col("sentence")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int64)
        .alias("sentence_len"),
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .fill_null(0)
        .cast(pl.Int64)
        .alias("only_sentence_len"),
    )
    x = x.filter(pl.col("sentence_len") > 3)

    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.when(
            pl.col("sentence").is_null() | (pl.col("sentence").str.len_chars() == 0)
        )
        .then(0)
        .otherwise(pl.col("sentence").str.split(" ").list.len())
        .cast(pl.Int64)
        .alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x):

    print("Aggregation")
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [40, 60, 70, 80, 100, 120, 140]
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") <= i)
            .count()
            .alias(f"sentence_{i}_cnt_v2")
            for i in [10, 20, 30]
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
            .count()
            .alias(f"short_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias(f"mid_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
            .count()
            .alias(f"long_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("only_sentence_len") >= i)
            .count()
            .alias(f"only_sentence_{i}_cnt")
            for i in [40, 60, 80, 100, 120]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40)
            )
            .count()
            .alias(f"short_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 100)
                & (pl.col("only_sentence_len") > 60)
            )
            .count()
            .alias(f"mid_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 120)
                & (pl.col("only_sentence_len") > 100)
            )
            .count()
            .alias(f"long_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [10, 15, 20, 25]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
            )
            .count()
            .alias(f"short_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
            )
            .count()
            .alias(f"mid_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20)
            )
            .count()
            .alias(f"long_sentence_word_cnt")
        ],
        *[pl.col("sentence").count().alias("sentence_cnt")],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in sentence_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in sentence_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in sentence_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in sentence_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in sentence_features],
        *[
            pl.col(feat).quantile(0.25).alias(f"{feat}_q1")
            for feat in sentence_features
        ],
        *[
            pl.col(feat).quantile(0.75).alias(f"{feat}_q3")
            for feat in sentence_features
        ],
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        *[
            (pl.col(f"sentence_{i}_cnt") / pl.col("sentence_cnt")).alias(
                f"sentence_{i}_cnt_ratio"
            )
            for i in [40, 60, 70, 80, 100, 120, 140]
        ],
        *[
            (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"short_sentence_cnt_ratio"
            )
        ],
        *[
            (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"mid_sentence_cnt_ratio"
            )
        ],
        *[
            (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"long_sentence_cnt_ratio"
            )
        ],
    ).sort("essay_id")

    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 19
word_features = [
    "word_len",
]


def Word_Features(x):
    print("Preprocess full_text and use spaces to seperate words fro the text")

    if "full_text_pre" in x.columns:
        base = pl.col("full_text_pre")
    else:
        base = pl.col("full_text").map_elements(dataPreprocessing, return_dtype=pl.Utf8)

    x = x.with_columns(base.str.split(" ").alias("word"))
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(
        pl.col("word").str.len_chars().fill_null(0).cast(pl.Int64).alias("word_len")
    )
    x = x.filter(pl.col("word_len") > 0)

    return x


def Word_aggregation(x):

    print("Aggregation")
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"word_{i}_cnt")
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            pl.col("word")
            .filter(pl.col("word_len") <= i)
            .count()
            .alias(f"word_{i}_cnt_v2")
            for i in [1, 2, 3]
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
            .count()
            .alias(f"short_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
            .count()
            .alias(f"mid_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
            .count()
            .alias(f"long_word_cnt")
        ],
        *[pl.col("word").count().alias("word_cnt")],
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in word_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in word_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in word_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in word_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in word_features],
        *[pl.col(feat).quantile(0.25).alias(f"{feat}_q1") for feat in word_features],
        *[pl.col(feat).quantile(0.75).alias(f"{feat}_q3") for feat in word_features],
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_cnt")).alias(f"word_{i}_cnt_ratio")
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"word_{i}_cnt_v2") / pl.col("word_cnt")).alias(
                f"word_{i}_cnt_v2_ratio"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_2_cnt_v2")).alias(
                f"word_{i}_pre2_ratio"
            )
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_3_cnt_v2")).alias(
                f"word_{i}_pre3_ratio"
            )
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
    ).sort("essay_id")

    df = df.to_pandas()  # polars -> pandas

    return df




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

train = train.with_columns(
    pl.col("full_text")
    .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
    .alias("full_text_pre")
)
train_text_pre = train["full_text_pre"].to_list()

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

    train_feats1 = Paragraph_Features(train.select(["essay_id", "paragraph"]))
    train_feats1 = Paragraph_aggregation(train_feats1)

    train_feats2 = Sentence_Features(train.select(["essay_id", "full_text_pre"]))
    train_feats2 = Sentence_aggregation(train_feats2)

    train_feats3 = Word_Features(train.select(["essay_id", "full_text_pre"]))
    train_feats3 = Word_aggregation(train_feats3)

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
X_num = np.clip(X_num, 0, 10000)

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
            thread_count=-1,
            objective="RMSE",
            eval_metric="RMSE",
            random_state=CFG.SEED,
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
test = test.with_columns(
    pl.col("full_text")
    .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
    .alias("full_text_pre")
)
test_text_pre = test["full_text_pre"].to_list()
test_tfid = vectorizer.transform(test_text_pre)  # CSR sparse matrix



## === cell 39
test_cnt = vectorizer_cnt.transform(test_text_pre)  # CSR sparse matrix



## === cell 40
t0 = time.time()
test_feats1 = Paragraph_Features(test.select(["essay_id", "paragraph"]))
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test.select(["essay_id", "full_text_pre"]))
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test.select(["essay_id", "full_text_pre"]))
test_feats3 = Word_aggregation(test_feats3)
print(f"Test feature extraction took {time.time()-t0:.1f}s")



## === cell 41
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
print("Shape of test_feats (engineered only):", test_feats.shape)
display(test_feats.head())

X_test_num = (
    test_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False)
)
X_test_num = np.clip(X_test_num, 0, 10000)

X_test_num_csr = sparse.csr_matrix(X_test_num)
X_test_all = sparse.hstack([X_test_num_csr, test_tfid, test_cnt], format="csr")



## === cell 42
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

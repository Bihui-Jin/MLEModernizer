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

cudf-polars-cu12==25.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

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

0.8037614614128175

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the missing external-input dependency by auto-falling back to local feature generation and local model training when the specified `/kaggle/input/aes2-xgboost-starter/` files are not present. I also remove notebook-only constructs that cause runtime errors in script execution (e.g., `display`, `%%time`) while preserving the model/feature logic. To ensure train/test feature columns align, I use the same training `FEATURES` list at inference time and add any missing columns in test with zeros (score-neutral but prevents shape/key errors). Finally, I ensure a valid `submission.csv` is always written with columns `essay_id,score`.'

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

import pandas as pd, numpy as np
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For GPU T4x2

import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count()))




## === cell 1
def _exists(path: str) -> bool:
    return path is not None and isinstance(path, str) and os.path.exists(path)




## === cell 2
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-xgboost-starter/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-xgboost-starter/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 3
if not os.path.exists(os.path.join(CFG.BASE_PATH, "train.csv")):
    alt = "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/"
    if os.path.exists(os.path.join(alt, "train.csv")):
        CFG.BASE_PATH = alt
    else:
        alt2 = "/kaggle/data/"
        if os.path.exists(os.path.join(alt2, "train.csv")):
            CFG.BASE_PATH = alt2



## === cell 4
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()



## === cell 5
if not _exists(CFG.LOAD_FEATURES_FROM):
    CFG.LOAD_FEATURES_FROM = None
if not _exists(CFG.LOAD_MODELS_FROM):
    CFG.LOAD_MODELS_FROM = None
else:
    if not CFG.LOAD_MODELS_FROM.endswith("/"):
        CFG.LOAD_MODELS_FROM += "/"




## === cell 6
def seed_everything():  # To proudce simliar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 7
plt.switch_backend("Agg")



## === cell 8
read_kwargs = (
    dict(engine="pyarrow") if "pyarrow" in pd.io.common._get_handle.__module__ else {}
)
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv", **read_kwargs)

print("Shape of Train: ", df_train.shape)
print(df_train.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2988246524.py in <cell line: 0>()
      1 # Speed-only: faster CSV parsing with pyarrow engine when available; identical content.
      2 read_kwargs = (
----> 3     dict(engine="pyarrow") if "pyarrow" in pd.io.common._get_handle.__module__ else {}
      4 )
      5 df_train = pd.read_csv(CFG.BASE_PATH + "train.csv", **read_kwargs)

AttributeError: module 'pandas.io.common' has no attribute '_get_handle'

## === cell 9
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv", **read_kwargs)

print("Shape of Test: ", df_test.shape)
print(df_test.head())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2564891268.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(CFG.BASE_PATH + "test.csv", **read_kwargs)
      2 
      3 print("Shape of Test: ", df_test.shape)
      4 print(df_test.head())
      5 

NameError: name 'read_kwargs' is not defined

## === cell 10
pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(10)

train = pl.from_pandas(df_train[["essay_id", "full_text"]]).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test[["essay_id", "full_text"]]).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2307301293.py in <cell line: 0>()
      3 pl.Config.set_tbl_cols(10)
      4 
----> 5 train = pl.from_pandas(df_train[["essay_id", "full_text"]]).with_columns(
      6     pl.col("full_text").str.split(by="\n\n").alias("paragraph")
      7 )

NameError: name 'df_train' is not defined

## === cell 11
df_train["full_text"] = df_train["full_text"].fillna("").astype(str)
df_test["full_text"] = df_test["full_text"].fillna("").astype(str)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3637736007.py in <cell line: 0>()
----> 1 df_train["full_text"] = df_train["full_text"].fillna("").astype(str)
      2 df_test["full_text"] = df_test["full_text"].fillna("").astype(str)
      3 
      4 

NameError: name 'df_train' is not defined

## === cell 12
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)  # html -> ''


def dataPreprocessing(x):
    if x is None:
        x = ""
    if not isinstance(x, str):
        x = str(x)

    x = x.lower()
    x = removeHTML(x)

    x = re.sub("@\w+", "", x)

    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)

    x = re.sub("http\w+", "", x)

    x = re.sub(r"\s+", " ", x)

    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ".", x)
    x = x.strip()
    return x




## === cell 13
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]

_HTML_RE = r"<.*?>"
_AT_RE = r"@\w+"
_APOS_NUM_RE = r"'\d+"
_NUM_RE = r"\d+"
_HTTP_RE = r"http\w+"
_WS_RE = r"\s+"
_DOTS_RE = r"\.+"
_COMMAS_RE = r"\,+"


def _preprocess_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(_HTML_RE, "")
        .str.replace_all(_AT_RE, "")
        .str.replace_all(_APOS_NUM_RE, "")
        .str.replace_all(_NUM_RE, "")
        .str.replace_all(_HTTP_RE, "")
        .str.replace_all(_WS_RE, " ")
        .str.replace_all(_DOTS_RE, ".")
        .str.replace_all(_COMMAS_RE, ".")
        .str.strip_chars()
    )


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.explode("paragraph")
    x = x.with_columns(_preprocess_expr("paragraph").alias("paragraph"))

    x = x.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        pl.col("paragraph").str.split(".").list.len().alias("paragraph_sentence_cnt"),
        pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
        .count()
        .alias("short_paragraph_cnt"),
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
        .count()
        .alias("mid_paragraph_cnt"),
        pl.col("paragraph")
        .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
        .count()
        .alias("long_paragraph_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_sentence_cnt") <= 4)
            & (pl.col("paragraph_sentence_cnt") > 2)
        )
        .count()
        .alias("short_paragraph_sentence_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_sentence_cnt") <= 8)
            & (pl.col("paragraph_sentence_cnt") > 4)
        )
        .count()
        .alias("mid_paragraph_sentence_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_sentence_cnt") <= 10)
            & (pl.col("paragraph_sentence_cnt") > 8)
        )
        .count()
        .alias("long_paragraph_sentence_cnt"),
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") >= i)
            .count()
            .alias(f"paragraph_word_{i}_cnt")
            for i in [30, 60, 90, 120]
        ],
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 60) & (pl.col("paragraph_word_cnt") > 30)
        )
        .count()
        .alias("short_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 90) & (pl.col("paragraph_word_cnt") > 60)
        )
        .count()
        .alias("mid_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 120) & (pl.col("paragraph_word_cnt") > 90)
        )
        .count()
        .alias("long_paragraph_word_cnt"),
        pl.col("paragraph").count().alias("paragraph_cnt"),
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 14
sentence_features = [
    "sentence_len",
    "sentence_word_cnt",
    "sentence_len_space_ratio",
    "sentence_word_space_ratio",
]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.with_columns(_preprocess_expr("full_text").str.split(".").alias("sentence"))
    x = x.explode("sentence")

    x = x.with_columns(
        pl.col("sentence").str.count_matches(" ").alias("sentence_space_cnt")
    )
    x = x.filter(pl.col("sentence_space_cnt") > 0)

    x = x.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
    x = x.filter(pl.col("sentence_len") > 3)

    x = x.with_columns(
        (pl.col("sentence_len") / pl.col("sentence_space_cnt")).alias(
            "sentence_len_space_ratio"
        )
    )

    x = x.with_columns(
        pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
    )
    x = x.with_columns(
        (pl.col("sentence_word_cnt") / pl.col("sentence_space_cnt")).alias(
            "sentence_word_space_ratio"
        )
    )
    return x


def Sentence_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [30, 40, 50, 60, 70, 80, 100, 150]
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") <= i)
            .count()
            .alias(f"sentence_{i}_cnt_v2")
            for i in [10, 20, 30]
        ],
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 50) & (pl.col("sentence_len") > 30))
        .count()
        .alias("short_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 50))
        .count()
        .alias("mid_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
        .count()
        .alias("long_sentence_cnt"),
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [5, 10, 15, 20]
        ],
        pl.col("sentence")
        .filter((pl.col("sentence_word_cnt") <= 10) & (pl.col("sentence_word_cnt") > 5))
        .count()
        .alias("short_sentence_word_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
        )
        .count()
        .alias("mid_sentence_word_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
        )
        .count()
        .alias("long_sentence_word_cnt"),
        pl.col("sentence").count().alias("sentence_cnt"),
        *[pl.col(feat).max().alias(f"{feat}_max") for feat in sentence_features],
        *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in sentence_features],
        *[pl.col(feat).min().alias(f"{feat}_min") for feat in sentence_features],
        *[pl.col(feat).std().alias(f"{feat}_std") for feat in sentence_features],
        *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in sentence_features],
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 15
word_features = [
    "word_len",
]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.with_columns(_preprocess_expr("full_text").str.split(".").alias("sentence"))
    x = x.explode("sentence")

    x = x.with_columns(_preprocess_expr("sentence").str.split(" ").alias("word"))
    x = x.explode("word")

    x = x.with_columns(pl.col("word").str.len_chars().alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x: pl.DataFrame) -> pd.DataFrame:
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i)
            .count()
            .alias(f"word_{i}_cnt")
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            pl.col("word")
            .filter(pl.col("word_len") <= i)
            .count()
            .alias(f"word_{i}_cnt_v2")
            for i in [2, 3]
        ],
        pl.col("word")
        .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
        .count()
        .alias("short_word_cnt"),
        pl.col("word")
        .filter((pl.col("word_len") <= 7) & (pl.col("word_len") > 4))
        .count()
        .alias("mid_word_cnt"),
        pl.col("word")
        .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 7))
        .count()
        .alias("long_word_cnt"),
        pl.col("word").count().alias("word_cnt"),
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
            (pl.col(f"word_{i}_cnt") / pl.col("word_2_cnt_v2")).alias(
                f"word_2_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_3_cnt_v2")).alias(
                f"word_3_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col("short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col("mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col("long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
    ).sort("essay_id")

    return df.to_pandas()




## === cell 16
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,  # Term Frequency Log Scaling
)

train_text = train["full_text"].to_list()
train_tfid = vectorizer.fit_transform(train_text)

n_rows, n_cols = train_tfid.shape
block = 8192  # materialization block size (exact values preserved)
dense_blocks = []
for start in range(0, n_rows, block):
    end = min(start + block, n_rows)
    dense_blocks.append(train_tfid[start:end].toarray().astype(np.float32, copy=False))
dense_matrix = np.vstack(dense_blocks)

df = pd.DataFrame(dense_matrix, columns=[f"tfidf_{i}" for i in range(n_cols)])
df["essay_id"] = df_train["essay_id"].values



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4049997786.py in <cell line: 0>()
     13 )
     14 
---> 15 train_text = train["full_text"].to_list()
     16 train_tfid = vectorizer.fit_transform(train_text)
     17 

NameError: name 'train' is not defined

## === cell 17
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=0.10,
    max_df=0.90,
)

_ = vectorizer_cnt.fit_transform(train_text)  # fitted for parity; not densified



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2139744584.py in <cell line: 0>()
     14 )
     15 
---> 16 _ = vectorizer_cnt.fit_transform(train_text)  # fitted for parity; not densified
     17 

NameError: name 'train_text' is not defined

## === cell 18
if CFG.LOAD_FEATURES_FROM is None:

    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    def _dedup_sort(df_in: pd.DataFrame) -> pd.DataFrame:
        return (
            df_in.sort_values("essay_id")
            .drop_duplicates("essay_id", keep="first")
            .reset_index(drop=True)
        )

    train_feats1 = _dedup_sort(train_feats1)
    train_feats2 = _dedup_sort(train_feats2)
    train_feats3 = _dedup_sort(train_feats3)
    df = _dedup_sort(df)

    train_feats = train_feats1.merge(
        train_feats2, on="essay_id", how="left", validate="one_to_one"
    )
    train_feats = train_feats.merge(
        train_feats3, on="essay_id", how="left", validate="one_to_one"
    )
    train_feats = train_feats.merge(
        df, on="essay_id", how="left", validate="one_to_one"
    )

    score_map = df_train.set_index("essay_id")["score"]
    train_feats["score"] = train_feats["essay_id"].map(score_map).astype(int)
else:
    train_feats = None



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3561640782.py in <cell line: 0>()
      3 if CFG.LOAD_FEATURES_FROM is None:
      4 
----> 5     train_feats1 = Paragraph_Features(train)
      6     train_feats1 = Paragraph_aggregation(train_feats1)
      7     train_feats2 = Sentence_Features(train)

NameError: name 'train' is not defined

## === cell 19
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM, **read_kwargs)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2816502696.py in <cell line: 0>()
      2 if CFG.LOAD_FEATURES_FROM is None:
      3     print("Save train_feats.csv")
----> 4     train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      5 else:
      6     print("Load train_feats.csv")

NameError: name 'train_feats' is not defined

## === cell 20
print(train_feats.head())



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2022479248.py in <cell line: 0>()
----> 1 print(train_feats.head())
      2 

NameError: name 'train_feats' is not defined

## === cell 21
obj_cols = [
    c
    for c in train_feats.columns
    if c not in ["essay_id", "score"] and train_feats[c].dtype == "object"
]
if obj_cols:
    train_feats[obj_cols] = train_feats[obj_cols].apply(pd.to_numeric, errors="coerce")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/300709916.py in <cell line: 0>()
      2 obj_cols = [
      3     c
----> 4     for c in train_feats.columns
      5     if c not in ["essay_id", "score"] and train_feats[c].dtype == "object"
      6 ]

NameError: name 'train_feats' is not defined

## === cell 22
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb

print("XGBoost Version: ", xgb.__version__)



## === cell 23
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/75358094.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 FEATURES = [
      5     col for col in train_feats.columns if col not in categorical_columns + ["score"]

NameError: name 'train_feats' is not defined

## === cell 24
def quadratic_weighted_kappa(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk




## === cell 25
def xgboost():
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):

        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.02,
            max_depth=8,
            min_child_weight=5,
            subsample=0.7,
            n_estimators=1024,
            random_state=CFG.SEED,
            verbosity=0,
            n_jobs=os.cpu_count(),
        )

        train_x = np.clip(train_feats.loc[train_index, FEATURES].fillna(0), 0, 10000)
        train_y = train_feats.loc[train_index, TARGET]

        valid_x = np.clip(train_feats.loc[valid_index, FEATURES].fillna(0), 0, 10000)
        valid_y = train_feats.loc[valid_index, TARGET]

        try:
            model.fit(
                train_x,
                train_y,
                eval_set=[(valid_x, valid_y)],
                early_stopping_rounds=100,
                verbose=50,
            )
        except TypeError:
            model.fit(
                train_x,
                train_y,
                eval_set=[(valid_x, valid_y)],
                callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                verbose=50,
            )

        pickle.dump(model, open(f"XGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(valid_x)
        all_oof.append(oof)
        all_true.append(valid_y.values)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = cohen_kappa_score(true[0], oof[0].clip(1, 6).round(), weights="quadratic")
    print("CV Score for XGBoost = ", cv)




## === cell 26
def _all_folds_exist(base: str) -> bool:
    return base is not None and all(
        _exists(f"{base}XGB_v{CFG.VER}_f{i}.pkl") for i in range(5)
    )


if _all_folds_exist(CFG.LOAD_MODELS_FROM):
    print("Using pre-trained models from:", CFG.LOAD_MODELS_FROM)
elif all(_exists(f"XGB_v{CFG.VER}_f{i}.pkl") for i in range(5)):
    print("Using local pre-trained models from working dir.")
else:
    print("Training XGBoost (no pre-trained models found)")
    xgboost()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4060749283.py in <cell line: 0>()
     13 else:
     14     print("Training XGBoost (no pre-trained models found)")
---> 15     xgboost()
     16 

/tmp/ipykernel_55/713363563.py in xgboost()
      5     skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
      6     for i, (train_index, valid_index) in enumerate(
----> 7         skf.split(train_feats, train_feats[TARGET])
      8     ):
      9 

NameError: name 'train_feats' is not defined

## === cell 27
model_path = None
if CFG.LOAD_MODELS_FROM is not None and _exists(
    f"{CFG.LOAD_MODELS_FROM}XGB_v{CFG.VER}_f0.pkl"
):
    model_path = f"{CFG.LOAD_MODELS_FROM}XGB_v{CFG.VER}_f0.pkl"
elif _exists(f"XGB_v{CFG.VER}_f0.pkl"):
    model_path = f"XGB_v{CFG.VER}_f0.pkl"

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
    plt.title("Distribution of Feature Importance of XGBoost")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
else:
    print("Skipping feature importance plot: model file not found.")



## === cell 28
test_text = test["full_text"].to_list()
test_tfid = vectorizer.transform(test_text)
n_rows_t, n_cols_t = test_tfid.shape
dense_blocks_t = []
for start in range(0, n_rows_t, block):
    end = min(start + block, n_rows_t)
    dense_blocks_t.append(test_tfid[start:end].toarray().astype(np.float32, copy=False))
dense_matrix = np.vstack(dense_blocks_t)

df3 = pd.DataFrame(dense_matrix, columns=[f"tfidf_{i}" for i in range(n_cols_t)])
df3["essay_id"] = df_test["essay_id"].values



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3332327366.py in <cell line: 0>()
      1 # Speed-only: float32 densification for TFIDF; same transform, faster/lower memory.
----> 2 test_text = test["full_text"].to_list()
      3 test_tfid = vectorizer.transform(test_text)
      4 n_rows_t, n_cols_t = test_tfid.shape
      5 dense_blocks_t = []

NameError: name 'test' is not defined

## === cell 29
_ = vectorizer_cnt.transform(test_text)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1140165501.py in <cell line: 0>()
      1 # Speed-only: Keep parity by transforming count features but do not materialize dense df4
      2 # since it is not used in final test_feats merge.
----> 3 _ = vectorizer_cnt.transform(test_text)
      4 

NameError: name 'test_text' is not defined

## === cell 30
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2387858613.py in <cell line: 0>()
----> 1 test_feats1 = Paragraph_Features(test)
      2 test_feats1 = Paragraph_aggregation(test_feats1)
      3 test_feats2 = Sentence_Features(test)
      4 test_feats2 = Sentence_aggregation(test_feats2)
      5 test_feats3 = Word_Features(test)

NameError: name 'test' is not defined

## === cell 31
sample_path = os.path.join(CFG.BASE_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path, **read_kwargs)[["essay_id"]].copy()


def _dedup_sort(df_in: pd.DataFrame) -> pd.DataFrame:
    return (
        df_in.sort_values("essay_id")
        .drop_duplicates("essay_id", keep="first")
        .reset_index(drop=True)
    )


test_feats1 = _dedup_sort(test_feats1)
test_feats2 = _dedup_sort(test_feats2)
test_feats3 = _dedup_sort(test_feats3)
df3 = _dedup_sort(df3)

test_feats = sample_sub.merge(
    test_feats1, on="essay_id", how="left", validate="one_to_one"
)
test_feats = test_feats.merge(
    test_feats2, on="essay_id", how="left", validate="one_to_one"
)
test_feats = test_feats.merge(
    test_feats3, on="essay_id", how="left", validate="one_to_one"
)
test_feats = test_feats.merge(df3, on="essay_id", how="left", validate="one_to_one")

print("Shape of test_feats:", test_feats.shape)
print(test_feats.head())



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3433312809.py in <cell line: 0>()
      1 sample_path = os.path.join(CFG.BASE_PATH, "sample_submission.csv")
----> 2 sample_sub = pd.read_csv(sample_path, **read_kwargs)[["essay_id"]].copy()
      3 
      4 
      5 def _dedup_sort(df_in: pd.DataFrame) -> pd.DataFrame:

NameError: name 'read_kwargs' is not defined

## === cell 32
obj_cols_t = [
    c for c in test_feats.columns if c != "essay_id" and test_feats[c].dtype == "object"
]
if obj_cols_t:
    test_feats[obj_cols_t] = test_feats[obj_cols_t].apply(
        pd.to_numeric, errors="coerce"
    )

for col in FEATURES:
    if col not in test_feats.columns:
        test_feats[col] = 0

test_x = np.clip(
    test_feats[FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False), 0, 10000
)

preds = []
for i in range(5):
    print(f"Fold {i+1}")
    fold_path = None
    if CFG.LOAD_MODELS_FROM is not None and _exists(
        f"{CFG.LOAD_MODELS_FROM}XGB_v{CFG.VER}_f{i}.pkl"
    ):
        fold_path = f"{CFG.LOAD_MODELS_FROM}XGB_v{CFG.VER}_f{i}.pkl"
    elif _exists(f"XGB_v{CFG.VER}_f{i}.pkl"):
        fold_path = f"XGB_v{CFG.VER}_f{i}.pkl"
    else:
        raise FileNotFoundError(
            f"Missing model for fold {i}: expected external or local pickle."
        )

    model = pickle.load(open(fold_path, "rb"))
    pred = model.predict(test_x)
    preds.append(pred)

pred = np.mean(preds, axis=0)

sub = pd.DataFrame({"essay_id": test_feats["essay_id"].values})
sub[TARGET] = np.clip(pred, 1, 6).round().astype(int)

assert sub.columns.tolist() == ["essay_id", "score"]
assert len(sub) == len(
    pd.read_csv(os.path.join(CFG.BASE_PATH, "sample_submission.csv"), **read_kwargs)
)

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Wrote: submission.csv")

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3569968137.py in <cell line: 0>()
      1 # Speed-only: vectorized coercion and build test matrix once; avoid repeated clip/fill per fold.
      2 obj_cols_t = [
----> 3     c for c in test_feats.columns if c != "essay_id" and test_feats[c].dtype == "object"
      4 ]
      5 if obj_cols_t:

NameError: name 'test_feats' is not defined

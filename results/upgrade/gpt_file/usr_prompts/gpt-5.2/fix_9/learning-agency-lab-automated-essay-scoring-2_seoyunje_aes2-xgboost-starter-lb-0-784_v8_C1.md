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

# 5. Code solution

## === cell 0
import os
import gc
import ctypes
import random
import re
import pickle
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb
import scipy.sparse as sp

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def display(x):
    print(x)


print("XGBoost Version:", xgb.__version__)




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-xgboost/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-xgboost/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()


def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()
clean_memory()



## === cell 3
df_train = (
    pd.read_csv(
        CFG.BASE_PATH + "train.csv",
        usecols=["essay_id", "full_text", "score"],
        dtype={"essay_id": "string", "full_text": "string", "score": "int8"},
    )
    .sort_values(by="essay_id")
    .reset_index(drop=True)
)
df_test = (
    pd.read_csv(
        CFG.BASE_PATH + "test.csv",
        usecols=["essay_id", "full_text"],
        dtype={"essay_id": "string", "full_text": "string"},
    )
    .sort_values(by="essay_id")
    .reset_index(drop=True)
)

print("Shape of Train:", df_train.shape)
display(df_train.head())
print("Shape of Test:", df_test.shape)
display(df_test.head())

train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema
schema_test = test.schema



## === cell 4
_html_re = re.compile(r"<.*?>")
_at_re = re.compile(r"@\w+")
_quote_num_re = re.compile(r"'\d+")
_num_re = re.compile(r"\d+")
_http_re = re.compile(r"http\w+")
_space_re = re.compile(r"\s+")
_keepchars_re = re.compile(r"[^\w\s.,;:\"''?!]")
_para_word_re = re.compile("paragraph")
_dots_re = re.compile(r"\.+")
_commas_re = re.compile(r"\,+")


def removeHTML(x: str) -> str:
    return _html_re.sub(r"", x)


def dataPreprocessing(x: str) -> str:
    x = str(x).lower()
    x = removeHTML(x)
    x = _at_re.sub("", x)
    x = _quote_num_re.sub("", x)
    x = _num_re.sub("", x)
    x = _http_re.sub("", x)
    x = _space_re.sub(" ", x)
    x = _keepchars_re.sub("", x)
    x = _para_word_re.sub("", x)
    x = _dots_re.sub(".", x)
    x = _commas_re.sub(",", x)
    x = x.strip()
    return x




## === cell 5
_word_good_re = re.compile(r"^[A-Za-z]{2,24}$")
_strip_edges_re = re.compile(r"^[\W_]+|[\W_]+$")
_token_re = re.compile(r"\S+")


@lru_cache(maxsize=200_000)
def count_misspelled_words(text: str) -> int:
    text = str(text)
    tokens = _token_re.findall(text)
    if not tokens:
        return 0
    nonempty = 0
    good = 0
    for t in tokens:
        tt = _strip_edges_re.sub("", t)
        if not tt:
            continue
        nonempty += 1
        if _word_good_re.fullmatch(tt) is not None:
            good += 1
    return nonempty - good


def _misspell_counts_parallel(
    texts: list[str], max_workers: int | None = None
) -> list[int]:
    if not texts:
        return []
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4))
    if max_workers <= 1 or len(texts) < 2048:
        return [count_misspelled_words(t) for t in texts]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(count_misspelled_words, texts, chunksize=2048))




## === cell 6
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def _polars_preprocess_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"[^\w\s.,;:\"''?!]", "")
        .str.replace_all(r"paragraph", "")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
    )


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    lf = x.lazy().explode("paragraph")
    lf = lf.with_columns(_polars_preprocess_expr("paragraph").alias("paragraph"))
    lf = lf.with_columns(
        pl.col("paragraph").str.len_chars().cast(pl.Int64).alias("paragraph_len"),
        pl.col("paragraph")
        .str.count_matches(",")
        .cast(pl.Int64)
        .alias("paragraph_comma_cnt"),
        pl.col("paragraph")
        .str.split(".")
        .list.len()
        .cast(pl.Int64)
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .str.split(" ")
        .list.len()
        .cast(pl.Int64)
        .alias("paragraph_word_cnt"),
    )
    df = lf.collect(streaming=True)
    paras = df.get_column("paragraph").to_list()
    miss = _misspell_counts_parallel(paras, max_workers=min(32, (os.cpu_count() or 4)))
    df = df.with_columns(pl.Series("paragraph_misspelled_cnt", miss, dtype=pl.Int64))
    return df


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
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
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
            for i in [20, 40, 60, 90, 120]
        ],
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 40) & (pl.col("paragraph_word_cnt") > 20)
        )
        .count()
        .alias("short_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 90) & (pl.col("paragraph_word_cnt") > 40)
        )
        .count()
        .alias("mid_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 120) & (pl.col("paragraph_word_cnt") > 90)
        )
        .count()
        .alias("long_paragraph_word_cnt"),
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
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_misspelled_cnt") <= 8)
            & (pl.col("paragraph_misspelled_cnt") > 4)
        )
        .count()
        .alias("short_paragraph_misspelled_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_misspelled_cnt") <= 12)
            & (pl.col("paragraph_misspelled_cnt") > 8)
        )
        .count()
        .alias("mid_paragraph_misspelled_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_misspelled_cnt") <= 16)
            & (pl.col("paragraph_misspelled_cnt") > 12)
        )
        .count()
        .alias("long_paragraph_misspelled_cnt"),
        pl.col("paragraph").count().alias("paragraph_cnt"),
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

    df = (
        x.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )
    return df.to_pandas()




## === cell 7
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(
    x: pl.DataFrame, clean_col: str = "full_text_clean"
) -> pl.DataFrame:
    if clean_col in x.columns:
        base = pl.col(clean_col)
    else:
        base = _polars_preprocess_expr("full_text")

    lf = (
        x.lazy().with_columns(base.str.split(".").alias("sentence")).explode("sentence")
    )

    lf = lf.with_columns(
        pl.col("sentence").str.len_chars().cast(pl.Int64).alias("sentence_len")
    )
    lf = lf.filter(pl.col("sentence_len") > 3)

    lf = lf.with_columns(
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .cast(pl.Int64)
        .alias("only_sentence_len"),
        pl.col("sentence")
        .str.split(" ")
        .list.len()
        .cast(pl.Int64)
        .alias("sentence_word_cnt"),
    )
    return lf.collect(streaming=True)


def Sentence_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
        .count()
        .alias("short_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
        .count()
        .alias("mid_sentence_cnt"),
        pl.col("sentence")
        .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
        .count()
        .alias("long_sentence_cnt"),
        *[
            pl.col("sentence")
            .filter(pl.col("only_sentence_len") >= i)
            .count()
            .alias(f"only_sentence_{i}_cnt")
            for i in [40, 60, 80, 100, 120]
        ],
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40)
        )
        .count()
        .alias("short_only_sentence_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 100) & (pl.col("only_sentence_len") > 60)
        )
        .count()
        .alias("mid_only_sentence_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 120) & (pl.col("only_sentence_len") > 100)
        )
        .count()
        .alias("long_only_sentence_cnt"),
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [10, 15, 20, 25]
        ],
        pl.col("sentence")
        .filter(
            (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
        )
        .count()
        .alias("short_sentence_word_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
        )
        .count()
        .alias("mid_sentence_word_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20)
        )
        .count()
        .alias("long_sentence_word_cnt"),
        pl.col("sentence").count().alias("sentence_cnt"),
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

    df = (
        x.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )

    df = df.with_columns(
        *[
            (pl.col(f"sentence_{i}_cnt") / pl.col("sentence_cnt")).alias(
                f"sentence_{i}_cnt_ratio"
            )
            for i in [40, 60, 70, 80, 100, 120, 140]
        ],
        (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "short_sentence_cnt_ratio"
        ),
        (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "mid_sentence_cnt_ratio"
        ),
        (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "long_sentence_cnt_ratio"
        ),
    ).sort("essay_id")

    return df.to_pandas()




## === cell 8
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame, clean_col: str = "full_text_clean") -> pl.DataFrame:
    if clean_col in x.columns:
        base = pl.col(clean_col)
    else:
        base = _polars_preprocess_expr("full_text")

    lf = x.lazy().with_columns(base.str.split(" ").alias("word")).explode("word")
    lf = lf.with_columns(
        pl.col("word").str.len_chars().cast(pl.Int64).alias("word_len")
    )
    lf = lf.filter(pl.col("word_len") > 0)
    return lf.collect(streaming=True)


def Word_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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
        pl.col("word")
        .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
        .count()
        .alias("short_word_cnt"),
        pl.col("word")
        .filter((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
        .count()
        .alias("mid_word_cnt"),
        pl.col("word")
        .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
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

    df = (
        x.lazy()
        .group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .collect(streaming=True)
    )
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
            (pl.col("short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col("mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col("long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
    ).sort("essay_id")

    return df.to_pandas()




## === cell 9
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 3),
    min_df=0.10,
    max_df=0.85,
)

train = train.with_columns(
    _polars_preprocess_expr("full_text").alias("full_text_clean")
)
test = test.with_columns(_polars_preprocess_expr("full_text").alias("full_text_clean"))

train_full_clean = train.get_column("full_text_clean").to_list()
test_full_clean = test.get_column("full_text_clean").to_list()

train_tfid = vectorizer.fit_transform(train_full_clean)  # CSR
train_cnt = vectorizer_cnt.fit_transform(train_full_clean)  # CSR




## === cell 10
def file_exists(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


local_train_cache = f"train_feats_{CFG.VER}.csv"
if file_exists(CFG.LOAD_FEATURES_FROM):
    print("Load train_feats.csv:", CFG.LOAD_FEATURES_FROM)
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)
elif os.path.exists(local_train_cache):
    print("Load cached train features:", local_train_cache)
    train_feats = pd.read_csv(local_train_cache)
else:
    print("Compute train features (no precomputed feature file found).")
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train, clean_col="full_text_clean")
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train, clean_col="full_text_clean")
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values

    train_feats.to_csv(local_train_cache, index=False)
    print("Saved:", local_train_cache)

display(train_feats.head())
print("train_feats shape:", train_feats.shape)



## === cell 11
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


train_num = train_feats[FEATURES].to_numpy(dtype=np.float32, copy=False)
if not train_num.flags["C_CONTIGUOUS"]:
    train_num = np.ascontiguousarray(train_num, dtype=np.float32)
np.nan_to_num(train_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(train_num, 0, 10000, out=train_num)

X_train = sp.hstack([sp.csr_matrix(train_num), train_tfid, train_cnt], format="csr")
y_train = train_feats[TARGET].to_numpy()

FEATURES_NUM = FEATURES




## === cell 12
def train_xgboost_models(X: sp.csr_matrix, y: np.ndarray):
    all_oof = []
    all_true = []

    y_strat = y.astype(np.int32, copy=False)
    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        verbosity=0,
        seed=CFG.SEED,
        tree_method="hist",
        nthread=os.cpu_count() or 4,
    )
    num_boost_round = 1000

    for i, (train_index, valid_index) in enumerate(
        skf.split(np.zeros_like(y_strat), y_strat)
    ):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        X_tr = X[train_index]
        X_va = X[valid_index]
        dtrain = xgb.QuantileDMatrix(X_tr, label=y[train_index])
        dvalid = xgb.QuantileDMatrix(X_va, label=y[valid_index])

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            verbose_eval=50,
        )

        pickle.dump(booster, open(f"XGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = booster.predict(dvalid)
        all_oof.append(oof)
        all_true.append(y[valid_index])

        del X_tr, X_va, dtrain, dvalid, oof, booster
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for XGBoost =", cv)


if file_exists(CFG.LOAD_MODELS_FROM) and os.path.exists(
    os.path.join(CFG.LOAD_MODELS_FROM, f"XGB_v{CFG.VER}_f0.pkl")
):
    print("Found external models; will use them for inference.")
else:
    print("Training XGBoost (no external models found).")
    train_xgboost_models(X_train, y_train)



## === cell 13
model_path0 = None
if file_exists(CFG.LOAD_MODELS_FROM) and os.path.exists(
    os.path.join(CFG.LOAD_MODELS_FROM, f"XGB_v{CFG.VER}_f0.pkl")
):
    model_path0 = os.path.join(CFG.LOAD_MODELS_FROM, f"XGB_v{CFG.VER}_f0.pkl")
elif os.path.exists(f"XGB_v{CFG.VER}_f0.pkl"):
    model_path0 = f"XGB_v{CFG.VER}_f0.pkl"

if model_path0:
    _ = pickle.load(open(model_path0, "rb"))
    print("Model found (skipping feature importance plot for speed).")
else:
    print("No model found to plot feature importance.")



## === cell 14
test_tfid = vectorizer.transform(test_full_clean)  # CSR
test_cnt = vectorizer_cnt.transform(test_full_clean)  # CSR

test_feats_cache = f"test_feats_{CFG.VER}.csv"
if os.path.exists(test_feats_cache):
    print("Load cached test features:", test_feats_cache)
    test_feats = pd.read_csv(test_feats_cache)
else:
    test_feats1 = Paragraph_Features(test)
    test_feats1 = Paragraph_aggregation(test_feats1)
    test_feats2 = Sentence_Features(test, clean_col="full_text_clean")
    test_feats2 = Sentence_aggregation(test_feats2)
    test_feats3 = Word_Features(test, clean_col="full_text_clean")
    test_feats3 = Word_aggregation(test_feats3)

    test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
    test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
    test_feats.to_csv(test_feats_cache, index=False)
    print("Saved:", test_feats_cache)

print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())

categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [
    col for col in test_feats.columns if col not in categorical_columns_test
]

test_num = test_feats[FEATURES_TEST].to_numpy(dtype=np.float32, copy=False)
if not test_num.flags["C_CONTIGUOUS"]:
    test_num = np.ascontiguousarray(test_num, dtype=np.float32)
np.nan_to_num(test_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(test_num, 0, 10000, out=test_num)

X_test = sp.hstack([sp.csr_matrix(test_num), test_tfid, test_cnt], format="csr")

dtest = xgb.DMatrix(X_test, nthread=os.cpu_count() or 4)



## === cell 15
preds = []
for i in range(10):
    print(f"Fold {i+1}")
    if file_exists(CFG.LOAD_MODELS_FROM) and os.path.exists(
        os.path.join(CFG.LOAD_MODELS_FROM, f"XGB_v{CFG.VER}_f{i}.pkl")
    ):
        model_path = os.path.join(CFG.LOAD_MODELS_FROM, f"XGB_v{CFG.VER}_f{i}.pkl")
    else:
        model_path = f"XGB_v{CFG.VER}_f{i}.pkl"

    model = pickle.load(open(model_path, "rb"))
    pred_i = model.predict(dtest)
    preds.append(pred_i)

pred = np.mean(preds, axis=0)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = np.clip(pred, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print("Wrote: submission.csv")

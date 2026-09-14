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

import numpy as np
import pandas as pd
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb

import scipy.sparse as sp

import warnings

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
    pd.read_csv(CFG.BASE_PATH + "train.csv")
    .sort_values(by="essay_id")
    .reset_index(drop=True)
)
df_test = (
    pd.read_csv(CFG.BASE_PATH + "test.csv")
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


def count_misspelled_words(text: str) -> int:
    text = str(text)
    tokens = text.split()
    if not tokens:
        return 0
    good = 0
    for t in tokens:
        tt = _strip_edges_re.sub("", t)
        if not tt:
            continue
        if _word_good_re.fullmatch(tt) is not None:
            good += 1
    nonempty = 0
    for t in tokens:
        tt = _strip_edges_re.sub("", t)
        if tt:
            nonempty += 1
    return nonempty - good




## === cell 6
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

    print("Caculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").str.len_chars().cast(pl.Int64).alias("paragraph_len")
    )

    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("paragraph_misspelled_cnt")
    )

    x = x.with_columns(
        pl.col("paragraph")
        .str.count_matches(",")
        .cast(pl.Int64)
        .alias("paragraph_comma_cnt")
    )

    print("Caculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(str(s).split(".")), return_dtype=pl.Int64)
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(str(s).split(" ")), return_dtype=pl.Int64)
        .alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()




## === cell 7
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(
        pl.col("sentence").str.len_chars().cast(pl.Int64).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(str(s).replace(" ", "")), return_dtype=pl.Int64)
        .alias("only_sentence_len")
    )

    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(str(s).split(" ")), return_dtype=pl.Int64)
        .alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
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


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    print("Preprocess full_text and use spaces to seperate words fro the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(pl.col("word").str.len_chars().cast(pl.Int64).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x: pl.DataFrame) -> pd.DataFrame:
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

train_full_clean = [dataPreprocessing(s) for s in df_train["full_text"].tolist()]
test_full_clean = [dataPreprocessing(s) for s in df_test["full_text"].tolist()]

train_tfid = vectorizer.fit_transform(train_full_clean)  # CSR
train_cnt = vectorizer_cnt.fit_transform(train_full_clean)  # CSR




## === cell 10
def file_exists(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


if file_exists(CFG.LOAD_FEATURES_FROM):
    print("Load train_feats.csv:", CFG.LOAD_FEATURES_FROM)
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)
else:
    print("Compute train features (no precomputed feature file found).")
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values

    out_path = f"train_feats_{CFG.VER}.csv"
    train_feats.to_csv(out_path, index=False)
    print("Saved:", out_path)

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


train_num = np.clip(
    train_feats[FEATURES].fillna(0).to_numpy(dtype=np.float32), 0, 10000
)
X_train = sp.hstack([sp.csr_matrix(train_num), train_tfid, train_cnt], format="csr")
y_train = train_feats[TARGET].to_numpy()

FEATURES_NUM = FEATURES




## === cell 12
def train_xgboost_models(X: sp.csr_matrix, y: np.ndarray):
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        verbosity=0,
        seed=CFG.SEED,
        tree_method="hist",  # faster, equivalent training objective/semantics
    )
    num_boost_round = 1000

    for i, (train_index, valid_index) in enumerate(skf.split(np.zeros_like(y), y)):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        dtrain = xgb.DMatrix(X[train_index], label=y[train_index])
        dvalid = xgb.DMatrix(X[valid_index], label=y[valid_index])

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

        del dtrain, dvalid, oof, booster
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
    model0 = pickle.load(open(model_path0, "rb"))

    try:
        score_map = model0.get_score(importance_type="gain")
        imp = []
        for k, v in score_map.items():
            if k.startswith("f"):
                idx = int(k[1:])
                if idx < len(FEATURES_NUM):
                    imp.append((FEATURES_NUM[idx], v))
        if imp:
            df_importance = pd.DataFrame(
                imp, columns=["features_name", "importance"]
            ).sort_values(by="importance", ascending=False)

            plt.figure(figsize=(12, 6))
            plt.bar(
                x=df_importance.head(30)["features_name"],
                height=df_importance.head(30)["importance"],
                color="pink",
                edgecolor="black",
            )
            plt.title(
                "Distribution of Feature Importance of XGBoost (numeric features)"
            )
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print("No mappable numeric feature importance available to plot.")
    except Exception:
        print("Could not plot feature importance.")
else:
    print("No model found to plot feature importance.")


## === cell 14
test_tfid = vectorizer.transform(test_full_clean)  # CSR
test_cnt = vectorizer_cnt.transform(test_full_clean)  # CSR

test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")

print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())

categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [
    col for col in test_feats.columns if col not in categorical_columns_test
]

test_num = np.clip(
    test_feats[FEATURES_TEST].fillna(0).to_numpy(dtype=np.float32), 0, 10000
)
X_test = sp.hstack([sp.csr_matrix(test_num), test_tfid, test_cnt], format="csr")
dtest = xgb.DMatrix(X_test)


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

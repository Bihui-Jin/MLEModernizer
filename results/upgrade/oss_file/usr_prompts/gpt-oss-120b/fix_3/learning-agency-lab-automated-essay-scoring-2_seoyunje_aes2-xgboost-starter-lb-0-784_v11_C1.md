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

0.7744315806846369

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

import pandas as pd, numpy as np
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For GPU T4x2

import warnings

warnings.filterwarnings("ignore")




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None  # Train models locally
    LOAD_FEATURES_FROM = None  # Build features locally
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 3
def seed_everything():  # To produce similar results in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 4
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
print("Shape of Train: ", df_train.shape)
print(df_train.head())




## === cell 5
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
print(df_test.head())




## === cell 6
train = (
    pl.from_pandas(df_train)
    .with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
    .with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .alias("clean_text")
    )
)
test = (
    pl.from_pandas(df_test)
    .with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
    .with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .alias("clean_text")
    )
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3887521142.py in <cell line: 0>()
      3     pl.from_pandas(df_train)
      4     .with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
----> 5     .with_columns(
      6         pl.col("full_text")
      7         .map_elements(lambda x: dataPreprocessing(x))

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in with_columns(self, *exprs, **named_exprs)
   9803         └─────┴──────┴─────────────┘
   9804         """
-> 9805         return self.lazy().with_columns(*exprs, **named_exprs).collect(_eager=True)
   9806 
   9807     def with_columns_seq(

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ComputeError: NameError: name 'dataPreprocessing' is not defined

## === cell 7
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)  # html -> ''


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub(r"\\s+", " ", x)
    x = re.sub(r'[^\w\s.,;:"' "?!]", "", x)
    x = re.sub("paragraph", "", x)
    x = re.sub(r"\\.+", ".", x)
    x = re.sub(r"\\,+", ",", x)
    x = x.strip()
    return x




## === cell 8
def count_misspelled_words(text):
    return 0




## === cell 9
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(x):
    x = x.explode("paragraph")
    print("Paragraph Preprocessing")
    x = x.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    print("Calculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda p: len(p)).alias("paragraph_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda p: count_misspelled_words(p))
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda p: p.count(","))
        .alias("paragraph_comma_cnt")
    )
    print("Calculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda p: len(p.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda p: len(p.split(" ")))
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
            .alias("short_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
            .count()
            .alias("mid_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
            .count()
            .alias("long_paragraph_cnt")
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
            .alias("short_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 8)
                & (pl.col("paragraph_sentence_cnt") > 4)
            )
            .count()
            .alias("mid_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 10)
                & (pl.col("paragraph_sentence_cnt") > 8)
            )
            .count()
            .alias("long_paragraph_sentence_cnt")
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
            .alias("short_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 90)
                & (pl.col("paragraph_word_cnt") > 40)
            )
            .count()
            .alias("mid_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 120)
                & (pl.col("paragraph_word_cnt") > 90)
            )
            .count()
            .alias("long_paragraph_word_cnt")
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
            .alias("short_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 12)
                & (pl.col("paragraph_misspelled_cnt") > 8)
            )
            .count()
            .alias("mid_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 16)
                & (pl.col("paragraph_misspelled_cnt") > 12)
            )
            .count()
            .alias("long_paragraph_misspelled_cnt")
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
    return df.to_pandas()




## === cell 10
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x):
    print("Preprocess full_text and split into sentences")
    x = x.with_columns(pl.col("clean_text").str.split(".").alias("sentence"))
    x = x.explode("sentence")
    print("Calculate sentence length")
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda s: len(s)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.replace(" ", "")))
        .alias("only_sentence_len")
    )
    print("Count words per sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")))
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
            .alias("short_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias("mid_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
            .count()
            .alias("long_sentence_cnt")
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
            .alias("short_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 100)
                & (pl.col("only_sentence_len") > 60)
            )
            .count()
            .alias("mid_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 120)
                & (pl.col("only_sentence_len") > 100)
            )
            .count()
            .alias("long_only_sentence_cnt")
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
            .alias("short_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
            )
            .count()
            .alias("mid_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20)
            )
            .count()
            .alias("long_sentence_word_cnt")
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




## === cell 11
word_features = ["word_len"]


def Word_Features(x):
    print("Preprocess full_text and split into words")
    x = x.with_columns(pl.col("clean_text").str.split(" ").alias("word"))
    x = x.explode("word")
    print("Calculate word length")
    x = x.with_columns(pl.col("word").map_elements(lambda w: len(w)).alias("word_len"))
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
            .alias("short_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
            .count()
            .alias("mid_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
            .count()
            .alias("long_word_cnt")
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




## === cell 12
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
    max_features=1000,  # limit dimensionality for speed & memory
)

train_tfid = vectorizer.fit_transform(df_train["full_text"])
df_tfid = pd.DataFrame(
    train_tfid.toarray(),
    columns=[f"tfidf_{i}" for i in range(train_tfid.shape[1])],
)
df_tfid["essay_id"] = df_train["essay_id"].values




## === cell 13
vectorizer_cnt = CountVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 4),
    min_df=0.10,
    max_df=0.85,
    max_features=800,  # keep size modest
)

train_cnt = vectorizer_cnt.fit_transform(df_train["full_text"])
df_cnt = pd.DataFrame(
    train_cnt.toarray(),
    columns=[f"cnt_{i}" for i in range(train_cnt.shape[1])],
)
df_cnt["essay_id"] = df_train["essay_id"].values




## === cell 14
print("Generating paragraph features...")
train_feats1 = Paragraph_Features(train)
train_feats1 = Paragraph_aggregation(train_feats1)

print("Generating sentence features...")
train_feats2 = Sentence_Features(train)
train_feats2 = Sentence_aggregation(train_feats2)

print("Generating word features...")
train_feats3 = Word_Features(train)
train_feats3 = Word_aggregation(train_feats3)

train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
train_feats = train_feats.merge(df_tfid, on="essay_id", how="left")
train_feats = train_feats.merge(df_cnt, on="essay_id", how="left")
train_feats["score"] = df_train["score"].values
print("Final training feature shape:", train_feats.shape)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/922596383.py in <cell line: 0>()
      1 print("Generating paragraph features...")
----> 2 train_feats1 = Paragraph_Features(train)
      3 train_feats1 = Paragraph_aggregation(train_feats1)
      4 
      5 print("Generating sentence features...")

NameError: name 'train' is not defined

## === cell 15
display(train_feats.head())




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3973869876.py in <cell line: 0>()
----> 1 display(train_feats.head())
      2 
      3 

NameError: name 'train_feats' is not defined

## === cell 16
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

print("XGBoost Version:", xgb.__version__)




## === cell 17
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/75358094.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 FEATURES = [
      5     col for col in train_feats.columns if col not in categorical_columns + ["score"]

NameError: name 'train_feats' is not defined

## === cell 18
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 19
def xgboost_train():
    all_oof = []
    all_true = []
    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_idx)}, valid size {len(valid_idx)}")
        print("#" * 25)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            n_estimators=1000,
            random_state=CFG.SEED,
            verbosity=0,
        )
        train_x = np.clip(train_feats.loc[train_idx, FEATURES].fillna(0), 0, 10000)
        train_y = train_feats.loc[train_idx, TARGET]
        valid_x = np.clip(train_feats.loc[valid_idx, FEATURES].fillna(0), 0, 10000)
        valid_y = train_feats.loc[valid_idx, TARGET]
        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            early_stopping_rounds=100,
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
    cv_score = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for XGBoost =", cv_score)
    cm = confusion_matrix(
        all_true, np.clip(all_oof, 1, 6).round(), labels=list(range(1, 7))
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(1, 7)))
    disp.plot()
    plt.show()




## === cell 20
print("Training XGBoost models...")
xgboost_train()




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/218341430.py in <cell line: 0>()
      1 print("Training XGBoost models...")
----> 2 xgboost_train()
      3 
      4 

/tmp/ipykernel_55/1868914711.py in xgboost_train()
      4     skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
      5     for i, (train_idx, valid_idx) in enumerate(
----> 6         skf.split(train_feats, train_feats[TARGET])
      7     ):
      8         print("#" * 25)

NameError: name 'train_feats' is not defined

## === cell 21
model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
df_importance = pd.DataFrame(
    {"features_name": FEATURES, "importance": model.feature_importances_}
).sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(
    data=df_importance.head(30),
    x="features_name",
    height="importance",
    color="pink",
    edgecolor="black",
)
plt.title("Top 30 Feature Importances (XGBoost)")
plt.xticks(rotation=90)
plt.show()




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2496018516.py in <cell line: 0>()
----> 1 model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
      2 df_importance = pd.DataFrame(
      3     {"features_name": FEATURES, "importance": model.feature_importances_}
      4 ).sort_values(by="importance", ascending=False)
      5 

FileNotFoundError: [Errno 2] No such file or directory: 'XGB_v1_f0.pkl'

## === cell 22
test_a = test.with_columns(
    pl.col("clean_text").alias("full_text")  # keep column name expected later
)
test_tfid = vectorizer.transform(df_test["full_text"])
df_test_tfid = pd.DataFrame(
    test_tfid.toarray(),
    columns=[f"tfidf_{i}" for i in range(test_tfid.shape[1])],
)
df_test_tfid["essay_id"] = df_test["essay_id"].values




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2068608691.py in <cell line: 0>()
      1 # Use the already cleaned column for TF‑IDF transformation
----> 2 test_a = test.with_columns(
      3     pl.col("clean_text").alias("full_text")  # keep column name expected later
      4 )
      5 test_tfid = vectorizer.transform(df_test["full_text"])

NameError: name 'test' is not defined

## === cell 23
test_b = test.with_columns(pl.col("clean_text").alias("full_text"))
test_cnt = vectorizer_cnt.transform(df_test["full_text"])
df_test_cnt = pd.DataFrame(
    test_cnt.toarray(),
    columns=[f"cnt_{i}" for i in range(test_cnt.shape[1])],
)
df_test_cnt["essay_id"] = df_test["essay_id"].values




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1008309253.py in <cell line: 0>()
----> 1 test_b = test.with_columns(pl.col("clean_text").alias("full_text"))
      2 test_cnt = vectorizer_cnt.transform(df_test["full_text"])
      3 df_test_cnt = pd.DataFrame(
      4     test_cnt.toarray(),
      5     columns=[f"cnt_{i}" for i in range(test_cnt.shape[1])],

NameError: name 'test' is not defined

## === cell 24
print("Generating test paragraph features...")
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)

print("Generating test sentence features...")
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)

print("Generating test word features...")
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1854122555.py in <cell line: 0>()
      1 print("Generating test paragraph features...")
----> 2 test_feats1 = Paragraph_Features(test)
      3 test_feats1 = Paragraph_aggregation(test_feats1)
      4 
      5 print("Generating test sentence features...")

NameError: name 'test' is not defined

## === cell 25
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_tfid, on="essay_id", how="left")
test_feats = test_feats.merge(df_test_cnt, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1062288022.py in <cell line: 0>()
----> 1 test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
      2 test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
      3 test_feats = test_feats.merge(df_test_tfid, on="essay_id", how="left")
      4 test_feats = test_feats.merge(df_test_cnt, on="essay_id", how="left")
      5 print("Shape of test_feats:", test_feats.shape)

NameError: name 'test_feats1' is not defined

## === cell 26
preds = []
categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [
    col for col in test_feats.columns if col not in categorical_columns_test
]

for i in range(10):
    print(f"Fold {i+1}")
    model = pickle.load(open(f"XGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(test_feats[FEATURES_TEST])
    preds.append(pred)

pred = np.mean(preds, axis=0)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2325049647.py in <cell line: 0>()
      1 preds = []
----> 2 categorical_columns_test = test_feats.select_dtypes(
      3     include=["object", "category"]
      4 ).columns.tolist()
      5 FEATURES_TEST = [

NameError: name 'test_feats' is not defined

## === cell 27
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub[TARGET] = pred.clip(1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission saved. Shape:", sub.shape)
display(sub.head())

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2568731914.py in <cell line: 0>()
      1 sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
----> 2 sub[TARGET] = pred.clip(1, 6).round().astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission saved. Shape:", sub.shape)
      5 display(sub.head())

NameError: name 'pred' is not defined

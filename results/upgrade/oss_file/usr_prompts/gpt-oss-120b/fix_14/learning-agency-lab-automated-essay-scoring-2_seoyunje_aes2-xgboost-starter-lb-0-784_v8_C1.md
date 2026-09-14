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

0.7715709745995879

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
import re
from tqdm import tqdm
import pickle

import pandas as pd, numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 3
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 4
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s.,;:\"\'?!]", "", x)
    x = re.sub(r"paragraph", "", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    return x.strip()


df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

train = (
    pl.from_pandas(df_train)
    .with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(str(x)))
        .alias("clean_text")
    )
    .with_columns(pl.col("clean_text").str.split("\n\n").alias("paragraph"))
)

test = (
    pl.from_pandas(df_test)
    .with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(str(x)))
        .alias("clean_text")
    )
    .with_columns(pl.col("clean_text").str.split("\n\n").alias("paragraph"))
)



## === cell 5
try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text):
        return len(spell.unknown(text.split()))

except Exception:

    def count_misspelled_words(text):
        return 0




## === cell 6
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]
sentence_features = ["sentence_len", "sentence_word_cnt"]
word_features = ["word_len"]




## === cell 7
def Paragraph_Features(df):
    df = df.explode("paragraph")
    df = df.with_columns(
        [
            pl.col("paragraph").str.lengths().alias("paragraph_len"),
            pl.col("paragraph")
            .apply(lambda x: count_misspelled_words(x) if isinstance(x, str) else 0)
            .alias("paragraph_misspelled_cnt"),
            pl.col("paragraph").str.count_match(",").alias("paragraph_comma_cnt"),
            pl.col("paragraph")
            .str.split(".")
            .list.len()
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph").str.split().list.len().alias("paragraph_word_cnt"),
        ]
    )
    return df


def Paragraph_aggregation(df):
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
    out = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return out.to_pandas()




## === cell 8
def Sentence_Features(df):
    df = df.with_columns(pl.col("clean_text").str.split(".").alias("sentence"))
    df = df.explode("sentence")
    df = df.with_columns(
        [
            pl.col("sentence").str.lengths().alias("sentence_len"),
            pl.col("sentence")
            .str.replace(" ", "")
            .str.lengths()
            .alias("only_sentence_len"),
            pl.col("sentence").str.split().list.len().alias("sentence_word_cnt"),
        ]
    )
    df = df.filter(pl.col("sentence_len") > 3)
    return df


def Sentence_aggregation(df):
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
    out = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    ratio_exprs = [
        (pl.col(f"sentence_{i}_cnt") / pl.col("sentence_cnt")).alias(
            f"sentence_{i}_cnt_ratio"
        )
        for i in [40, 60, 70, 80, 100, 120, 140]
    ] + [
        (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "short_sentence_cnt_ratio"
        ),
        (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "mid_sentence_cnt_ratio"
        ),
        (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
            "long_sentence_cnt_ratio"
        ),
    ]
    out = out.with_columns(ratio_exprs)
    return out.to_pandas()




## === cell 9
def Word_Features(df):
    df = df.with_columns(pl.col("clean_text").str.split().alias("word"))
    df = df.explode("word")
    df = df.with_columns([pl.col("word").str.lengths().alias("word_len")])
    df = df.filter(pl.col("word_len") > 0)
    return df


def Word_aggregation(df):
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
    out = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    out = out.with_columns(
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
    )
    return out.to_pandas()




## === cell 10
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
train_tfid = vectorizer.fit_transform(df_train["full_text"].astype(str))



## === cell 11
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
train_b = df_train.copy()
train_b["full_text"] = train_b["full_text"].apply(dataPreprocessing)
test_b = df_test.copy()
test_b["full_text"] = test_b["full_text"].apply(dataPreprocessing)

train_cnt = vectorizer_cnt.fit_transform(train_b["full_text"])



## === cell 12
train_feats1 = Paragraph_Features(train)
train_feats1 = Paragraph_aggregation(train_feats1)

train_feats2 = Sentence_Features(train)
train_feats2 = Sentence_aggregation(train_feats2)

train_feats3 = Word_Features(train)
train_feats3 = Word_aggregation(train_feats3)

numeric_train = (
    train_feats1.merge(train_feats2, on="essay_id", how="left")
    .merge(train_feats3, on="essay_id", how="left")
    .sort_values("essay_id")
)
numeric_train = numeric_train.drop(columns=["essay_id"])

numeric_train = numeric_train.replace([np.inf, -np.inf], np.nan).fillna(0)

from scipy.sparse import csr_matrix, hstack

numeric_sparse = csr_matrix(
    np.nan_to_num(
        numeric_train.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
)

X_all = hstack([numeric_sparse, train_tfid, train_cnt]).tocsr()
y_all = df_train["score"].values



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/821117758.py in <cell line: 0>()
----> 1 train_feats1 = Paragraph_Features(train)
      2 train_feats1 = Paragraph_aggregation(train_feats1)
      3 
      4 train_feats2 = Sentence_Features(train)
      5 train_feats2 = Sentence_aggregation(train_feats2)

/tmp/ipykernel_55/3735007452.py in Paragraph_Features(df)
      4     df = df.with_columns(
      5         [
----> 6             pl.col("paragraph").str.lengths().alias("paragraph_len"),
      7             pl.col("paragraph")
      8             .apply(lambda x: count_misspelled_words(x) if isinstance(x, str) else 0)

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 13
TARGET = "score"




## === cell 14
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 15
def train_xgboost():
    all_oof = []
    all_true = []
    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_all, y_all), 1):
        print(f"\n=== Fold {fold} ===")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            n_estimators=1000,
            random_state=CFG.SEED,
            verbosity=0,
            tree_method="hist",
            n_jobs=-1,
        )
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_valid = X_all[valid_idx]
        y_valid = y_all[valid_idx]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=50,
        )
        pickle.dump(model, open(f"XGB_v{CFG.VER}_f{fold-1}.pkl", "wb"))

        oof_pred = model.predict(X_valid)
        all_oof.append(oof_pred)
        all_true.append(y_valid)

        del X_train, y_train, X_valid, y_valid, oof_pred, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)
    score = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print(f"\nOverall CV QWK: {score:.6f}")




## === cell 16
if CFG.LOAD_MODELS_FROM is None:
    print("Starting XGBoost training...")
    train_xgboost()
else:
    print("Skipping training – models would be loaded from external path.")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3558794469.py in <cell line: 0>()
      1 if CFG.LOAD_MODELS_FROM is None:
      2     print("Starting XGBoost training...")
----> 3     train_xgboost()
      4 else:
      5     print("Skipping training – models would be loaded from external path.")

/tmp/ipykernel_55/1831501023.py in train_xgboost()
      3     all_true = []
      4     skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
----> 5     for fold, (train_idx, valid_idx) in enumerate(skf.split(X_all, y_all), 1):
      6         print(f"\n=== Fold {fold} ===")
      7         model = xgb.XGBRegressor(

NameError: name 'X_all' is not defined

## === cell 17
model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
df_importance = pd.DataFrame(
    {
        "features_name": np.arange(X_all.shape[1]),
        "importance": model.feature_importances_,
    }
).sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(
    data=df_importance.head(30),
    x="features_name",
    height="importance",
    color="pink",
    edgecolor="black",
)
plt.title("Top 30 Feature Importances")
plt.xticks(rotation=90)
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4250937362.py in <cell line: 0>()
----> 1 model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
      2 df_importance = pd.DataFrame(
      3     {
      4         "features_name": np.arange(X_all.shape[1]),
      5         "importance": model.feature_importances_,

FileNotFoundError: [Errno 2] No such file or directory: 'XGB_v1_f0.pkl'

## === cell 18
test_tfid = vectorizer.transform(df_test["full_text"].astype(str))
test_cnt = vectorizer_cnt.transform(test_b["full_text"])



## === cell 19
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)

test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)

test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)

numeric_test = (
    test_feats1.merge(test_feats2, on="essay_id", how="left")
    .merge(test_feats3, on="essay_id", how="left")
    .sort_values("essay_id")
)
numeric_test = numeric_test.drop(columns=["essay_id"])

numeric_test = numeric_test.replace([np.inf, -np.inf], np.nan).fillna(0)

numeric_test_sparse = csr_matrix(
    np.nan_to_num(
        numeric_test.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
)

X_test_all = hstack([numeric_test_sparse, test_tfid, test_cnt]).tocsr()
print("Test features shape:", X_test_all.shape)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4113416880.py in <cell line: 0>()
----> 1 test_feats1 = Paragraph_Features(test)
      2 test_feats1 = Paragraph_aggregation(test_feats1)
      3 
      4 test_feats2 = Sentence_Features(test)
      5 test_feats2 = Sentence_aggregation(test_feats2)

/tmp/ipykernel_55/3735007452.py in Paragraph_Features(df)
      4     df = df.with_columns(
      5         [
----> 6             pl.col("paragraph").str.lengths().alias("paragraph_len"),
      7             pl.col("paragraph")
      8             .apply(lambda x: count_misspelled_words(x) if isinstance(x, str) else 0)

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 20
preds = []
for i in range(10):
    model_path = f"XGB_v{CFG.VER}_f{i}.pkl"
    model = pickle.load(open(model_path, "rb"))
    preds.append(model.predict(X_test_all))
pred = np.mean(preds, axis=0)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3217546733.py in <cell line: 0>()
      2 for i in range(10):
      3     model_path = f"XGB_v{CFG.VER}_f{i}.pkl"
----> 4     model = pickle.load(open(model_path, "rb"))
      5     preds.append(model.predict(X_test_all))
      6 pred = np.mean(preds, axis=0)

FileNotFoundError: [Errno 2] No such file or directory: 'XGB_v1_f0.pkl'

## === cell 21
sub = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].values,
        TARGET: np.clip(pred, 1, 6).round().astype(int),
    }
)
sub.to_csv("submission.csv", index=False)
print("Submission saved – shape:", sub.shape)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4102821382.py in <cell line: 0>()
      2     {
      3         "essay_id": df_test["essay_id"].values,
----> 4         TARGET: np.clip(pred, 1, 6).round().astype(int),
      5     }
      6 )

NameError: name 'pred' is not defined

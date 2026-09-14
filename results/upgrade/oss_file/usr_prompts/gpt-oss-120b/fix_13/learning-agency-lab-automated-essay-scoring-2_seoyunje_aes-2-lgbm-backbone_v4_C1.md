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
lightgbm==4.6.0
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

0.8122359916145172

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
import warnings
import pickle

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from joblib import Parallel, delayed

import lightgbm as lgb
from lightgbm import early_stopping

warnings.filterwarnings("ignore")


class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 1
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 2
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 3
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
print("Shape of Train: ", df_train.shape)
display(df_train.head())

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
display(df_test.head())




## === cell 4
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()




## === cell 5
pass




## === cell 6
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r'[^\w\s.,;:"\'?!]', "", x)
    x = re.sub("paragraph", "", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split("\n\n").alias("paragraph"),
    pl.col("full_text")
    .map_elements(lambda x: dataPreprocessing(x))
    .alias("clean_text"),
)

test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split("\n\n").alias("paragraph"),
    pl.col("full_text")
    .map_elements(lambda x: dataPreprocessing(x))
    .alias("clean_text"),
)

try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text):
        return len(spell.unknown(text.split()))

except Exception:

    def count_misspelled_words(text):
        return 0




## === cell 7
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(x):
    x = x.explode("paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(dataPreprocessing).alias("paragraph_clean")
    )
    x = x.with_columns(
        pl.col("paragraph_clean").str.lengths().alias("paragraph_len"),
        pl.col("paragraph_clean")
        .apply(count_misspelled_words)
        .alias("paragraph_misspelled_cnt"),
        pl.col("paragraph_clean").str.count_matches(",").alias("paragraph_comma_cnt"),
        pl.col("paragraph_clean")
        .str.split(".")
        .arr.lengths()
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph_clean")
        .str.split(" ")
        .arr.lengths()
        .alias("paragraph_word_cnt"),
    )
    return x


def Paragraph_aggregation(x):
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




## === cell 8
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x):
    x = x.with_columns(pl.col("clean_text").str.split(".").alias("sentence"))
    x = x.explode("sentence")
    x = x.with_columns(
        pl.col("sentence").str.lengths().alias("sentence_len"),
        pl.col("sentence")
        .str.replace(" ", "")
        .str.lengths()
        .alias("only_sentence_len"),
        pl.col("sentence").str.split(" ").arr.lengths().alias("sentence_word_cnt"),
    )
    x = x.filter(pl.col("sentence_len") > 3)
    return x


def Sentence_aggregation(x):
    aggs = [pl.col("sentence").count().alias("sentence_cnt")]
    for i in [40, 60, 70, 80, 100, 120, 140]:
        aggs.append(
            ((pl.col("sentence_len") >= i).cast(pl.Int64).sum()).alias(
                f"sentence_{i}_cnt"
            )
        )
    for i in [10, 20, 30]:
        aggs.append(
            ((pl.col("sentence_len") <= i).cast(pl.Int64).sum()).alias(
                f"sentence_{i}_cnt_v2"
            )
        )
    aggs.append(
        ((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
        .cast(pl.Int64)
        .sum()
        .alias("short_sentence_cnt")
    )
    aggs.append(
        ((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
        .cast(pl.Int64)
        .sum()
        .alias("mid_sentence_cnt")
    )
    aggs.append(
        ((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
        .cast(pl.Int64)
        .sum()
        .alias("long_sentence_cnt")
    )
    for i in [40, 60, 80, 100, 120]:
        aggs.append(
            ((pl.col("only_sentence_len") >= i).cast(pl.Int64).sum()).alias(
                f"only_sentence_{i}_cnt"
            )
        )
    aggs.append(
        ((pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40))
        .cast(pl.Int64)
        .sum()
        .alias("short_only_sentence_cnt")
    )
    aggs.append(
        ((pl.col("only_sentence_len") <= 100) & (pl.col("only_sentence_len") > 60))
        .cast(pl.Int64)
        .sum()
        .alias("mid_only_sentence_cnt")
    )
    aggs.append(
        ((pl.col("only_sentence_len") <= 120) & (pl.col("only_sentence_len") > 100))
        .cast(pl.Int64)
        .sum()
        .alias("long_only_sentence_cnt")
    )
    for i in [10, 15, 20, 25]:
        aggs.append(
            ((pl.col("sentence_word_cnt") >= i).cast(pl.Int64).sum()).alias(
                f"sentence_word_{i}_cnt"
            )
        )
    aggs.append(
        ((pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10))
        .cast(pl.Int64)
        .sum()
        .alias("short_sentence_word_cnt")
    )
    aggs.append(
        ((pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15))
        .cast(pl.Int64)
        .sum()
        .alias("mid_sentence_word_cnt")
    )
    aggs.append(
        ((pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20))
        .cast(pl.Int64)
        .sum()
        .alias("long_sentence_word_cnt")
    )
    for feat in sentence_features:
        aggs.extend(
            [
                pl.col(feat).max().alias(f"{feat}_max"),
                pl.col(feat).mean().alias(f"{feat}_mean"),
                pl.col(feat).min().alias(f"{feat}_min"),
                pl.col(feat).std().alias(f"{feat}_std"),
                pl.col(feat).sum().alias(f"{feat}_sum"),
                pl.col(feat).quantile(0.25).alias(f"{feat}_q1"),
                pl.col(feat).quantile(0.75).alias(f"{feat}_q3"),
            ]
        )
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




## === cell 9
word_features = ["word_len"]


def Word_Features(x):
    x = x.with_columns(pl.col("clean_text").str.split(" ").alias("word"))
    x = x.explode("word")
    x = x.with_columns(pl.col("word").str.lengths().alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x):
    aggs = [pl.col("word").count().alias("word_cnt")]
    for i in [3, 4, 5, 6, 7, 8, 10]:
        aggs.append(
            ((pl.col("word_len") >= i).cast(pl.Int64).sum()).alias(f"word_{i}_cnt")
        )
    for i in [1, 2, 3]:
        aggs.append(
            ((pl.col("word_len") <= i).cast(pl.Int64).sum()).alias(f"word_{i}_cnt_v2")
        )
    aggs.append(
        ((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
        .cast(pl.Int64)
        .sum()
        .alias("short_word_cnt")
    )
    aggs.append(
        ((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
        .cast(pl.Int64)
        .sum()
        .alias("mid_word_cnt")
    )
    aggs.append(
        ((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
        .cast(pl.Int64)
        .sum()
        .alias("long_word_cnt")
    )
    for feat in word_features:
        aggs.extend(
            [
                pl.col(feat).max().alias(f"{feat}_max"),
                pl.col(feat).mean().alias(f"{feat}_mean"),
                pl.col(feat).min().alias(f"{feat}_min"),
                pl.col(feat).std().alias(f"{feat}_std"),
                pl.col(feat).sum().alias(f"{feat}_sum"),
                pl.col(feat).quantile(0.25).alias(f"{feat}_q1"),
                pl.col(feat).quantile(0.75).alias(f"{feat}_q3"),
            ]
        )
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




## === cell 10
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

train_a = train.select(pl.col("clean_text").alias("full_text"))
train_tfid = vectorizer.fit_transform([i for i in train_a["full_text"]])
df = pd.DataFrame.sparse.from_spmatrix(train_tfid)
df.columns = [f"tfidf_{i}" for i in range(df.shape[1])]
df["essay_id"] = df_train["essay_id"]




## === cell 11
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

train_b = train.select(pl.col("clean_text").alias("full_text"))
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b["full_text"]])
df2 = pd.DataFrame.sparse.from_spmatrix(train_cnt)
df2.columns = [f"cnt_{i}" for i in range(df2.shape[1])]
df2["essay_id"] = df_train["essay_id"]




## === cell 12
train_feats1 = Paragraph_Features(train)
train_feats1 = Paragraph_aggregation(train_feats1)
train_feats2 = Sentence_Features(train)
train_feats2 = Sentence_aggregation(train_feats2)
train_feats3 = Word_Features(train)
train_feats3 = Word_aggregation(train_feats3)

train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
train_feats = train_feats.merge(df, on="essay_id", how="left")
train_feats = train_feats.merge(df2, on="essay_id", how="left")
train_feats["score"] = df_train["score"].values




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3522516273.py in <cell line: 0>()
----> 1 train_feats1 = Paragraph_Features(train)
      2 train_feats1 = Paragraph_aggregation(train_feats1)
      3 train_feats2 = Sentence_Features(train)
      4 train_feats2 = Sentence_aggregation(train_feats2)
      5 train_feats3 = Word_Features(train)

/tmp/ipykernel_55/2792212505.py in Paragraph_Features(x)
     15     x = x.with_columns(
     16         # use .str.lengths() instead of the unavailable .str.n_chars()
---> 17         pl.col("paragraph_clean").str.lengths().alias("paragraph_len"),
     18         pl.col("paragraph_clean")
     19         .apply(count_misspelled_words)

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 13
print("Saving computed training features...")
train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/524941347.py in <cell line: 0>()
      1 print("Saving computed training features...")
----> 2 train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      3 
      4 

NameError: name 'train_feats' is not defined

## === cell 14
display(train_feats.head())




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3973869876.py in <cell line: 0>()
----> 1 display(train_feats.head())
      2 
      3 

NameError: name 'train_feats' is not defined

## === cell 15
print("LightGBM Version: ", lgb.__version__)




## === cell 16
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


a = 2.948  # shift used in the original notebook




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
def _train_single_fold(i, train_index, valid_index):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
    print("#" * 25)

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        colsample_bytree=0.8,
        max_depth=5,
        num_leaves=10,
        reg_alpha=0.2,
        reg_lambda=0.8,
        n_estimators=1024,
        random_state=CFG.SEED,
        verbosity=-1,
    )

    train_x = np.clip(train_feats.loc[train_index, FEATURES].fillna(0), 0, 10000)
    train_y = train_feats.loc[train_index, TARGET] - a
    valid_x = np.clip(train_feats.loc[valid_index, FEATURES].fillna(0), 0, 10000)
    valid_y = train_feats.loc[valid_index, TARGET] - a

    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=[early_stopping(stopping_rounds=100)],
    )

    pickle.dump(model, open(f"LGB_v{CFG.VER}_f{i}.pkl", "wb"))

    oof = model.predict(valid_x, num_iteration=model.best_iteration_)
    del train_x, train_y, valid_x, valid_y, oof, model
    clean_memory()
    return (oof + a, valid_y.values + a)


def lightgbm_training():
    skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
    splits = list(skf.split(train_feats, train_feats[TARGET]))

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_train_single_fold)(i, tr_idx, val_idx)
        for i, (tr_idx, val_idx) in enumerate(splits)
    )

    all_oof = np.concatenate([r[0] for r in results])
    all_true = np.concatenate([r[1] for r in results])

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for LightGBM = ", cv)




## === cell 19
print("Training LightGBM models...")
lightgbm_training()




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2383630424.py in <cell line: 0>()
      1 print("Training LightGBM models...")
----> 2 lightgbm_training()
      3 
      4 

/tmp/ipykernel_55/3637439851.py in lightgbm_training()
     41 def lightgbm_training():
     42     skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
---> 43     splits = list(skf.split(train_feats, train_feats[TARGET]))
     44 
     45     results = Parallel(n_jobs=-1, backend="loky")(

NameError: name 'train_feats' is not defined

## === cell 20
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)

test_a = test.select(pl.col("clean_text").alias("full_text"))
test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
df3 = pd.DataFrame.sparse.from_spmatrix(test_tfid)
df3.columns = [f"tfidf_{i}" for i in range(df3.shape[1])]
df3["essay_id"] = df_test["essay_id"]

test_b = test.select(pl.col("clean_text").alias("full_text"))
test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
df4 = pd.DataFrame.sparse.from_spmatrix(test_cnt)
df4.columns = [f"cnt_{i}" for i in range(df4.shape[1])]
df4["essay_id"] = df_test["essay_id"]

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df3, on="essay_id", how="left")
test_feats = test_feats.merge(df4, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2753104679.py in <cell line: 0>()
----> 1 test_feats1 = Paragraph_Features(test)
      2 test_feats1 = Paragraph_aggregation(test_feats1)
      3 test_feats2 = Sentence_Features(test)
      4 test_feats2 = Sentence_aggregation(test_feats2)
      5 test_feats3 = Word_Features(test)

/tmp/ipykernel_55/2792212505.py in Paragraph_Features(x)
     15     x = x.with_columns(
     16         # use .str.lengths() instead of the unavailable .str.n_chars()
---> 17         pl.col("paragraph_clean").str.lengths().alias("paragraph_len"),
     18         pl.col("paragraph_clean")
     19         .apply(count_misspelled_words)

AttributeError: 'ExprStringNameSpace' object has no attribute 'lengths'

## === cell 21
categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [
    col for col in test_feats.columns if col not in categorical_columns_test
]

preds = []
for i in range(15):
    print(f"Fold {i+1} prediction")
    model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(test_feats[FEATURES_TEST]) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/696226433.py in <cell line: 0>()
----> 1 categorical_columns_test = test_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 FEATURES_TEST = [
      5     col for col in test_feats.columns if col not in categorical_columns_test

NameError: name 'test_feats' is not defined

## === cell 22
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = pred1.clip(1, 6).round()
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3068440612.py in <cell line: 0>()
      1 sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
----> 2 sub["score"] = pred1.clip(1, 6).round()
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 display(sub.head())

NameError: name 'pred1' is not defined

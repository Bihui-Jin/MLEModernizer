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

import pandas as pd, numpy as np
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import VotingRegressor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
def seed_everything():  # To produce similar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 4
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")

print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 5
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()



## === cell 6
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 7
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData




## === cell 8
def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)  # html -> ''


def dataPreprocessing(x):

    x = x.lower()
    x = removeHTML(x)

    x = re.sub("@\w+", "", x)

    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)

    x = re.sub("http\w+", "", x)

    x = re.sub(r"\s+", " ", x)
    x = re.sub(r'[^\w\s.,;:"' "?!]", "", x)
    x = re.sub("paragraph", "", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x




## === cell 9
try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text):
        return len(spell.unknown(text.split()))

except Exception:

    def count_misspelled_words(text):
        return 0




## === cell 10
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
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: count_misspelled_words(x))
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: x.count(","))
        .alias("paragraph_comma_cnt")
    )
    print("Calculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
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




## === cell 11
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x):
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")

    print("Calculate the length of a sentence")
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.replace(" ", "")))
        .alias("only_sentence_len")
    )
    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x):
    print("Aggregation")
    aggs = []

    aggs.append(pl.col("sentence").count().alias("sentence_cnt"))

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
        (
            ((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
            .cast(pl.Int64)
            .sum()
        ).alias("short_sentence_cnt")
    )
    aggs.append(
        (
            ((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .cast(pl.Int64)
            .sum()
        ).alias("mid_sentence_cnt")
    )
    aggs.append(
        (
            ((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
            .cast(pl.Int64)
            .sum()
        ).alias("long_sentence_cnt")
    )

    for i in [40, 60, 80, 100, 120]:
        aggs.append(
            ((pl.col("only_sentence_len") >= i).cast(pl.Int64).sum()).alias(
                f"only_sentence_{i}_cnt"
            )
        )

    aggs.append(
        (
            ((pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40))
            .cast(pl.Int64)
            .sum()
        ).alias("short_only_sentence_cnt")
    )
    aggs.append(
        (
            ((pl.col("only_sentence_len") <= 100) & (pl.col("only_sentence_len") > 60))
            .cast(pl.Int64)
            .sum()
        ).alias("mid_only_sentence_cnt")
    )
    aggs.append(
        (
            ((pl.col("only_sentence_len") <= 120) & (pl.col("only_sentence_len") > 100))
            .cast(pl.Int64)
            .sum()
        ).alias("long_only_sentence_cnt")
    )

    for i in [10, 15, 20, 25]:
        aggs.append(
            ((pl.col("sentence_word_cnt") >= i).cast(pl.Int64).sum()).alias(
                f"sentence_word_{i}_cnt"
            )
        )

    aggs.append(
        (
            ((pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10))
            .cast(pl.Int64)
            .sum()
        ).alias("short_sentence_word_cnt")
    )
    aggs.append(
        (
            ((pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15))
            .cast(pl.Int64)
            .sum()
        ).alias("mid_sentence_word_cnt")
    )
    aggs.append(
        (
            ((pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20))
            .cast(pl.Int64)
            .sum()
        ).alias("long_sentence_word_cnt")
    )

    for feat in sentence_features:
        aggs.append(pl.col(feat).max().alias(f"{feat}_max"))
        aggs.append(pl.col(feat).mean().alias(f"{feat}_mean"))
        aggs.append(pl.col(feat).min().alias(f"{feat}_min"))
        aggs.append(pl.col(feat).std().alias(f"{feat}_std"))
        aggs.append(pl.col(feat).sum().alias(f"{feat}_sum"))
        aggs.append(pl.col(feat).quantile(0.25).alias(f"{feat}_q1"))
        aggs.append(pl.col(feat).quantile(0.75).alias(f"{feat}_q3"))

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

    df = df.to_pandas()  # polars -> pandas
    return df




## === cell 12
word_features = [
    "word_len",
]


def Word_Features(x):
    print("Preprocess full_text and use spaces to separate words from the text")

    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")

    print("Calculate the length of a word")
    x = x.with_columns(pl.col("word").map_elements(lambda x: len(x)).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)

    return x


def Word_aggregation(x):

    print("Aggregation")
    aggs = []

    aggs.append(pl.col("word").count().alias("word_cnt"))

    for i in [3, 4, 5, 6, 7, 8, 10]:
        aggs.append(
            ((pl.col("word_len") >= i).cast(pl.Int64).sum()).alias(f"word_{i}_cnt")
        )

    for i in [1, 2, 3]:
        aggs.append(
            ((pl.col("word_len") <= i).cast(pl.Int64).sum()).alias(f"word_{i}_cnt_v2")
        )

    aggs.append(
        (
            ((pl.col("word_len") <= 4) & (pl.col("word_len") > 2)).cast(pl.Int64).sum()
        ).alias("short_word_cnt")
    )
    aggs.append(
        (
            ((pl.col("word_len") <= 6) & (pl.col("word_len") > 4)).cast(pl.Int64).sum()
        ).alias("mid_word_cnt")
    )
    aggs.append(
        (
            ((pl.col("word_len") <= 10) & (pl.col("word_len") > 6)).cast(pl.Int64).sum()
        ).alias("long_word_cnt")
    )

    for feat in word_features:
        aggs.append(pl.col(feat).max().alias(f"{feat}_max"))
        aggs.append(pl.col(feat).mean().alias(f"{feat}_mean"))
        aggs.append(pl.col(feat).min().alias(f"{feat}_min"))
        aggs.append(pl.col(feat).std().alias(f"{feat}_std"))
        aggs.append(pl.col(feat).sum().alias(f"{feat}_sum"))
        aggs.append(pl.col(feat).quantile(0.25).alias(f"{feat}_q1"))
        aggs.append(pl.col(feat).quantile(0.75).alias(f"{feat}_q3"))

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

    df = df.to_pandas()  # polars -> pandas
    return df




## === cell 13
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

train_a = train.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
train_tfid = vectorizer.fit_transform([i for i in train_a["full_text"]])

dense_matrix = train_tfid.toarray()
df = pd.DataFrame(dense_matrix)
df.columns = [f"tfidf_{i}" for i in range(len(df.columns))]
df["essay_id"] = df_train["essay_id"]



## === cell 14
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


train_b = train.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b["full_text"]])

dense_matrix2 = train_cnt.toarray()
df2 = pd.DataFrame(dense_matrix2)
df2.columns = [f"cnt_{i}" for i in range(len(df2.columns))]
df2["essay_id"] = df_train["essay_id"]



## === cell 15
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



## === cell 16
print("Saving computed training features...")
train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)



## === cell 17
display(train_feats.head())



## === cell 18
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping

print("LightGBM Version: ", lgb.__version__)




## === cell 19
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


a = 2.948  # shift used in the original notebook



## === cell 20
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## === cell 21
def lightgbm_training():
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        skf.split(train_feats, train_feats[TARGET])
    ):
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
        all_oof.append(oof + a)
        all_true.append(valid_y.values + a)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for LightGBM = ", cv)




## === cell 22
print("Training LightGBM models...")
lightgbm_training()



## === cell 23
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)

test_a = test.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
df3 = pd.DataFrame(test_tfid.toarray())
df3.columns = [f"tfidf_{i}" for i in range(len(df3.columns))]
df3["essay_id"] = df_test["essay_id"]

test_b = test.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
df4 = pd.DataFrame(test_cnt.toarray())
df4.columns = [f"cnt_{i}" for i in range(len(df4.columns))]
df4["essay_id"] = df_test["essay_id"]

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df3, on="essay_id", how="left")
test_feats = test_feats.merge(df4, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())



## === cell 24
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



## === cell 25
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pred1.clip(1, 6).round()
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())

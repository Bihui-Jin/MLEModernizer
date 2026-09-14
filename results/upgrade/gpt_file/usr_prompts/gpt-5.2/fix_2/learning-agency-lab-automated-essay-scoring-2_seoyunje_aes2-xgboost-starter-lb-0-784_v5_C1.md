# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the missing external-input dependency by auto-falling back to local feature generation and local model training when the specified `/kaggle/input/aes2-xgboost-starter/` files are not present. I also remove notebook-only constructs that cause runtime errors in script execution (e.g., `display`, `%%time`) while preserving the model/feature logic. To ensure train/test feature columns align, I use the same training `FEATURES` list at inference time and add any missing columns in test with zeros (score-neutral but prevents shape/key errors). Finally, I ensure a valid `submission.csv` is always written with columns `essay_id,score`.'

# 9. Code solution

## === cell 0
def display(x):
    try:
        import pandas as _pd

        if isinstance(x, _pd.DataFrame):
            print(x.head())
        else:
            print(x)
    except Exception:
        print(x)




## === cell 1
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




## === cell 2
def _exists(path: str) -> bool:
    return path is not None and isinstance(path, str) and os.path.exists(path)




## === cell 3
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-xgboost-starter/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-xgboost-starter/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 4
if not os.path.exists(os.path.join(CFG.BASE_PATH, "train.csv")):
    alt = "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/"
    if os.path.exists(os.path.join(alt, "train.csv")):
        CFG.BASE_PATH = alt
    else:
        alt2 = "/kaggle/data/"
        if os.path.exists(os.path.join(alt2, "train.csv")):
            CFG.BASE_PATH = alt2



## === cell 5
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()



## === cell 6
if not _exists(CFG.LOAD_FEATURES_FROM):
    CFG.LOAD_FEATURES_FROM = None
if not _exists(CFG.LOAD_MODELS_FROM):
    CFG.LOAD_MODELS_FROM = None
else:
    if not CFG.LOAD_MODELS_FROM.endswith("/"):
        CFG.LOAD_MODELS_FROM += "/"




## === cell 7
def seed_everything():  # To proudce simliar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 8
plt.switch_backend("Agg")



## === cell 9
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")

print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 10
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")

print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 11
train = pl.read_csv(os.path.join(CFG.BASE_PATH, "train.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.read_csv(os.path.join(CFG.BASE_PATH, "test.csv")).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData



## === cell 12
df_train["full_text"] = df_train["full_text"].fillna("").astype(str)
df_test["full_text"] = df_test["full_text"].fillna("").astype(str)




## === cell 13
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




## === cell 14
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Features(x):
    x = x.explode("paragraph")

    print("Paragraph Preprocessing")
    x = x.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))

    print("Caculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    print("Caculate the number of sentences and words in each paragraph")
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
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
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
            for i in [30, 60, 90, 120]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 60)
                & (pl.col("paragraph_word_cnt") > 30)
            )
            .count()
            .alias(f"short_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 90)
                & (pl.col("paragraph_word_cnt") > 60)
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
        *[pl.col("paragraph").count().alias("paragraph_cnt")],
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 15
sentence_features = [
    "sentence_len",
    "sentence_word_cnt",
    "sentence_len_space_ratio",
    "sentence_word_space_ratio",
]


def Sentence_Features(x):
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: x.count(" "))
        .alias("sentence_space_cnt")
    )
    x = x.filter(pl.col("sentence_space_cnt") > 0)
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        (pl.col("sentence_len") / pl.col("sentence_space_cnt")).alias(
            "sentence_len_space_ratio"
        )
    )

    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    x = x.with_columns(
        (pl.col("sentence_word_cnt") / pl.col("sentence_space_cnt")).alias(
            "sentence_word_space_ratio"
        )
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
            for i in [30, 40, 50, 60, 70, 80, 100, 150]
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
            .filter((pl.col("sentence_len") <= 50) & (pl.col("sentence_len") > 30))
            .count()
            .alias(f"short_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 50))
            .count()
            .alias(f"mid_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias(f"long_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [5, 10, 15, 20]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 10) & (pl.col("sentence_word_cnt") > 5)
            )
            .count()
            .alias(f"short_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
            )
            .count()
            .alias(f"mid_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
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
    ]

    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")

    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 16
word_features = [
    "word_len",
]


def Word_Features(x):
    print("Preprocess full_text and use spaces to seperate words from the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")

    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(pl.col("word").map_elements(lambda x: len(x)).alias("word_len"))
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
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            pl.col("word")
            .filter(pl.col("word_len") <= i)
            .count()
            .alias(f"word_{i}_cnt_v2")
            for i in [2, 3]
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
            .count()
            .alias(f"short_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 7) & (pl.col("word_len") > 4))
            .count()
            .alias(f"mid_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 7))
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
            (pl.col(f"word_{i}_cnt") / pl.col(f"word_2_cnt_v2")).alias(
                f"word_2_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / pl.col(f"word_3_cnt_v2")).alias(
                f"word_3_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 8, 10, 15]
        ],
        *[
            (pl.col(f"short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col(f"mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
        *[
            (pl.col(f"long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [2, 3]
        ],
    ).sort("essay_id")
    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 17
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

train_tfid = vectorizer.fit_transform([i for i in train["full_text"]])
dense_matrix = train_tfid.toarray()

df = pd.DataFrame(dense_matrix)
df.columns = [f"tfidf_{i}" for i in range(len(df.columns))]
df["essay_id"] = df_train["essay_id"]



## === cell 18
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

train_cnt = vectorizer_cnt.fit_transform([i for i in train["full_text"]])
dense_matrix2 = train_cnt.toarray()

df2 = pd.DataFrame(dense_matrix2)
df2.columns = [f"cnt_{i}" for i in range(len(df2.columns))]
df2["essay_id"] = df_train["essay_id"]



## === cell 19
if CFG.LOAD_FEATURES_FROM is None:

    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats = train_feats.merge(df, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = None



## === cell 20
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## === cell 21
display(train_feats.head())



## === cell 22
for c in train_feats.columns:
    if c not in ["essay_id", "score"] and train_feats[c].dtype == "object":
        train_feats[c] = pd.to_numeric(train_feats[c], errors="coerce")



## === cell 23
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb

print("XGBoost Version: ", xgb.__version__)



## === cell 24
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## === cell 25
def quadratic_weighted_kappa(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk




## === cell 26
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




## === cell 27
if CFG.LOAD_MODELS_FROM is None:
    print("Training XGBoost")
    xgboost()
else:
    print("Using pre-trained models from:", CFG.LOAD_MODELS_FROM)



## === cell 28
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



## === cell 29
test_tfid = vectorizer.transform([i for i in test["full_text"]])
dense_matrix = test_tfid.toarray()
df3 = pd.DataFrame(dense_matrix)
tfid_columns = [f"tfidf_{i}" for i in range(len(df3.columns))]
df3.columns = tfid_columns
df3["essay_id"] = df_test["essay_id"]



## === cell 30
test_cnt = vectorizer_cnt.transform([i for i in test["full_text"]])
dense_matrix = test_cnt.toarray()
df4 = pd.DataFrame(dense_matrix)
cnt_columns = [f"cnt_{i}" for i in range(len(df4.columns))]
df4.columns = cnt_columns
df4["essay_id"] = df_test["essay_id"]



## === cell 31
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)



## === cell 32
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df3, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())



## === cell 33
for c in test_feats.columns:
    if c != "essay_id" and test_feats[c].dtype == "object":
        test_feats[c] = pd.to_numeric(test_feats[c], errors="coerce")

for col in FEATURES:
    if col not in test_feats.columns:
        test_feats[col] = 0

test_x = np.clip(test_feats[FEATURES].fillna(0), 0, 10000)

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



## === cell 34
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub[TARGET] = np.clip(pred, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print("Wrote: submission.csv")

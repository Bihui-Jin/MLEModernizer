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
import time
import string
import re
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping

from scipy import sparse

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-lgbm/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-lgbm/train_feats_1.csv"
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


clean_memory()




## === cell 3
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 4
if not os.path.exists(os.path.join(CFG.BASE_PATH, "train.csv")):
    CFG.BASE_PATH = "/kaggle/input/"

df_train = pd.read_csv(os.path.join(CFG.BASE_PATH, "train.csv")).sort_values(
    by="essay_id"
)
df_test = pd.read_csv(os.path.join(CFG.BASE_PATH, "test.csv")).sort_values(
    by="essay_id"
)

print("Shape of Train:", df_train.shape)
print(df_train.head())
print("Shape of Test:", df_test.shape)
print(df_test.head())



## === cell 5
DO_PLOTS = False
if DO_PLOTS:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df_train["score"])
    plt.title("Distribution of Score")
    plt.xlabel("Score of Essay")
    plt.ylabel("Frequency")
    plt.show()



## === cell 6
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema
schema_test = test.schema




## === cell 7
def removeHTML(x: str) -> str:
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x: str) -> str:
    x = (x if isinstance(x, str) else "").lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s.,;:\"''?!]", "", x)
    x = re.sub("paragraph", "", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x




## === cell 8
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def count_misspelled_words(text: str) -> int:
    """
    Lightweight, deterministic misspelling proxy:
    counts tokens that are likely 'non-standard' due to unusual patterns.
    This preserves the feature pipeline shape without external dependencies.
    """
    text = text if isinstance(text, str) else ""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0

    cnt = 0
    for w in words:
        if len(w) >= 18:
            cnt += 1
            continue
        if re.search(r"(.)\1\1", w):  # 3 repeated chars
            cnt += 1
            continue
        if w.count("'") > 1:
            cnt += 1
            continue
        if not re.search(r"[aeiou]", w) and len(w) >= 6:
            cnt += 1
            continue
    return cnt




## === cell 9
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(
    x: pl.DataFrame, paragraph_col: str = "paragraph"
) -> pl.DataFrame:
    x = x.explode(paragraph_col)

    x = x.with_columns(pl.col(paragraph_col).alias("paragraph"))

    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda s: len(s)).alias("paragraph_len")
    )

    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(count_misspelled_words)
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: s.count(","))
        .alias("paragraph_comma_cnt")
    )

    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("paragraph_word_cnt"),
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


def Sentence_Features(
    x: pl.DataFrame, full_text_col: str = "full_text"
) -> pl.DataFrame:
    x = x.with_columns(pl.col(full_text_col).str.split(".").alias("sentence"))
    x = x.explode("sentence")

    x = x.with_columns(
        pl.col("sentence").map_elements(lambda s: len(s)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.replace(" ", "")))
        .alias("only_sentence_len")
    )

    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("sentence_word_cnt")
    )
    return x


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


def Word_Features(x: pl.DataFrame, full_text_col: str = "full_text") -> pl.DataFrame:
    x = x.with_columns(pl.col(full_text_col).str.split(" ").alias("word"))
    x = x.explode("word")

    x = x.with_columns(pl.col("word").map_elements(lambda s: len(s)).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


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
    eps = 1e-6
    df = df.with_columns(
        *[
            (pl.col(f"word_{i}_cnt") / (pl.col("word_cnt") + eps)).alias(
                f"word_{i}_cnt_ratio"
            )
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"word_{i}_cnt_v2") / (pl.col("word_cnt") + eps)).alias(
                f"word_{i}_cnt_v2_ratio"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / (pl.col("word_2_cnt_v2") + eps)).alias(
                f"word_{i}_pre2_ratio"
            )
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col(f"word_{i}_cnt") / (pl.col("word_3_cnt_v2") + eps)).alias(
                f"word_{i}_pre3_ratio"
            )
            for i in [3, 4, 5, 6, 7, 8, 10]
        ],
        *[
            (pl.col("short_word_cnt") / (pl.col(f"word_{i}_cnt_v2") + eps)).alias(
                f"short_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col("mid_word_cnt") / (pl.col(f"word_{i}_cnt_v2") + eps)).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col("long_word_cnt") / (pl.col(f"word_{i}_cnt_v2") + eps)).alias(
                f"long_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
    ).sort("essay_id")
    return df.to_pandas()




## === cell 12
t0 = time.time()
train_clean = train.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text"),
    pl.col("paragraph")
    .map_elements(lambda lst: [dataPreprocessing(s) for s in (lst or [])])
    .alias("clean_paragraph"),
)
test_clean = test.with_columns(
    pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text"),
    pl.col("paragraph")
    .map_elements(lambda lst: [dataPreprocessing(s) for s in (lst or [])])
    .alias("clean_paragraph"),
)
print(f"Cleaned text built in {time.time()-t0:.1f}s")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_54/3259412389.py in <cell line: 0>()
      1 # Runtime fix: build cleaned full_text once (used by both vectorizers and engineered features).
      2 t0 = time.time()
----> 3 train_clean = train.with_columns(
      4     pl.col("full_text").map_elements(dataPreprocessing).alias("clean_text"),
      5     pl.col("paragraph")

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

ComputeError: TypeError: the truth value of a Series is ambiguous

Here are some things you might want to try:
- instead of `if s`, use `if not s.is_empty()`
- instead of `s1 and s2`, use `s1 & s2`
- instead of `s1 or s2`, use `s1 | s2`
- instead of `s in [y, z]`, use `s.is_in([y, z])`


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
    sublinear_tf=True,
)

train_tfid = vectorizer.fit_transform(train_clean["clean_text"].to_list())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/961438949.py in <cell line: 0>()
     12 
     13 # Runtime fix: avoid Python list comprehension over polars Series; use to_list() directly.
---> 14 train_tfid = vectorizer.fit_transform(train_clean["clean_text"].to_list())
     15 

NameError: name 'train_clean' is not defined

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

train_cnt = vectorizer_cnt.fit_transform(train_clean["clean_text"].to_list())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2575722201.py in <cell line: 0>()
     10 )
     11 
---> 12 train_cnt = vectorizer_cnt.fit_transform(train_clean["clean_text"].to_list())
     13 

NameError: name 'train_clean' is not defined

## === cell 15
if CFG.LOAD_FEATURES_FROM is not None and os.path.exists(CFG.LOAD_FEATURES_FROM):
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)
else:
    print("Compute train features (no cached features found)")
    t0 = time.time()
    train_feats1 = Paragraph_aggregation(
        Paragraph_Features(
            train_clean.select(["essay_id", "clean_paragraph"]),
            paragraph_col="clean_paragraph",
        )
    )
    train_feats2 = Sentence_aggregation(
        Sentence_Features(
            train_clean.select(["essay_id", "clean_text"]), full_text_col="clean_text"
        )
    )
    train_feats3 = Word_aggregation(
        Word_Features(
            train_clean.select(["essay_id", "clean_text"]), full_text_col="clean_text"
        )
    )

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
    print(f"Train engineered features computed in {time.time() - t0:.1f}s")

print(train_feats.shape)
print(train_feats.head())



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2973397980.py in <cell line: 0>()
      9     train_feats1 = Paragraph_aggregation(
     10         Paragraph_Features(
---> 11             train_clean.select(["essay_id", "clean_paragraph"]),
     12             paragraph_col="clean_paragraph",
     13         )

NameError: name 'train_clean' is not defined

## === cell 16
print("LightGBM Version:", lgb.__version__)


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
    df = preds - labels
    dg = preds - a
    grad = (df / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.948
b = 1.092

categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3005013054.py in <cell line: 0>()
     25 b = 1.092
     26 
---> 27 categorical_columns = train_feats.select_dtypes(
     28     include=["object", "category"]
     29 ).columns.tolist()

NameError: name 'train_feats' is not defined

## === cell 17
t0 = time.time()
X_dense = train_feats[FEATURES].fillna(0).to_numpy()
X_dense = np.clip(X_dense, 0, 10000)
X_sparse = sparse.hstack(
    [sparse.csr_matrix(X_dense), train_tfid.tocsr(), train_cnt.tocsr()],
    format="csr",
)
y = (train_feats[TARGET].to_numpy() - a).astype(np.float64)
print(f"Built train matrices in {time.time()-t0:.1f}s; X_sparse shape={X_sparse.shape}")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3717134048.py in <cell line: 0>()
      2 # This avoids per-fold pandas fillna/clip copies and eliminates expensive dense vectorizer materialization.
      3 t0 = time.time()
----> 4 X_dense = train_feats[FEATURES].fillna(0).to_numpy()
      5 X_dense = np.clip(X_dense, 0, 10000)
      6 X_sparse = sparse.hstack(

NameError: name 'train_feats' is not defined

## === cell 18
def train_lightgbm_models():
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
            objective=qwk_obj,
            metrics="None",
            learning_rate=0.05,
            colsample_bytree=0.8,
            max_depth=5,
            num_leaves=10,
            reg_alpha=0.2,
            reg_lambda=0.8,
            n_estimators=1024,
            class_weight="balanced",
            random_state=CFG.SEED,
            verbosity=-1,
            n_jobs=-1,
        )

        train_x = X_sparse[train_index]
        train_y = y[train_index]

        valid_x = X_sparse[valid_index]
        valid_y = y[valid_index]

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
        all_true.append(valid_y + a)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for LightGBM =", cv)

    if DO_PLOTS:
        cm = confusion_matrix(
            all_true, np.clip(all_oof, 1, 6).round(), labels=[x for x in range(1, 7)]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
        )
        disp.plot()
        plt.show()




## === cell 19
have_pretrained = False
if CFG.LOAD_MODELS_FROM is not None and os.path.isdir(CFG.LOAD_MODELS_FROM):
    expected = os.path.join(CFG.LOAD_MODELS_FROM, f"LGB_v{CFG.VER}_f0.pkl")
    have_pretrained = os.path.exists(expected)

if not have_pretrained:
    print("Training LightGBM (no pretrained models found)")
    train_lightgbm_models()
else:
    print("Using pretrained models from:", CFG.LOAD_MODELS_FROM)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1353769843.py in <cell line: 0>()
      6 if not have_pretrained:
      7     print("Training LightGBM (no pretrained models found)")
----> 8     train_lightgbm_models()
      9 else:
     10     print("Using pretrained models from:", CFG.LOAD_MODELS_FROM)

/tmp/ipykernel_54/1472275516.py in train_lightgbm_models()
      5     skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
      6     for i, (train_index, valid_index) in enumerate(
----> 7         skf.split(train_feats, train_feats[TARGET])
      8     ):
      9         print("#" * 25)

NameError: name 'train_feats' is not defined

## === cell 20
test_tfid = vectorizer.transform(test_clean["clean_text"].to_list())
test_cnt = vectorizer_cnt.transform(test_clean["clean_text"].to_list())



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1359920132.py in <cell line: 0>()
      1 # Runtime fix: keep test vectorizer outputs sparse; reuse pre-cleaned text.
----> 2 test_tfid = vectorizer.transform(test_clean["clean_text"].to_list())
      3 test_cnt = vectorizer_cnt.transform(test_clean["clean_text"].to_list())
      4 

NameError: name 'test_clean' is not defined

## === cell 21
t0 = time.time()
test_feats1 = Paragraph_aggregation(
    Paragraph_Features(
        test_clean.select(["essay_id", "clean_paragraph"]),
        paragraph_col="clean_paragraph",
    )
)
test_feats2 = Sentence_aggregation(
    Sentence_Features(
        test_clean.select(["essay_id", "clean_text"]), full_text_col="clean_text"
    )
)
test_feats3 = Word_aggregation(
    Word_Features(
        test_clean.select(["essay_id", "clean_text"]), full_text_col="clean_text"
    )
)

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape, f"(built in {time.time()-t0:.1f}s)")
print(test_feats.head())



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2850809541.py in <cell line: 0>()
      2 test_feats1 = Paragraph_aggregation(
      3     Paragraph_Features(
----> 4         test_clean.select(["essay_id", "clean_paragraph"]),
      5         paragraph_col="clean_paragraph",
      6     )

NameError: name 'test_clean' is not defined

## === cell 22
X_test_dense = test_feats[FEATURES].fillna(0).to_numpy()
X_test_dense = np.clip(X_test_dense, 0, 10000)
X_test_sparse = sparse.hstack(
    [sparse.csr_matrix(X_test_dense), test_tfid.tocsr(), test_cnt.tocsr()],
    format="csr",
)

preds = []
for i in range(15):
    print(f"Fold {i+1}")
    if have_pretrained:
        model_path = os.path.join(CFG.LOAD_MODELS_FROM, f"LGB_v{CFG.VER}_f{i}.pkl")
    else:
        model_path = f"LGB_v{CFG.VER}_f{i}.pkl"

    model = pickle.load(open(model_path, "rb"))
    pred = model.predict(X_test_sparse) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1873961302.py in <cell line: 0>()
      1 # Runtime fix: build test matrix once (dense engineered + sparse tfidf + sparse counts) to speed prediction loop.
----> 2 X_test_dense = test_feats[FEATURES].fillna(0).to_numpy()
      3 X_test_dense = np.clip(X_test_dense, 0, 10000)
      4 X_test_sparse = sparse.hstack(
      5     [sparse.csr_matrix(X_test_dense), test_tfid.tocsr(), test_cnt.tocsr()],

NameError: name 'test_feats' is not defined

## === cell 23
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = np.clip(pred1, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print(sub.head())
print("Saved to submission.csv")

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1096984345.py in <cell line: 0>()
      1 sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
----> 2 sub["score"] = np.clip(pred1, 1, 6).round().astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 
      5 print("Submission shape", sub.shape)

NameError: name 'pred1' is not defined

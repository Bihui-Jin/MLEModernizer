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
import re
from pathlib import Path
import pickle

import pandas as pd, numpy as np
import polars as pl  # For Feature Engineering

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from scipy import sparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For GPU T4x2

import warnings

warnings.filterwarnings("ignore")


def display(x):
    try:
        from IPython.display import display as ipy_display  # type: ignore

        return ipy_display(x)
    except Exception:
        print(x)
        return None


def resolve_base_path(preferred: str) -> str:
    """
    Fixes FileNotFoundError due to different Kaggle mount layouts by selecting
    the first existing dataset directory.
    """
    candidates = [
        preferred,
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/",
        "/kaggle/data/",
    ]
    for c in candidates:
        if (Path(c) / "train.csv").exists() and (Path(c) / "test.csv").exists():
            return c if c.endswith("/") else c + "/"
    return preferred if preferred.endswith("/") else preferred + "/"




## === cell 1
nltk.data.path = ["/kaggle/input"] + nltk.data.path




## === cell 2
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


CFG.BASE_PATH = resolve_base_path(CFG.BASE_PATH)
print("Resolved BASE_PATH:", CFG.BASE_PATH)



## === cell 3
for fn in ["train.csv", "test.csv", "sample_submission.csv"]:
    p = Path(CFG.BASE_PATH) / fn
    print(fn, "exists:", p.exists())



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
pl.Config.set_tbl_rows(10)




## === cell 6
def seed_everything():  # To proudce simliar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 7
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")



## === cell 8
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 9
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 10
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData



## === cell 11
assert "essay_id" in df_train.columns and "score" in df_train.columns
assert "essay_id" in df_test.columns and "full_text" in df_test.columns



## === cell 12
_HTML_RE = re.compile(r"<.*?>")
_AT_RE = re.compile(r"@\w+")
_APOS_DIG_RE = re.compile(r"'\d+")
_DIG_RE = re.compile(r"\d+")
_HTTP_RE = re.compile(r"http\w+")
_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r'[^\w\s.,;:"' "?!]")
_DOTPLUS_RE = re.compile(r"\.+")
_COMMAPLUS_RE = re.compile(r"\,+")


def removeHTML(x: str) -> str:
    return _HTML_RE.sub(r"", x)


def dataPreprocessing(x: str) -> str:
    x = x.lower()
    x = removeHTML(x)
    x = _AT_RE.sub("", x)
    x = _APOS_DIG_RE.sub("", x)
    x = _DIG_RE.sub("", x)
    x = _HTTP_RE.sub("", x)
    x = _WS_RE.sub(" ", x)
    x = _NONWORD_RE.sub("", x)
    x = x.replace("paragraph", "")
    x = _DOTPLUS_RE.sub(".", x)
    x = _COMMAPLUS_RE.sub(",", x)
    x = x.strip()
    return x


def dataPreprocessing_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r'[^\w\s.,;:"' "?!]", "")
        .str.replace_all(r"paragraph", "")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
    )




## === cell 13
pass



## === cell 14
COMMON_WORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "it",
    "that",
    "for",
    "on",
    "with",
    "as",
    "was",
    "are",
    "be",
    "this",
    "have",
    "or",
    "at",
    "from",
    "by",
    "an",
    "but",
    "not",
    "they",
    "we",
    "you",
    "i",
    "he",
    "she",
    "them",
    "his",
    "her",
    "their",
    "my",
    "our",
    "your",
    "so",
    "if",
    "then",
    "than",
    "when",
    "what",
    "which",
    "who",
    "whom",
    "because",
    "can",
    "could",
    "should",
    "would",
    "will",
    "just",
    "also",
    "about",
    "into",
    "more",
    "most",
    "some",
    "no",
    "yes",
}

_word_re = re.compile(r"[a-zA-Z']+")


def count_misspelled_words(text: str) -> int:
    if text is None:
        return 0
    tokens = _word_re.findall(text.lower())
    miss = 0
    for w in tokens:
        if len(w) < 5:
            continue
        if w in COMMON_WORDS:
            continue
        if w.count("'") >= 2:
            miss += 1
            continue
        if re.search(r"(.)\1\1", w):
            miss += 1
            continue
        vowels = sum(ch in "aeiou" for ch in w)
        if vowels <= 1:
            miss += 1
            continue
    return miss


_COMMON_WORDS_RE = r"^(?:" + "|".join(sorted(map(re.escape, COMMON_WORDS))) + r")$"


def misspelled_count_expr(col: str) -> pl.Expr:
    triple_repeat = (
        pl.element().str.to_lowercase().str.extract_all(r"([a-z])\1{2,}").list.len() > 0
    )

    w = pl.col(col).str.extract_all(r"[A-Za-z']+").alias("_wlist")
    return w.list.eval(
        pl.when(pl.element().str.len_chars() < 5)
        .then(0)
        .when(pl.element().str.to_lowercase().str.contains(_COMMON_WORDS_RE))
        .then(0)
        .when(pl.element().str.count_matches("'") >= 2)
        .then(1)
        .when(triple_repeat)
        .then(1)
        .when(pl.element().str.to_lowercase().str.count_matches(r"[aeiou]") <= 1)
        .then(1)
        .otherwise(0)
    ).list.sum()




## === cell 15
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
    x = x.with_columns(dataPreprocessing_expr("paragraph").alias("paragraph"))
    print("Caculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        misspelled_count_expr("paragraph").alias("paragraph_misspelled_cnt"),
        pl.col("paragraph").str.count_matches(",").alias("paragraph_comma_cnt"),
    )
    print("Caculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .str.count_matches(r"\.")
        .add(1)
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph").str.count_matches(r" ").add(1).alias("paragraph_word_cnt"),
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




## === cell 16
sentence_features = ["sentence_len", "sentence_word_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(
        dataPreprocessing_expr("full_text").str.split(".").alias("sentence")
    )
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .alias("only_sentence_len")
    )
    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence").str.count_matches(r" ").add(1).alias("sentence_word_cnt")
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




## === cell 17
word_features = [
    "word_len",
]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    print("Preprocess full_text and use spaces to seperate words fro the text")
    x = x.with_columns(dataPreprocessing_expr("full_text").str.split(" ").alias("word"))
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(pl.col("word").str.len_chars().alias("word_len"))
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




## === cell 18
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

train_a = train.with_columns(dataPreprocessing_expr("full_text").alias("full_text"))
train_tfid = vectorizer.fit_transform(train_a["full_text"].to_list())



## === cell 19
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

train_b = train.with_columns(dataPreprocessing_expr("full_text").alias("full_text"))
train_cnt = vectorizer_cnt.fit_transform(train_b["full_text"].to_list())



## === cell 20
t0 = time.time()
train_feats1 = Paragraph_Features(train)
train_feats1 = Paragraph_aggregation(train_feats1)
train_feats2 = Sentence_Features(train)
train_feats2 = Sentence_aggregation(train_feats2)
train_feats3 = Word_Features(train)
train_feats3 = Word_aggregation(train_feats3)

train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
train_feats["score"] = df_train["score"].values
print(f"Train handcrafted feature extraction time: {time.time()-t0:.1f}s")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3864582593.py in <cell line: 0>()
      1 t0 = time.time()
----> 2 train_feats1 = Paragraph_Features(train)
      3 train_feats1 = Paragraph_aggregation(train_feats1)
      4 train_feats2 = Sentence_Features(train)
      5 train_feats2 = Sentence_aggregation(train_feats2)

/tmp/ipykernel_55/2781919772.py in Paragraph_Features(x)
     13     x = x.with_columns(dataPreprocessing_expr("paragraph").alias("paragraph"))
     14     print("Caculate the length of each paragraph")
---> 15     x = x.with_columns(
     16         pl.col("paragraph").str.len_chars().alias("paragraph_len"),
     17         misspelled_count_expr("paragraph").alias("paragraph_misspelled_cnt"),

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

ComputeError: regex error: regex parse error:
    ([a-z])\1{2,}
           ^^
error: backreferences are not supported

## === cell 21
print("Save train_feats.csv")
train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
print("train_feats saved:", Path(f"train_feats_{CFG.VER}.csv").exists())



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2691869107.py in <cell line: 0>()
      1 print("Save train_feats.csv")
----> 2 train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      3 print("train_feats saved:", Path(f"train_feats_{CFG.VER}.csv").exists())
      4 

NameError: name 'train_feats' is not defined

## === cell 22
display(train_feats.head())
print("train_feats shape:", train_feats.shape)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3096950040.py in <cell line: 0>()
----> 1 display(train_feats.head())
      2 print("train_feats shape:", train_feats.shape)
      3 

NameError: name 'train_feats' is not defined

## === cell 23
train_feats = train_feats.replace([np.inf, -np.inf], np.nan)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2187932451.py in <cell line: 0>()
----> 1 train_feats = train_feats.replace([np.inf, -np.inf], np.nan)
      2 

NameError: name 'train_feats' is not defined

## === cell 24
import xgboost as xgb

print("XGBoost Version: ", xgb.__version__)



## === cell 25
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()

HAND_FEATURES = [
    c
    for c in train_feats.columns
    if c not in categorical_columns + ["score", "essay_id"]
]
TARGET = "score"

TFIDF_FEATURES = [f"tfidf_{i}" for i in range(train_tfid.shape[1])]
CNT_FEATURES = [f"cnt_{i}" for i in range(train_cnt.shape[1])]
FEATURES = HAND_FEATURES + TFIDF_FEATURES + CNT_FEATURES
print("Num HAND_FEATURES:", len(HAND_FEATURES))
print("Num TFIDF_FEATURES:", len(TFIDF_FEATURES))
print("Num CNT_FEATURES:", len(CNT_FEATURES))
print("Num FEATURES total:", len(FEATURES))




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2066173697.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 
      5 HAND_FEATURES = [

NameError: name 'train_feats' is not defined

## === cell 26
def quadratic_weighted_kappa(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk




## === cell 27
def xgboost_train():
    all_oof = []
    all_true = []

    X_hand = np.clip(train_feats[HAND_FEATURES].fillna(0).to_numpy(), 0, 10000)
    X_hand = sparse.csr_matrix(X_hand)
    X_all = sparse.hstack([X_hand, train_tfid, train_cnt], format="csr")

    y_all = train_feats[TARGET].to_numpy()

    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)
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
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            n_estimators=1000,
            random_state=CFG.SEED,
            verbosity=0,
        )

        X_tr = X_all[train_index]
        y_tr = y_all[train_index]
        X_va = X_all[valid_index]
        y_va = y_all[valid_index]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=50,
        )

        pickle.dump(model, open(f"XGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(X_va)
        all_oof.append(oof)
        all_true.append(y_va)

        del X_tr, y_tr, X_va, y_va, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = cohen_kappa_score(true[0], oof[0].clip(1, 6).round(), weights="quadratic")
    print("CV Score for XGBoost = ", cv)
    cm = confusion_matrix(
        true[0], oof[0].clip(1, 6).round(), labels=[x for x in range(1, 7)]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    disp.plot()
    plt.show()




## === cell 28
print("Training XGBoost")
xgboost_train()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2000765327.py in <cell line: 0>()
      1 print("Training XGBoost")
----> 2 xgboost_train()
      3 

/tmp/ipykernel_55/1218878240.py in xgboost_train()
      3     all_true = []
      4 
----> 5     X_hand = np.clip(train_feats[HAND_FEATURES].fillna(0).to_numpy(), 0, 10000)
      6     X_hand = sparse.csr_matrix(X_hand)
      7     X_all = sparse.hstack([X_hand, train_tfid, train_cnt], format="csr")

NameError: name 'train_feats' is not defined

## === cell 29
model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))

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
plt.show()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1704583636.py in <cell line: 0>()
----> 1 model = pickle.load(open(f"XGB_v{CFG.VER}_f0.pkl", "rb"))
      2 
      3 df_importance = pd.DataFrame(
      4     {
      5         "features_name": FEATURES,

FileNotFoundError: [Errno 2] No such file or directory: 'XGB_v1_f0.pkl'

## === cell 30
test_a = test.with_columns(dataPreprocessing_expr("full_text").alias("full_text"))
test_tfid = vectorizer.transform(test_a["full_text"].to_list())



## === cell 31
test_b = test.with_columns(dataPreprocessing_expr("full_text").alias("full_text"))
test_cnt = vectorizer_cnt.transform(test_b["full_text"].to_list())



## === cell 32
t0 = time.time()
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)
print(f"Test handcrafted feature extraction time: {time.time()-t0:.1f}s")



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
ComputeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2779024510.py in <cell line: 0>()
      1 t0 = time.time()
----> 2 test_feats1 = Paragraph_Features(test)
      3 test_feats1 = Paragraph_aggregation(test_feats1)
      4 test_feats2 = Sentence_Features(test)
      5 test_feats2 = Sentence_aggregation(test_feats2)

/tmp/ipykernel_55/2781919772.py in Paragraph_Features(x)
     13     x = x.with_columns(dataPreprocessing_expr("paragraph").alias("paragraph"))
     14     print("Caculate the length of each paragraph")
---> 15     x = x.with_columns(
     16         pl.col("paragraph").str.len_chars().alias("paragraph_len"),
     17         misspelled_count_expr("paragraph").alias("paragraph_misspelled_cnt"),

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

ComputeError: regex error: regex parse error:
    ([a-z])\1{2,}
           ^^
error: backreferences are not supported

## === cell 33
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.replace([np.inf, -np.inf], np.nan)
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1528796659.py in <cell line: 0>()
----> 1 test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
      2 test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
      3 test_feats = test_feats.replace([np.inf, -np.inf], np.nan)
      4 print("Shape of test_feats:", test_feats.shape)
      5 display(test_feats.head())

NameError: name 'test_feats1' is not defined

## === cell 34
preds = []

missing_hand = [c for c in HAND_FEATURES if c not in test_feats.columns]
if missing_hand:
    for c in missing_hand:
        test_feats[c] = 0

X_test_hand = np.clip(test_feats[HAND_FEATURES].fillna(0).to_numpy(), 0, 10000)
X_test_hand = sparse.csr_matrix(X_test_hand)
test_matrix = sparse.hstack([X_test_hand, test_tfid, test_cnt], format="csr")

for i in range(10):
    print(f"Fold {i+1}")
    model = pickle.load(open(f"XGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred_fold = model.predict(test_matrix)
    preds.append(pred_fold)

pred = np.mean(preds, axis=0)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3133803404.py in <cell line: 0>()
      1 preds = []
      2 
----> 3 missing_hand = [c for c in HAND_FEATURES if c not in test_feats.columns]
      4 if missing_hand:
      5     for c in missing_hand:

NameError: name 'HAND_FEATURES' is not defined

## === cell 35
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pd.Series(pred).clip(1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print(
    "Saved submission:",
    Path("submission.csv").exists(),
    "->",
    Path("submission.csv").resolve(),
)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/452017333.py in <cell line: 0>()
      1 sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
----> 2 sub["score"] = pd.Series(pred).clip(1, 6).round().astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 display(sub.head())

NameError: name 'pred' is not defined

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
df_train = df_train.sort_values(by="essay_id").reset_index(drop=True)
print("Shape of Train: ", df_train.shape)
display(df_train.head())



## === cell 9
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id").reset_index(drop=True)

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
assert df_train["essay_id"].is_monotonic_increasing
assert df_test["essay_id"].is_monotonic_increasing



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
        pl.element()
        .str.to_lowercase()
        .str.contains(
            r"(?:a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{3,}|p{3,}|q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,})"
        )
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

    eps = 1e-6
    cast_cols = ["word_2_cnt_v2", "word_3_cnt_v2"]
    for c in cast_cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    df = df.with_columns(
        *[pl.col(c).cast(pl.Float64) for c in df.columns if c.endswith("_cnt")],
        *[pl.col(c).cast(pl.Float64) for c in df.columns if c.endswith("_cnt_v2")],
    )

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




## === cell 18
t0 = time.time()
train_text_pp = train.select(
    pl.col("essay_id"),
    dataPreprocessing_expr("full_text").alias("full_text_pp"),
)
test_text_pp = test.select(
    pl.col("essay_id"),
    dataPreprocessing_expr("full_text").alias("full_text_pp"),
)
print(f"Preprocessed full_text (train+test) time: {time.time()-t0:.1f}s")



## === cell 19
TFIDF_MIN_DF = (
    3  # was 0.05 (very aggressive); integer keeps semantics "seen in >=3 docs"
)
TFIDF_MAX_DF = 0.95

vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    sublinear_tf=True,  # Term Frequency Log Scaling
)

train_text_list = train_text_pp["full_text_pp"].to_numpy().tolist()
train_tfid = vectorizer.fit_transform(train_text_list)
print("train_tfid shape:", train_tfid.shape)



## === cell 20
CNT_MIN_DF = 3  # was 0.10
CNT_MAX_DF = 0.85

vectorizer_cnt = CountVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 4),
    min_df=CNT_MIN_DF,
    max_df=CNT_MAX_DF,
)

train_cnt = vectorizer_cnt.fit_transform(train_text_list)
print("train_cnt shape:", train_cnt.shape)



## === cell 21
t0 = time.time()

cache_key = f"v{CFG.VER}_tfmin{TFIDF_MIN_DF}_cntmin{CNT_MIN_DF}"
train_feats_path = Path(f"train_feats_{cache_key}.csv")

if train_feats_path.exists():
    print("Loading cached train feats from:", train_feats_path)
    train_feats = pd.read_csv(train_feats_path)
else:
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(
        train.with_columns(train_text_pp["full_text_pp"].alias("full_text"))
    )
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(
        train.with_columns(train_text_pp["full_text_pp"].alias("full_text"))
    )
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
    print(f"Train handcrafted feature extraction time: {time.time()-t0:.1f}s")



## === cell 22
print("Save train_feats.csv")
train_feats.to_csv(train_feats_path, index=False)
print("train_feats saved:", train_feats_path.exists())



## === cell 23
display(train_feats.head())
print("train_feats shape:", train_feats.shape)



## === cell 24
train_feats = train_feats.replace([np.inf, -np.inf], np.nan)



## === cell 25
import xgboost as xgb

print("XGBoost Version: ", xgb.__version__)



## === cell 26
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




## === cell 27
def quadratic_weighted_kappa(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk




## === cell 28
def _detect_xgb_tree_method() -> str:
    try:
        import subprocess, shlex

        cmd = "nvidia-smi -L"
        out = subprocess.check_output(
            shlex.split(cmd), stderr=subprocess.STDOUT, timeout=2
        ).decode("utf-8", "ignore")
        if "GPU" in out:
            return "gpu_hist"
    except Exception:
        pass
    return "hist"


def fit_ordinal_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Greedy coordinate-descent on 5 thresholds that map continuous preds -> {1..6}.
    Deterministic and fast; uses only OOF predictions (no leakage).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    thr = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)

    def apply_thr(pred, t):
        return np.digitize(pred, t) + 1  # bins -> 1..6

    best = quadratic_weighted_kappa(y_true, apply_thr(y_pred, thr))

    for _ in range(6):
        improved = False
        for k in range(5):
            base = thr[k]
            grid = np.linspace(base - 0.6, base + 0.6, 31)
            lo = 1.0 if k == 0 else thr[k - 1] + 1e-3
            hi = 6.0 if k == 4 else thr[k + 1] - 1e-3
            grid = grid[(grid > lo) & (grid < hi)]
            if grid.size == 0:
                continue

            local_best = best
            local_thr = base
            for g in grid:
                t2 = thr.copy()
                t2[k] = float(g)
                s = quadratic_weighted_kappa(y_true, apply_thr(y_pred, t2))
                if s > local_best:
                    local_best = s
                    local_thr = float(g)
            if local_best > best:
                thr[k] = local_thr
                best = local_best
                improved = True
        if not improved:
            break

    return thr


def apply_ordinal_thresholds(y_pred: np.ndarray, thr: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred).astype(float)
    thr = np.asarray(thr).astype(float)
    return (np.digitize(y_pred, thr) + 1).astype(int)


def xgboost_train():
    X_hand = np.clip(train_feats[HAND_FEATURES].fillna(0).to_numpy(), 0, 10000)
    X_hand = sparse.csr_matrix(X_hand)
    X_all = sparse.hstack([X_hand, train_tfid, train_cnt], format="csr")

    y_all = train_feats[TARGET].to_numpy()

    tree_method = _detect_xgb_tree_method()
    print("Using XGBoost tree_method:", tree_method)

    n_jobs = int(os.environ.get("OMP_NUM_THREADS", "4"))

    skf = StratifiedKFold(n_splits=10, random_state=CFG.SEED, shuffle=True)

    oof_pred = np.zeros(len(train_feats), dtype=float)

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
            tree_method=tree_method,
            n_jobs=n_jobs,
        )

        X_tr = X_all[train_index]
        y_tr = y_all[train_index]
        X_va = X_all[valid_index]
        y_va = y_all[valid_index]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=0,  # Speed: avoid log I/O; does not affect training/results.
        )

        pickle.dump(model, open(f"XGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof_pred[valid_index] = model.predict(X_va)

        del X_tr, y_tr, X_va, y_va, model
        clean_memory()

    cv_round = cohen_kappa_score(
        y_all, np.clip(oof_pred, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for XGBoost (naive round) = ", cv_round)

    thr = fit_ordinal_thresholds(y_all.astype(int), oof_pred)
    cv_thr = cohen_kappa_score(
        y_all, apply_ordinal_thresholds(oof_pred, thr), weights="quadratic"
    )
    print("CV Score for XGBoost (thresholds) = ", cv_thr)
    print("Learned thresholds:", thr)

    with open(f"thresholds_v{CFG.VER}.pkl", "wb") as f:
        pickle.dump(thr, f)

    return cv_thr




## === cell 29
print("Training XGBoost")
_ = xgboost_train()



## === cell 30
model_path = Path(f"XGB_v{CFG.VER}_f0.pkl")
if model_path.exists():
    model = pickle.load(open(model_path, "rb"))

    df_importance = pd.DataFrame(
        {
            "features_name": FEATURES,
            "importance": model.feature_importances_,
        }
    )
    df_importance = df_importance.sort_values(by="importance", ascending=False)
    print("Top-10 feature importances:")
    print(df_importance.head(10).to_string(index=False))
else:
    print("Model file not found for importance:", model_path)



## === cell 31
test_text_list = test_text_pp["full_text_pp"].to_numpy().tolist()
test_tfid = vectorizer.transform(test_text_list)
print("test_tfid shape:", test_tfid.shape)



## === cell 32
test_cnt = vectorizer_cnt.transform(test_text_list)
print("test_cnt shape:", test_cnt.shape)



## === cell 33
t0 = time.time()
test_feats_path = Path(f"test_feats_{cache_key}.csv")

if test_feats_path.exists():
    print("Loading cached test feats from:", test_feats_path)
    test_feats = pd.read_csv(test_feats_path)
else:
    test_feats1 = Paragraph_Features(test)
    test_feats1 = Paragraph_aggregation(test_feats1)
    test_feats2 = Sentence_Features(
        test.with_columns(test_text_pp["full_text_pp"].alias("full_text"))
    )
    test_feats2 = Sentence_aggregation(test_feats2)
    test_feats3 = Word_Features(
        test.with_columns(test_text_pp["full_text_pp"].alias("full_text"))
    )
    test_feats3 = Word_aggregation(test_feats3)
    print(f"Test handcrafted feature extraction time: {time.time()-t0:.1f}s")

    test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
    test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
    test_feats = test_feats.replace([np.inf, -np.inf], np.nan)
    test_feats.to_csv(test_feats_path, index=False)

print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())



## === cell 34
preds = []

missing_hand = [c for c in HAND_FEATURES if c not in test_feats.columns]
if missing_hand:
    for c in missing_hand:
        test_feats[c] = 0

assert all(c in test_feats.columns for c in HAND_FEATURES)

X_test_hand = np.clip(test_feats[HAND_FEATURES].fillna(0).to_numpy(), 0, 10000)
X_test_hand = sparse.csr_matrix(X_test_hand)
test_matrix = sparse.hstack([X_test_hand, test_tfid, test_cnt], format="csr")

for i in range(10):
    print(f"Fold {i+1}")
    model = pickle.load(open(f"XGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred_fold = model.predict(test_matrix)
    preds.append(pred_fold)

pred = np.mean(preds, axis=0)



## === cell 35
thr_path = Path(f"thresholds_v{CFG.VER}.pkl")
if thr_path.exists():
    thr = pickle.load(open(thr_path, "rb"))
    pred_score = apply_ordinal_thresholds(np.clip(pred, 1, 6), thr)
else:
    pred_score = pd.Series(pred).clip(1, 6).round().astype(int).to_numpy()

sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pred_score.astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print(
    "Saved submission:",
    Path("submission.csv").exists(),
    "->",
    Path("submission.csv").resolve(),
)

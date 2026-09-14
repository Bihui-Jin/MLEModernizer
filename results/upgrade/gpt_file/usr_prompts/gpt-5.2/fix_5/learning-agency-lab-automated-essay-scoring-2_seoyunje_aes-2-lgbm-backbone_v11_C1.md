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
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping

import warnings

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print("LightGBM Version:", lgb.__version__)




## === cell 1
class CFG:
    SEED = 2024
    VER = 1

    LOAD_MODELS_FROM = "/kaggle/input/aes2-lightgbm/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-lightgbm/train_feats_1.csv"

    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

    N_SPLITS = 15


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



## === cell 2
train_path = os.path.join(CFG.BASE_PATH, "train.csv")
test_path = os.path.join(CFG.BASE_PATH, "test.csv")
sample_path = os.path.join(CFG.BASE_PATH, "sample_submission.csv")

df_train = (
    pl.read_csv(train_path, columns=["essay_id", "full_text", "score"])
    .sort("essay_id")
    .to_pandas()
)
df_test = (
    pl.read_csv(test_path, columns=["essay_id", "full_text"])
    .sort("essay_id")
    .to_pandas()
)

print("Shape of Train:", df_train.shape)
print(df_train.head())

print("Shape of Test:", df_test.shape)
print(df_test.head())



## === cell 3
if False:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df_train["score"])
    plt.title("Distribution of Score")
    plt.xlabel("Score of Essay")
    plt.ylabel("Frequency")
    plt.show()



## === cell 4
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema
schema_test = test.schema



## === cell 5
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not have",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(map(re.escape, cList.keys())))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


_html_re = re.compile(r"<.*?>")
_mention_re = re.compile(r"@\w+")
_apost_num_re = re.compile(r"'\d+")
_num_re = re.compile(r"\d+")
_http_re = re.compile(r"http\w+")
_space_re = re.compile(r"\s+")
_dot_re = re.compile(r"\.+")
_comma_re = re.compile(r"\,+")
_keep_re = re.compile(r"[^\w\s.,;:\"\"''?!]")


def removeHTML(x):
    return _html_re.sub(r"", x)


def dataPreprocessing(x):
    x = str(x).lower()
    x = removeHTML(x)
    x = _mention_re.sub("", x)
    x = _apost_num_re.sub("", x)
    x = _num_re.sub("", x)
    x = _http_re.sub("", x)
    x = _space_re.sub(" ", x)
    x = expandContractions(x)
    x = _dot_re.sub(".", x)
    x = _comma_re.sub(",", x)
    x = _keep_re.sub("", x)
    x = x.strip()
    return x




## === cell 6
_word_re = re.compile(r"[a-zA-Z']+")
_triple_re = re.compile(r"(.)\1\1")


def count_misspelled_words(text: str) -> int:
    text = str(text).lower()
    miss = 0
    for m in _word_re.finditer(text):
        w = m.group(0)
        lw = len(w)
        if lw <= 1 or lw >= 20:
            miss += 1
        elif _triple_re.search(w) is not None:
            miss += 1
    return miss




## === cell 7
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def ensure_preprocessed_full_text(x: pl.DataFrame) -> pl.DataFrame:
    if "full_text_pre" in x.columns:
        return x
    return x.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
        .alias("full_text_pre")
    )


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = ensure_preprocessed_full_text(x)
    x = x.explode("paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(dataPreprocessing, return_dtype=pl.Utf8)
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(s), return_dtype=pl.Int64)
        .alias("paragraph_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: s.count(","), return_dtype=pl.Int64)
        .alias("paragraph_comma_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(".")), return_dtype=pl.Int64)
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(" ")), return_dtype=pl.Int64)
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
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = ensure_preprocessed_full_text(x)
    x = x.with_columns(pl.col("full_text_pre").str.split(".").alias("sentence"))
    x = x.explode("sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s), return_dtype=pl.Int64)
        .alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("sentence_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.replace(" ", "")), return_dtype=pl.Int64)
        .alias("only_sentence_len")
    )
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")), return_dtype=pl.Int64)
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




## === cell 9
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = ensure_preprocessed_full_text(x)
    x = x.with_columns(pl.col("full_text_pre").str.split(" ").alias("word"))
    x = x.explode("word")
    x = x.with_columns(
        pl.col("word")
        .map_elements(lambda s: len(s), return_dtype=pl.Int64)
        .alias("word_len")
    )
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

df_train["full_text_pre"] = df_train["full_text"].map(dataPreprocessing)
df_test["full_text_pre"] = df_test["full_text"].map(dataPreprocessing)

train = train.with_columns(
    pl.Series("full_text_pre", df_train["full_text_pre"].to_list())
)
test = test.with_columns(pl.Series("full_text_pre", df_test["full_text_pre"].to_list()))


def fit_vectorizers_and_transform_train(train_pl: pl.DataFrame):
    train_pl = ensure_preprocessed_full_text(train_pl)
    texts = train_pl["full_text_pre"].to_list()
    train_tfid = vectorizer.fit_transform(texts)
    train_cnt = vectorizer_cnt.fit_transform(texts)
    return train_pl, train_tfid, train_cnt


train, train_tfid, train_cnt = fit_vectorizers_and_transform_train(train)



## === cell 11
load_feats_path = CFG.LOAD_FEATURES_FROM
if (load_feats_path is not None) and os.path.exists(load_feats_path):
    print("Load train_feats.csv from:", load_feats_path)
    train_feats = pd.read_csv(load_feats_path)
else:
    print("Precomputed train_feats not found; computing train features.")
    train_feats1 = Paragraph_aggregation(Paragraph_Features(train))
    train_feats2 = Sentence_aggregation(Sentence_Features(train))
    train_feats3 = Word_aggregation(Word_Features(train))

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values

    out_path = f"train_feats_{CFG.VER}.csv"
    train_feats.to_csv(out_path, index=False)
    print("Saved:", out_path)

print("train_feats shape:", train_feats.shape)
print(train_feats.head())




## === cell 12
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
    df_ = preds - labels
    dg = preds - a
    grad = (df_ / g - f * dg / g**2) * len(labels)
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



## === cell 13
from scipy import sparse


def build_sparse_design_matrix(
    feats_df: pd.DataFrame, tfid_mat, cnt_mat, clip_lo=0.0, clip_hi=10000.0
):
    X_dense = feats_df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    if np.isnan(X_dense).any():
        X_dense = np.nan_to_num(X_dense, nan=0.0, copy=False)
    np.clip(X_dense, clip_lo, clip_hi, out=X_dense)

    X = sparse.hstack(
        [sparse.csr_matrix(X_dense), tfid_mat.tocsr(), cnt_mat.tocsr()],
        format="csr",
    )
    return X


X_train_all = build_sparse_design_matrix(train_feats, train_tfid, train_cnt)
y_train_all = (train_feats[TARGET].to_numpy() - a).astype(np.float32, copy=False)




## === cell 14
def train_lightgbm_and_save(train_feats: pd.DataFrame):
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, random_state=CFG.SEED, shuffle=True)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 4)))

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
            n_jobs=n_threads,  # speed: parallelism, same algorithm
        )

        train_x = X_train_all[train_index]
        train_y = y_train_all[train_index]
        valid_x = X_train_all[valid_index]
        valid_y = y_train_all[valid_index]

        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
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

    if False:
        cm = confusion_matrix(
            all_true, np.clip(all_oof, 1, 6).round(), labels=[x for x in range(1, 7)]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
        )
        disp.plot()
        plt.show()




## === cell 15
models_dir = CFG.LOAD_MODELS_FROM
use_external_models = (
    bool(models_dir)
    and os.path.isdir(models_dir)
    and any(Path(models_dir).glob(f"LGB_v{CFG.VER}_f*.pkl"))
)

if use_external_models:
    print("Using external models from:", models_dir)
else:
    print("External models not found; training models locally.")
    train_lightgbm_and_save(train_feats)



## === cell 16
if False:
    if use_external_models:
        model0_path = os.path.join(models_dir, f"LGB_v{CFG.VER}_f0.pkl")
    else:
        model0_path = f"LGB_v{CFG.VER}_f0.pkl"

    model = pickle.load(open(model0_path, "rb"))

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
    plt.title("Distribution of Feature Importance of LightGBM")
    plt.xticks(rotation=90)
    plt.show()



## === cell 17
test = ensure_preprocessed_full_text(test)
test_texts = test["full_text_pre"].to_list()
test_tfid = vectorizer.transform(test_texts)
test_cnt = vectorizer_cnt.transform(test_texts)



## === cell 18
test_feats_cache = f"test_feats_{CFG.VER}.csv"
if os.path.exists(test_feats_cache):
    print("Loading cached test_feats from:", test_feats_cache)
    test_feats = pd.read_csv(test_feats_cache)
else:
    test_feats1 = Paragraph_aggregation(Paragraph_Features(test))
    test_feats2 = Sentence_aggregation(Sentence_Features(test))
    test_feats3 = Word_aggregation(Word_Features(test))

    test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
    test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
    test_feats.to_csv(test_feats_cache, index=False)
    print("Saved:", test_feats_cache)

print("Shape of test_feats:", test_feats.shape)
print(test_feats.head())



## === cell 19
X_test_all = build_sparse_design_matrix(test_feats, test_tfid, test_cnt)

preds = []

if use_external_models:
    fold_paths = sorted(
        Path(models_dir).glob(f"LGB_v{CFG.VER}_f*.pkl"),
        key=lambda p: int(p.stem.split("_f")[-1]),
    )
else:
    fold_paths = sorted(
        Path(".").glob(f"LGB_v{CFG.VER}_f*.pkl"),
        key=lambda p: int(p.stem.split("_f")[-1]),
    )

if len(fold_paths) == 0:
    raise FileNotFoundError("No fold model files found to run inference.")

for p in fold_paths:
    with open(p, "rb") as f:
        model = pickle.load(f)
    pred = model.predict(X_test_all) + a
    preds.append(pred)
    del model
    clean_memory()

pred1 = np.mean(preds, axis=0)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = np.clip(pred1, 1, 6).round().astype(int)

sub = sub.sort_values("essay_id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False)

print("Submission shape:", sub.shape)
print(sub.head())
print("Wrote submission.csv")

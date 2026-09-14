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

DO_PLOTS = False

N_THREADS = int(os.environ.get("OMP_NUM_THREADS", max(1, os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_THREADS))




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-lightgbm/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-lightgbm/train_feats_1.csv"
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
def seed_everything():  # To produce similar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 4
try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        return x


df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id").reset_index(drop=True)

print("Shape of Train: ", df_train.shape)
display(df_train.head())




## === cell 5
if DO_PLOTS:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df_train["score"])
    plt.title("Distribution of Score")
    plt.xlabel("Score of Essay")
    plt.ylabel("Frequency")
    plt.show()




## === cell 6
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id").reset_index(drop=True)

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
    "doesn't": "does not",
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




## === cell 9
c_re = re.compile("(%s)" % "|".join(map(re.escape, cList.keys())))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)  # html -> ''


_re_at = re.compile(r"@\w+")
_re_apnum = re.compile(r"'\d+")
_re_num = re.compile(r"\d+")
_re_http = re.compile(r"http\w+")
_re_space = re.compile(r"\s+")
_re_dots = re.compile(r"\.+")
_re_commas = re.compile(r"\,+")
_re_keep = re.compile(r'[^\w\s.,;:""\'\'?!]')


def dataPreprocessing(x):
    x = str(x)
    x = x.lower()
    x = removeHTML(x)
    x = _re_at.sub("", x)
    x = _re_apnum.sub("", x)
    x = _re_num.sub("", x)
    x = _re_http.sub("", x)
    x = _re_space.sub(" ", x)
    x = expandContractions(x)
    x = _re_dots.sub(".", x)
    x = _re_commas.sub(",", x)
    x = _re_keep.sub("", x)
    x = x.strip()
    return x




## === cell 10
from functools import lru_cache

try:
    from spellchecker import SpellChecker  # type: ignore

    _spell = SpellChecker()

    @lru_cache(maxsize=200_000)
    def count_misspelled_words(text: str) -> int:
        misspelled_words = _spell.unknown(str(text).split())
        return len(misspelled_words)

except Exception:
    _word_re = re.compile(r"[a-z]+")

    @lru_cache(maxsize=200_000)
    def count_misspelled_words(text: str) -> int:
        tokens = str(text).lower().split()
        cnt = 0
        for t in tokens:
            w = re.sub(r"[^a-z]", "", t)
            if w == "":
                continue
            if len(w) > 18:
                cnt += 1
            elif _word_re.fullmatch(w) is None:
                cnt += 1
        return cnt




## === cell 11
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def Paragraph_Features(x, paragraph_col: str = "paragraph"):
    x = x.explode(paragraph_col)

    print("Paragraph Preprocessing")
    if paragraph_col == "paragraph":
        x = x.with_columns(
            pl.col("paragraph").map_elements(dataPreprocessing, return_dtype=pl.String)
        )
    else:
        x = x.with_columns(pl.col(paragraph_col).alias("paragraph"))

    print("Caculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        pl.col("paragraph").str.count_matches(",").alias("paragraph_comma_cnt"),
    )

    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("paragraph_misspelled_cnt")
    )

    print("Caculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph").str.split(".").list.len().alias("paragraph_sentence_cnt"),
        pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
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
    df = df.to_pandas()
    return df




## === cell 12
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x, text_col: str = "full_text"):
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(pl.col(text_col).str.split(".").alias("sentence"))
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
    x = x.filter(pl.col("sentence_len") > 3)

    x = x.with_columns(
        pl.col("sentence")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("sentence_misspelled_cnt")
    )

    x = x.with_columns(
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .alias("only_sentence_len")
    )

    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
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

    df = df.to_pandas()
    return df




## === cell 13
word_features = [
    "word_len",
]


def Word_Features(x, text_col: str = "full_text"):
    print("Preprocess full_text and use spaces to seperate words fro the text")

    x = x.with_columns(pl.col(text_col).str.split(" ").alias("word"))
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(pl.col("word").str.len_chars().alias("word_len"))
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

    df = df.to_pandas()
    return df




## === cell 14
print("Precomputing full_text_pre for train/test (single pass preprocessing)")
train = train.with_columns(
    pl.col("full_text")
    .map_elements(dataPreprocessing, return_dtype=pl.String)
    .alias("full_text_pre")
)
test = test.with_columns(
    pl.col("full_text")
    .map_elements(dataPreprocessing, return_dtype=pl.String)
    .alias("full_text_pre")
)
train = train.with_columns(
    pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph_pre")
)
test = test.with_columns(
    pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph_pre")
)

clean_memory()




## === cell 15
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 5),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

train_tfid = vectorizer.fit_transform(train["full_text_pre"].to_list())




## === cell 16
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.10,
    max_df=0.85,
)

train_cnt = vectorizer_cnt.fit_transform(train["full_text_pre"].to_list())




## === cell 17
if (CFG.LOAD_FEATURES_FROM is not None) and (
    not os.path.exists(CFG.LOAD_FEATURES_FROM)
):
    print(
        f"WARNING: CFG.LOAD_FEATURES_FROM not found: {CFG.LOAD_FEATURES_FROM}. Will compute features instead."
    )
    CFG.LOAD_FEATURES_FROM = None

if (CFG.LOAD_MODELS_FROM is not None) and (not os.path.exists(CFG.LOAD_MODELS_FROM)):
    print(
        f"WARNING: CFG.LOAD_MODELS_FROM not found: {CFG.LOAD_MODELS_FROM}. Will train models instead."
    )
    CFG.LOAD_MODELS_FROM = None




## === cell 18
if CFG.LOAD_FEATURES_FROM is None:
    train_feats1 = Paragraph_Features(train, paragraph_col="paragraph_pre")
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train, text_col="full_text_pre")
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train, text_col="full_text_pre")
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

display(train_feats.head())
print("train_feats shape (engineered only):", train_feats.shape)




## === cell 19
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping

print("LightGBM Version: ", lgb.__version__)




## === cell 20
def quadratic_weighted_kappa(y_pred, dataset):
    y_true = dataset.get_label() + a
    y_pred_ = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred_, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_pred, dataset):
    y_true = dataset.get_label()  # labels are stored as (score - a)
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




## === cell 21
from scipy import sparse

categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()

ENGINEERED_FEATURES = [
    col
    for col in train_feats.columns
    if col not in categorical_columns + ["score", "essay_id"]
]
TARGET = "score"

TFIDF_FEATURES = [f"tfidf_{i}" for i in range(train_tfid.shape[1])]
CNT_FEATURES = [f"cnt_{i}" for i in range(train_cnt.shape[1])]
FEATURES = ENGINEERED_FEATURES + TFIDF_FEATURES + CNT_FEATURES

print("n_engineered_features:", len(ENGINEERED_FEATURES))
print("n_tfidf_features:", len(TFIDF_FEATURES))
print("n_cnt_features:", len(CNT_FEATURES))
print("n_features total:", len(FEATURES))

X_eng = np.clip(
    train_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False),
    0,
    10000,
)
X = sparse.hstack(
    [sparse.csr_matrix(X_eng), train_tfid.tocsr(), train_cnt.tocsr()], format="csr"
)
y = train_feats[TARGET].to_numpy(dtype=np.float32, copy=False) - a

clean_memory()




## === cell 22
def lightgbm_train_and_save(n_splits=15):
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=CFG.SEED, shuffle=True)
    splits = list(skf.split(np.zeros(len(y)), (y + a)))  # stratify on original labels

    for i, (train_index, valid_index) in enumerate(splits):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        params = dict(
            objective=qwk_obj,
            metric="None",
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
            num_threads=N_THREADS,
            force_col_wise=True,  # Speed/memory for wide sparse matrices; no semantic change.
        )

        train_data = lgb.Dataset(
            X[train_index], label=y[train_index], free_raw_data=True
        )
        valid_data = lgb.Dataset(
            X[valid_index],
            label=y[valid_index],
            reference=train_data,
            free_raw_data=True,
        )

        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[valid_data],
            valid_names=["valid"],
            feval=quadratic_weighted_kappa,
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
        )

        pickle.dump(model, open(f"LGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(X[valid_index], num_iteration=model.best_iteration)
        all_oof.append(oof + a)
        all_true.append(y[valid_index] + a)

        del train_data, valid_data, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for LightGBM = ", cv)

    if DO_PLOTS:
        cm = confusion_matrix(
            all_true, np.clip(all_oof, 1, 6).round(), labels=[x for x in range(1, 7)]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
        )
        disp.plot()
        plt.show()




## === cell 23
def _list_available_model_paths():
    paths = []
    if CFG.LOAD_MODELS_FROM:
        base = CFG.LOAD_MODELS_FROM
        for i in range(50):
            p = os.path.join(base, f"LGB_v{CFG.VER}_f{i}.pkl")
            if os.path.exists(p):
                paths.append(p)
            else:
                break
    else:
        for i in range(50):
            p = f"LGB_v{CFG.VER}_f{i}.pkl"
            if os.path.exists(p):
                paths.append(p)
            else:
                break
    return paths


available_model_paths = _list_available_model_paths()
if len(available_model_paths) == 0:
    print("Training LightGBM (no pre-trained models found)")
    lightgbm_train_and_save(n_splits=15)
    available_model_paths = _list_available_model_paths()

print("Available model count:", len(available_model_paths))
assert len(available_model_paths) > 0, "No models available after training/loading."




## === cell 24
if DO_PLOTS:
    model = pickle.load(open(available_model_paths[0], "rb"))

    try:
        imp = model.feature_importance()
    except Exception:
        imp = model.feature_importances_

    df_importance = pd.DataFrame(
        {"features_name": FEATURES, "importance": imp}
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




## === cell 25
test_tfid = vectorizer.transform(test["full_text_pre"].to_list())




## === cell 26
test_cnt = vectorizer_cnt.transform(test["full_text_pre"].to_list())




## === cell 27
test_feats1 = Paragraph_Features(test, paragraph_col="paragraph_pre")
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test, text_col="full_text_pre")
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test, text_col="full_text_pre")
test_feats3 = Word_aggregation(test_feats3)

test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")

print("Shape of test_feats (engineered only):", test_feats.shape)
display(test_feats.head())




## === cell 28
missing_eng = [c for c in ENGINEERED_FEATURES if c not in test_feats.columns]
extra_eng = [
    c for c in test_feats.columns if c not in (ENGINEERED_FEATURES + ["essay_id"])
]

for c in missing_eng:
    test_feats[c] = 0
if len(extra_eng) > 0:
    test_feats = test_feats.drop(columns=extra_eng)

test_feats = test_feats[["essay_id"] + ENGINEERED_FEATURES]

X_test_eng = np.clip(
    test_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False),
    0,
    10000,
)
X_test = sparse.hstack(
    [sparse.csr_matrix(X_test_eng), test_tfid.tocsr(), test_cnt.tocsr()], format="csr"
)

print(
    "Missing engineered added to test:",
    len(missing_eng),
    " Extra engineered dropped from test:",
    len(extra_eng),
)
clean_memory()




## === cell 29
preds = []
for j, path in enumerate(available_model_paths):
    print(f"Model {j+1}/{len(available_model_paths)}: {path}")
    model = pickle.load(open(path, "rb"))
    pred = (
        model.predict(X_test, num_iteration=getattr(model, "best_iteration", None)) + a
    )
    preds.append(pred)

pred1 = np.mean(preds, axis=0)

sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = np.clip(pred1, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print("Wrote: submission.csv")

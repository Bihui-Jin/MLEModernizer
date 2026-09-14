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
import pickle
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

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

from scipy import sparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

try:
    from IPython.display import display
except Exception:

    def display(x):
        print(x)


PLOT = False




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/lgbm-deberta-embeddings/"
    LOAD_FEATURES_FROM = "/kaggle/input/lgbm-deberta-embeddings/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 2
def _pick_base_path():
    candidates = [
        CFG.BASE_PATH,
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/",
        "/kaggle/data/",
    ]
    for base in candidates:
        if os.path.exists(os.path.join(base, "train.csv")) and os.path.exists(
            os.path.join(base, "test.csv")
        ):
            return base
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle paths."
    )


CFG.BASE_PATH = _pick_base_path()

if CFG.LOAD_FEATURES_FROM and (not os.path.exists(CFG.LOAD_FEATURES_FROM)):
    CFG.LOAD_FEATURES_FROM = None

if CFG.LOAD_MODELS_FROM and (not os.path.exists(CFG.LOAD_MODELS_FROM)):
    CFG.LOAD_MODELS_FROM = None

print("Using BASE_PATH:", CFG.BASE_PATH)
print("LOAD_FEATURES_FROM:", CFG.LOAD_FEATURES_FROM)
print("LOAD_MODELS_FROM:", CFG.LOAD_MODELS_FROM)




## === cell 3
Clean = True


def clean_memory():
    if Clean:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()


clean_memory()




## === cell 4
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 5
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id").reset_index(drop=True)
print("Shape of Train: ", df_train.shape)
display(df_train.head())




## === cell 6
if PLOT:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df_train["score"])
    plt.title("Distribution of Score")
    plt.xlabel("Score of Essay")
    plt.ylabel("Frequency")
    plt.show()




## === cell 7
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id").reset_index(drop=True)
print("Shape of Test: ", df_test.shape)
display(df_test.head())




## === cell 8
train_pl = pl.from_pandas(df_train)
test_pl = pl.from_pandas(df_test)

train = train_pl.with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = test_pl.with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))

schema_train = train.schema
schema_test = test.schema




## === cell 9
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




## === cell 10
c_re = re.compile("(%s)" % "|".join(map(re.escape, cList.keys())))
_html_re = re.compile(r"<.*?>")
_mention_re = re.compile(r"@\w+")
_apost_num_re = re.compile(r"'\d+")
_num_re = re.compile(r"\d+")
_http_re = re.compile(r"http\w+")
_space_re = re.compile(r"\s+")
_dots_re = re.compile(r"\.+")
_commas_re = re.compile(r"\,+")
_nonword_re = re.compile(r'[^\w\s.,;:""\'?!]')


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    return _html_re.sub(r"", x)


@lru_cache(maxsize=250_000)
def dataPreprocessing(x):
    x = str(x).lower()
    x = removeHTML(x)
    x = _mention_re.sub("", x)
    x = _apost_num_re.sub("", x)
    x = _num_re.sub("", x)
    x = _http_re.sub("", x)
    x = _space_re.sub(" ", x)
    x = expandContractions(x)
    x = _dots_re.sub(".", x)
    x = _commas_re.sub(",", x)
    x = _nonword_re.sub("", x)
    x = x.strip()
    return x




## === cell 11
_word_re = re.compile(r"[a-z]+")
_vowel_re = re.compile(r"[aeiouy]")


@lru_cache(maxsize=500_000)
def count_misspelled_words(text: str) -> int:
    s = str(text).lower()
    tokens = _word_re.findall(s)
    miss = 0
    for w in tokens:
        if len(w) <= 2:
            continue
        if len(set(w)) <= 2 and len(w) >= 5:
            miss += 1
            continue
        if not _vowel_re.search(w):
            miss += 1
            continue
    return miss




## === cell 12
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    stop_words = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")

_stop_set = set(stop_words)


def count_stop_words(text):
    text = str(text).split()
    return sum(1 for token in text if token in _stop_set)


def Cleaning(text):
    text = str(text).split()
    return " ".join([t for t in text if t not in _stop_set])




## === cell 13
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
    "paragraph_uni_word_cnt",
]


def Paragraph_Features(x: pl.DataFrame) -> pl.DataFrame:
    x = x.explode("paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
        .alias("paragraph")
    )
    x = x.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        pl.col("paragraph").str.count_matches(",").alias("paragraph_comma_cnt"),
        (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
            "paragraph_sentence_cnt"
        ),
        (pl.col("paragraph").str.count_matches(" ") + 1).alias("paragraph_word_cnt"),
        pl.col("paragraph")
        .str.split(" ")
        .list.n_unique()
        .alias("paragraph_uni_word_cnt"),
        pl.col("paragraph")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("paragraph_misspelled_cnt"),
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




## === cell 14
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x: pl.DataFrame) -> pl.DataFrame:
    if "full_text_clean" not in x.columns:
        x = x.with_columns(
            pl.col("full_text")
            .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
            .alias("full_text_clean")
        )
    x = x.with_columns(
        pl.col("full_text_clean").str.split(".").alias("sentence")
    ).explode("sentence")

    x = x.with_columns(pl.col("sentence").str.len_chars().alias("sentence_len")).filter(
        pl.col("sentence_len") > 3
    )

    x = x.with_columns(
        pl.col("sentence")
        .map_elements(count_misspelled_words, return_dtype=pl.Int64)
        .alias("sentence_misspelled_cnt"),
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.len_chars()
        .alias("only_sentence_len"),
        (pl.col("sentence").str.count_matches(" ") + 1).alias("sentence_word_cnt"),
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




## === cell 15
word_features = ["word_len"]


def Word_Features(x: pl.DataFrame) -> pl.DataFrame:
    if "full_text_clean" not in x.columns:
        x = x.with_columns(
            pl.col("full_text")
            .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
            .alias("full_text_clean")
        )
    x = x.with_columns(pl.col("full_text_clean").str.split(" ").alias("word")).explode(
        "word"
    )
    x = x.with_columns(pl.col("word").str.len_chars().alias("word_len")).filter(
        pl.col("word_len") > 0
    )
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




## === cell 16
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
    ngram_range=(2, 3),
    min_df=0.10,
    max_df=0.80,
)

t0 = time.time()
train_text_clean = train_pl["full_text"].map_elements(dataPreprocessing).to_list()
train_text_clean_no_stop = [Cleaning(t) for t in train_text_clean]
print(f"Text preprocessing (train) time: {time.time()-t0:.1f}s")

t0 = time.time()
train_tfid = vectorizer.fit_transform(train_text_clean)
train_cnt = vectorizer_cnt.fit_transform(train_text_clean_no_stop)
print(f"Vectorizers fit_transform time: {time.time()-t0:.1f}s")




## === cell 17
if CFG.LOAD_FEATURES_FROM is None:
    t0 = time.time()
    train_feats1 = Paragraph_aggregation(Paragraph_Features(train))

    train_pl_clean = train_pl.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
        .alias("full_text_clean")
    )
    train_feats2 = Sentence_aggregation(Sentence_Features(train_pl_clean))
    train_feats3 = Word_aggregation(Word_Features(train_pl_clean))
    print(f"Engineered feature extraction (train) time: {time.time()-t0:.1f}s")

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values

    emb_dir = "/kaggle/input/aes2-embedding-features/"
    paths = {
        "train_max": os.path.join(emb_dir, "train_max.npy"),
        "train_mean": os.path.join(emb_dir, "train_mean.npy"),
        "train_min": os.path.join(emb_dir, "train_min.npy"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        train_max = pd.DataFrame(np.load(paths["train_max"]))
        train_mean = pd.DataFrame(np.load(paths["train_mean"]))
        train_min = pd.DataFrame(np.load(paths["train_min"]))
        train_max.columns = [f"max_{i}" for i in range(train_max.shape[1])]
        train_min.columns = [f"min_{i}" for i in range(train_min.shape[1])]
        train_mean.columns = [f"mean_{i}" for i in range(train_mean.shape[1])]
        train_feats = pd.concat([train_feats, train_max, train_mean, train_min], axis=1)
    else:
        print("Embedding features not found; continuing without them.")
else:
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)




## === cell 18
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Loaded train_feats.csv")

display(train_feats.head())
print("train_feats shape:", train_feats.shape)




## === cell 19
print("LightGBM Version: ", lgb.__version__)




## === cell 20
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




## === cell 21
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
ENGINEERED_FEATURES = [
    c
    for c in train_feats.columns
    if c not in categorical_columns + ["score", "essay_id"]
]
TARGET = "score"

X_dense = np.clip(
    train_feats[ENGINEERED_FEATURES].fillna(0).to_numpy(dtype=np.float32), 0, 10000
)

X_sparse = sparse.hstack(
    [
        sparse.csr_matrix(X_dense),
        train_tfid.astype(np.float32),
        train_cnt.astype(np.float32),
    ],
    format="csr",
)

y = train_feats[TARGET].to_numpy(dtype=np.float32) - a

print("Engineered dense dims:", X_dense.shape)
print("Sparse train matrix shape:", X_sparse.shape)




## === cell 22
if CFG.LOAD_MODELS_FROM is None:
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

        oof_pred = model.predict(valid_x, num_iteration=model.best_iteration_)
        all_oof.append(oof_pred + a)
        all_true.append(valid_y + a)

        del train_x, train_y, valid_x, valid_y, oof_pred, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = cohen_kappa_score(true[0], oof[0].clip(1, 6).round(), weights="quadratic")
    print("CV Score for LightGBM = ", cv)

    if PLOT:
        cm = confusion_matrix(
            true[0], oof[0].clip(1, 6).round(), labels=[x for x in range(1, 7)]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
        )
        disp.plot()
        plt.show()




## === cell 23
if CFG.LOAD_MODELS_FROM:
    model = pickle.load(open(f"{CFG.LOAD_MODELS_FROM}LGB_v{CFG.VER}_f0.pkl", "rb"))
else:
    model = pickle.load(open(f"LGB_v{CFG.VER}_f0.pkl", "rb"))

try:
    imp = model.feature_importances_
    if PLOT and len(imp) >= len(ENGINEERED_FEATURES):
        df_importance = pd.DataFrame(
            {
                "features_name": ["dense_" + c for c in ENGINEERED_FEATURES],
                "importance": imp[: len(ENGINEERED_FEATURES)],
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
        plt.title(
            "Distribution of Feature Importance of LightGBM (engineered dense only)"
        )
        plt.xticks(rotation=90)
        plt.show()
except Exception as e:
    print("Skipped importance plot due to:", repr(e))




## === cell 24
def optimize_threshold(true, pred, steps=100):
    trial_x = [[], [], [], [], []]
    trial_y = [[], [], [], [], []]

    threshold = [1.5, 2.5, 3.5, 4.5, 5.5]
    pred2 = pd.cut(
        pred, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
    ).astype("int32")
    best = cohen_kappa_score(true, pred2, weights="quadratic")

    for k in range(5):
        for sign in [1, -1]:
            v = threshold[k]
            threshold2 = threshold.copy()
            stop = 0
            while stop < steps:
                v += sign * 0.001
                threshold2[k] = v
                pred2 = pd.cut(
                    pred, [-np.inf] + threshold2 + [np.inf], labels=[1, 2, 3, 4, 5, 6]
                ).astype("int32")
                metric = cohen_kappa_score(true, pred2, weights="quadratic")

                trial_x[k].append(v)
                trial_y[k].append(metric)

                if metric <= best:
                    stop += 1
                else:
                    stop = 0
                    best = metric
                    threshold = threshold2.copy()

    pred2 = pd.cut(
        pred, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
    ).astype("int32")
    best = cohen_kappa_score(true, pred2, weights="quadratic")
    threshold = [np.round(t, 3) for t in threshold]
    return best, threshold, trial_x, trial_y




## === cell 25
if CFG.LOAD_MODELS_FROM is None:
    best, threshold, trial_x, trial_y = optimize_threshold(true[0], oof[0])
    print("Best thresholds are: ", threshold)
    print("=> achieve Overall CV QWK score = ", best)
else:
    threshold = [1.623, 2.5, 3.424, 4.5, 5.585]




## === cell 26
t0 = time.time()
test_text_clean = test_pl["full_text"].map_elements(dataPreprocessing).to_list()
test_text_clean_no_stop = [Cleaning(t) for t in test_text_clean]
print(f"Text preprocessing (test) time: {time.time()-t0:.1f}s")

t0 = time.time()
test_tfid = vectorizer.transform(test_text_clean)
test_cnt = vectorizer_cnt.transform(test_text_clean_no_stop)
print(f"Vectorizers transform time: {time.time()-t0:.1f}s")




## === cell 27
pass




## === cell 28
t0 = time.time()
test_feats1 = Paragraph_aggregation(Paragraph_Features(test))

test_pl_clean = test_pl.with_columns(
    pl.col("full_text")
    .map_elements(dataPreprocessing, return_dtype=pl.Utf8)
    .alias("full_text_clean")
)
test_feats2 = Sentence_aggregation(Sentence_Features(test_pl_clean))
test_feats3 = Word_aggregation(Word_Features(test_pl_clean))
print(f"Feature extraction time: {time.time() - t0:.1f}s")




## === cell 29
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")

emb_dir = "/kaggle/input/aes2-embedding-features/"
paths = {
    "test_max": os.path.join(emb_dir, "test_max.npy"),
    "test_mean": os.path.join(emb_dir, "test_mean.npy"),
    "test_min": os.path.join(emb_dir, "test_min.npy"),
}
if all(os.path.exists(p) for p in paths.values()):
    test_max = pd.DataFrame(np.load(paths["test_max"]))
    test_mean = pd.DataFrame(np.load(paths["test_mean"]))
    test_min = pd.DataFrame(np.load(paths["test_min"]))
    test_max.columns = [f"max_{i}" for i in range(test_max.shape[1])]
    test_min.columns = [f"min_{i}" for i in range(test_min.shape[1])]
    test_mean.columns = [f"mean_{i}" for i in range(test_mean.shape[1])]
    test_feats = pd.concat([test_feats, test_max, test_mean, test_min], axis=1)
else:
    print("Test embedding features not found; continuing without them.")

print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())




## === cell 30
categorical_columns = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
ENGINEERED_FEATURES_TEST = [
    c for c in test_feats.columns if c not in categorical_columns + ["essay_id"]
]

missing = [c for c in ENGINEERED_FEATURES if c not in test_feats.columns]
if missing:
    for c in missing:
        test_feats[c] = 0.0
ENGINEERED_FEATURES_TEST = ENGINEERED_FEATURES  # enforce same order

X_test_dense = np.clip(
    test_feats[ENGINEERED_FEATURES_TEST].fillna(0).to_numpy(dtype=np.float32), 0, 10000
)
X_test_sparse = sparse.hstack(
    [
        sparse.csr_matrix(X_test_dense),
        test_tfid.astype(np.float32),
        test_cnt.astype(np.float32),
    ],
    format="csr",
)

preds = []
for i in range(15):
    print(f"Fold {i+1}")
    if CFG.LOAD_MODELS_FROM:
        model = pickle.load(
            open(f"{CFG.LOAD_MODELS_FROM}LGB_v{CFG.VER}_f{i}.pkl", "rb")
        )
    else:
        model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(X_test_sparse) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)




## === cell 31
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pd.cut(
    pred1, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
).astype("int32")
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
display(sub.head())
print("Saved: submission.csv")

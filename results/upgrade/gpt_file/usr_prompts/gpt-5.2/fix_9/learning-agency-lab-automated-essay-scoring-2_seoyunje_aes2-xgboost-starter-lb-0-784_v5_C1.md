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
import string
import re
from tqdm import tqdm
import pickle

import pandas as pd, numpy as np
import polars as pl  # kept installed parity; no longer used for heavy FE to avoid explode blow-up

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

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count()))

from scipy import sparse




## === cell 1
def _exists(path: str) -> bool:
    return path is not None and isinstance(path, str) and os.path.exists(path)




## === cell 2
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-xgboost-starter/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-xgboost-starter/train_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 3
if not os.path.exists(os.path.join(CFG.BASE_PATH, "train.csv")):
    alt = "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/"
    if os.path.exists(os.path.join(alt, "train.csv")):
        CFG.BASE_PATH = alt
    else:
        alt2 = "/kaggle/data/"
        if os.path.exists(os.path.join(alt2, "train.csv")):
            CFG.BASE_PATH = alt2



## === cell 4
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()



## === cell 5
if not _exists(CFG.LOAD_FEATURES_FROM):
    CFG.LOAD_FEATURES_FROM = None
if not _exists(CFG.LOAD_MODELS_FROM):
    CFG.LOAD_MODELS_FROM = None
else:
    if not CFG.LOAD_MODELS_FROM.endswith("/"):
        CFG.LOAD_MODELS_FROM += "/"




## === cell 6
def seed_everything():  # To proudce simliar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 7
plt.switch_backend("Agg")




## === cell 8
def _read_csv(path: str) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401

        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path)


read_kwargs = {}  # kept for downstream cells that expect this name

df_train = _read_csv(os.path.join(CFG.BASE_PATH, "train.csv"))
print("Shape of Train: ", df_train.shape)
print(df_train.head())



## === cell 9
df_test = _read_csv(os.path.join(CFG.BASE_PATH, "test.csv"))
print("Shape of Test: ", df_test.shape)
print(df_test.head())



## === cell 10
df_train["full_text"] = df_train["full_text"].fillna("").astype(str)
df_test["full_text"] = df_test["full_text"].fillna("").astype(str)



## === cell 11
_HTML_RE = re.compile(r"<.*?>")
_AT_RE = re.compile(r"@\w+")
_APOS_NUM_RE = re.compile(r"'\d+")
_NUM_RE = re.compile(r"\d+")
_HTTP_RE = re.compile(r"http\w+")
_WS_RE = re.compile(r"\s+")
_DOTS_RE = re.compile(r"\.+")
_COMMAS_RE = re.compile(r"\,+")


def removeHTML(x):
    return _HTML_RE.sub(r"", x)  # html -> ''


def dataPreprocessing(x):
    if x is None:
        x = ""
    if not isinstance(x, str):
        x = str(x)

    x = x.lower()
    x = removeHTML(x)

    x = _AT_RE.sub("", x)

    x = _APOS_NUM_RE.sub("", x)
    x = _NUM_RE.sub("", x)

    x = _HTTP_RE.sub("", x)

    x = _WS_RE.sub(" ", x)

    x = _DOTS_RE.sub(".", x)
    x = _COMMAS_RE.sub(".", x)
    x = x.strip()
    return x




## === cell 12
paragraph_features = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]

sentence_features = [
    "sentence_len",
    "sentence_word_cnt",
    "sentence_len_space_ratio",
    "sentence_word_space_ratio",
]

word_features = ["word_len"]


def _preprocess_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(_HTML_RE, "", regex=True)
    s = s.str.replace(_AT_RE, "", regex=True)
    s = s.str.replace(_APOS_NUM_RE, "", regex=True)
    s = s.str.replace(_NUM_RE, "", regex=True)
    s = s.str.replace(_HTTP_RE, "", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True)
    s = s.str.replace(_DOTS_RE, ".", regex=True)
    s = s.str.replace(_COMMAS_RE, ".", regex=True)
    s = s.str.strip()
    return s


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b != 0)
    return out


def _agg_stat(arr: np.ndarray, fn, default=np.nan):
    if arr.size == 0:
        return default
    return fn(arr)


def _fe_paragraph(df: pd.DataFrame) -> pd.DataFrame:
    essay_id = df["essay_id"].to_numpy()
    paragraphs = df["full_text"].str.split("\n\n")
    paragraphs = paragraphs.map(lambda lst: [dataPreprocessing(p) for p in lst])

    par_len = paragraphs.map(lambda lst: [len(p) for p in lst]).to_numpy()
    par_sent_cnt = paragraphs.map(
        lambda lst: [len(p.split(".")) for p in lst]
    ).to_numpy()
    par_word_cnt = paragraphs.map(
        lambda lst: [len(p.split(" ")) for p in lst]
    ).to_numpy()

    n = len(df)
    out = {"essay_id": essay_id}

    ge_len = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
    le_len = [100, 200]
    for t in ge_len:
        out[f"paragraph_{t}_cnt"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) >= t) for L in par_len),
            dtype=np.int32,
            count=n,
        )
    for t in le_len:
        out[f"paragraph_{t}_cnt_v2"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) <= t) for L in par_len),
            dtype=np.int32,
            count=n,
        )

    for t in [2, 4, 6, 8, 10]:
        out[f"paragraph_sentence_{t}_cnt"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) >= t) for L in par_sent_cnt),
            dtype=np.int32,
            count=n,
        )

    out["short_paragraph_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 300) & (np.asarray(L) > 100)) for L in par_len),
        dtype=np.int32,
        count=n,
    )
    out["mid_paragraph_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 500) & (np.asarray(L) > 300)) for L in par_len),
        dtype=np.int32,
        count=n,
    )
    out["long_paragraph_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 700) & (np.asarray(L) > 500)) for L in par_len),
        dtype=np.int32,
        count=n,
    )

    out["short_paragraph_sentence_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 4) & (np.asarray(L) > 2)) for L in par_sent_cnt),
        dtype=np.int32,
        count=n,
    )
    out["mid_paragraph_sentence_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 8) & (np.asarray(L) > 4)) for L in par_sent_cnt),
        dtype=np.int32,
        count=n,
    )
    out["long_paragraph_sentence_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 10) & (np.asarray(L) > 8)) for L in par_sent_cnt),
        dtype=np.int32,
        count=n,
    )

    for t in [30, 60, 90, 120]:
        out[f"paragraph_word_{t}_cnt"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) >= t) for L in par_word_cnt),
            dtype=np.int32,
            count=n,
        )

    out["short_paragraph_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 60) & (np.asarray(L) > 30)) for L in par_word_cnt),
        dtype=np.int32,
        count=n,
    )
    out["mid_paragraph_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 90) & (np.asarray(L) > 60)) for L in par_word_cnt),
        dtype=np.int32,
        count=n,
    )
    out["long_paragraph_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 120) & (np.asarray(L) > 90)) for L in par_word_cnt),
        dtype=np.int32,
        count=n,
    )

    out["paragraph_cnt"] = np.fromiter(
        (len(L) for L in par_len), dtype=np.int32, count=n
    )

    return pd.DataFrame(out).sort_values("essay_id").reset_index(drop=True)


def _fe_sentence(df: pd.DataFrame) -> pd.DataFrame:
    essay_id = df["essay_id"].to_numpy()
    text = _preprocess_series(df["full_text"])

    sent_lists = text.str.split(".")
    sent_space_l = sent_lists.map(lambda lst: [s.count(" ") for s in lst]).to_numpy()
    sent_len_l = sent_lists.map(lambda lst: [len(s) for s in lst]).to_numpy()
    sent_word_cnt_l = sent_lists.map(
        lambda lst: [len(s.split(" ")) for s in lst]
    ).to_numpy()

    n = len(df)
    out = {"essay_id": essay_id}

    f_space = [None] * n
    f_len = [None] * n
    f_wc = [None] * n
    for i in range(n):
        sc = np.asarray(sent_space_l[i], dtype=np.int32)
        sl = np.asarray(sent_len_l[i], dtype=np.int32)
        sw = np.asarray(sent_word_cnt_l[i], dtype=np.int32)
        m = (sc > 0) & (sl > 3)
        f_space[i] = sc[m]
        f_len[i] = sl[m]
        f_wc[i] = sw[m]

    ge_len = [30, 40, 50, 60, 70, 80, 100, 150]
    le_len = [10, 20, 30]
    for t in ge_len:
        out[f"sentence_{t}_cnt"] = np.fromiter(
            (np.sum(f_len[i] >= t) for i in range(n)), dtype=np.int32, count=n
        )
    for t in le_len:
        out[f"sentence_{t}_cnt_v2"] = np.fromiter(
            (np.sum(f_len[i] <= t) for i in range(n)), dtype=np.int32, count=n
        )

    out["short_sentence_cnt"] = np.fromiter(
        (np.sum((f_len[i] <= 50) & (f_len[i] > 30)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )
    out["mid_sentence_cnt"] = np.fromiter(
        (np.sum((f_len[i] <= 70) & (f_len[i] > 50)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )
    out["long_sentence_cnt"] = np.fromiter(
        (np.sum((f_len[i] <= 100) & (f_len[i] > 70)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )

    for t in [5, 10, 15, 20]:
        out[f"sentence_word_{t}_cnt"] = np.fromiter(
            (np.sum(f_wc[i] >= t) for i in range(n)), dtype=np.int32, count=n
        )

    out["short_sentence_word_cnt"] = np.fromiter(
        (np.sum((f_wc[i] <= 10) & (f_wc[i] > 5)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )
    out["mid_sentence_word_cnt"] = np.fromiter(
        (np.sum((f_wc[i] <= 15) & (f_wc[i] > 10)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )
    out["long_sentence_word_cnt"] = np.fromiter(
        (np.sum((f_wc[i] <= 20) & (f_wc[i] > 15)) for i in range(n)),
        dtype=np.int32,
        count=n,
    )

    out["sentence_cnt"] = np.fromiter(
        (len(f_len[i]) for i in range(n)), dtype=np.int32, count=n
    )

    out["sentence_len_max"] = np.fromiter(
        (_agg_stat(f_len[i].astype(np.float64, copy=False), np.max) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_mean"] = np.fromiter(
        (_agg_stat(f_len[i].astype(np.float64, copy=False), np.mean) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_min"] = np.fromiter(
        (_agg_stat(f_len[i].astype(np.float64, copy=False), np.min) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_std"] = np.fromiter(
        (_agg_stat(f_len[i].astype(np.float64, copy=False), np.std) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_sum"] = np.fromiter(
        (
            _agg_stat(f_len[i].astype(np.float64, copy=False), np.sum, default=0.0)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )

    out["sentence_word_cnt_max"] = np.fromiter(
        (_agg_stat(f_wc[i].astype(np.float64, copy=False), np.max) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_cnt_mean"] = np.fromiter(
        (_agg_stat(f_wc[i].astype(np.float64, copy=False), np.mean) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_cnt_min"] = np.fromiter(
        (_agg_stat(f_wc[i].astype(np.float64, copy=False), np.min) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_cnt_std"] = np.fromiter(
        (_agg_stat(f_wc[i].astype(np.float64, copy=False), np.std) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_cnt_sum"] = np.fromiter(
        (
            _agg_stat(f_wc[i].astype(np.float64, copy=False), np.sum, default=0.0)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )

    def _ratio_stats(arr_num, arr_den):
        if arr_num.size == 0:
            empty = np.array([], dtype=np.float64)
            return empty
        return arr_num.astype(np.float64) / arr_den.astype(np.float64)

    out["sentence_len_space_ratio_max"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_len[i], f_space[i]), np.max) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_space_ratio_mean"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_len[i], f_space[i]), np.mean) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_space_ratio_min"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_len[i], f_space[i]), np.min) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_space_ratio_std"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_len[i], f_space[i]), np.std) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_len_space_ratio_sum"] = np.fromiter(
        (
            _agg_stat(_ratio_stats(f_len[i], f_space[i]), np.sum, default=0.0)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )

    out["sentence_word_space_ratio_max"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_wc[i], f_space[i]), np.max) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_space_ratio_mean"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_wc[i], f_space[i]), np.mean) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_space_ratio_min"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_wc[i], f_space[i]), np.min) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_space_ratio_std"] = np.fromiter(
        (_agg_stat(_ratio_stats(f_wc[i], f_space[i]), np.std) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["sentence_word_space_ratio_sum"] = np.fromiter(
        (
            _agg_stat(_ratio_stats(f_wc[i], f_space[i]), np.sum, default=0.0)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )

    return pd.DataFrame(out).sort_values("essay_id").reset_index(drop=True)


def _fe_word(df: pd.DataFrame) -> pd.DataFrame:
    essay_id = df["essay_id"].to_numpy()
    text = _preprocess_series(df["full_text"])

    sent_lists = text.str.split(".")
    word_lists = sent_lists.map(lambda sents: [w for s in sents for w in s.split(" ")])

    word_lens = word_lists.map(
        lambda lst: [len(w) for w in lst if len(w) > 0]
    ).to_numpy()

    n = len(df)
    out = {"essay_id": essay_id}

    for t in [3, 4, 5, 6, 8, 10, 15]:
        out[f"word_{t}_cnt"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) >= t) for L in word_lens),
            dtype=np.int32,
            count=n,
        )
    for t in [2, 3]:
        out[f"word_{t}_cnt_v2"] = np.fromiter(
            (np.sum(np.asarray(L, dtype=np.int32) <= t) for L in word_lens),
            dtype=np.int32,
            count=n,
        )

    out["short_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 4) & (np.asarray(L) > 2)) for L in word_lens),
        dtype=np.int32,
        count=n,
    )
    out["mid_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 7) & (np.asarray(L) > 4)) for L in word_lens),
        dtype=np.int32,
        count=n,
    )
    out["long_word_cnt"] = np.fromiter(
        (np.sum((np.asarray(L) <= 10) & (np.asarray(L) > 7)) for L in word_lens),
        dtype=np.int32,
        count=n,
    )

    out["word_cnt"] = np.fromiter((len(L) for L in word_lens), dtype=np.int32, count=n)

    out["word_len_max"] = np.fromiter(
        (_agg_stat(np.asarray(word_lens[i], dtype=np.int32), np.max) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["word_len_mean"] = np.fromiter(
        (
            _agg_stat(np.asarray(word_lens[i], dtype=np.int32), np.mean)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )
    out["word_len_min"] = np.fromiter(
        (_agg_stat(np.asarray(word_lens[i], dtype=np.int32), np.min) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["word_len_std"] = np.fromiter(
        (_agg_stat(np.asarray(word_lens[i], dtype=np.int32), np.std) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    out["word_len_sum"] = np.fromiter(
        (
            _agg_stat(np.asarray(word_lens[i], dtype=np.int32), np.sum, default=0.0)
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )
    out["word_len_q1"] = np.fromiter(
        (
            _agg_stat(
                np.asarray(word_lens[i], dtype=np.int32), lambda a: np.quantile(a, 0.25)
            )
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )
    out["word_len_q3"] = np.fromiter(
        (
            _agg_stat(
                np.asarray(word_lens[i], dtype=np.int32), lambda a: np.quantile(a, 0.75)
            )
            for i in range(n)
        ),
        dtype=np.float64,
        count=n,
    )

    w2 = out["word_2_cnt_v2"].astype(np.float64)
    w3 = out["word_3_cnt_v2"].astype(np.float64)
    for t in [3, 4, 5, 6, 8, 10, 15]:
        out[f"word_2_{t}_cnt_ratio"] = _safe_div(
            out[f"word_{t}_cnt"].astype(np.float64), w2
        )
        out[f"word_3_{t}_cnt_ratio"] = _safe_div(
            out[f"word_{t}_cnt"].astype(np.float64), w3
        )

    for t in [2, 3]:
        denom = out[f"word_{t}_cnt_v2"].astype(np.float64)
        out[f"short_word_ratio_{t}"] = _safe_div(
            out["short_word_cnt"].astype(np.float64), denom
        )
        out[f"mid_word_ratio_{t}"] = _safe_div(
            out["mid_word_cnt"].astype(np.float64), denom
        )
        out[f"long_word_ratio_{t}"] = _safe_div(
            out["long_word_cnt"].astype(np.float64), denom
        )

    return pd.DataFrame(out).sort_values("essay_id").reset_index(drop=True)


def Paragraph_Features(x):
    return x


def Paragraph_aggregation(x):
    return _fe_paragraph(x)


def Sentence_Features(x):
    return x


def Sentence_aggregation(x):
    return _fe_sentence(x)


def Word_Features(x):
    return x


def Word_aggregation(x):
    return _fe_word(x)




## === cell 13
pass



## === cell 14
pass



## === cell 15
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

train_text = df_train["full_text"].tolist()
train_tfid = vectorizer.fit_transform(train_text).tocsr()
tfidf_feature_names = [f"tfidf_{i}" for i in range(train_tfid.shape[1])]



## === cell 16
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



## === cell 17
if CFG.LOAD_FEATURES_FROM is None:
    train_base = df_train[["essay_id", "full_text"]].copy()

    train_feats1 = Paragraph_Features(train_base)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train_base)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train_base)
    train_feats3 = Word_aggregation(train_feats3)

    def _dedup_sort(df_in: pd.DataFrame) -> pd.DataFrame:
        return (
            df_in.sort_values("essay_id")
            .drop_duplicates("essay_id", keep="first")
            .reset_index(drop=True)
        )

    train_feats1 = _dedup_sort(train_feats1)
    train_feats2 = _dedup_sort(train_feats2)
    train_feats3 = _dedup_sort(train_feats3)

    train_feats = train_feats1.merge(
        train_feats2, on="essay_id", how="left", validate="one_to_one"
    )
    train_feats = train_feats.merge(
        train_feats3, on="essay_id", how="left", validate="one_to_one"
    )

    score_map = df_train.set_index("essay_id")["score"]
    train_feats["score"] = train_feats["essay_id"].map(score_map).astype(int)
else:
    train_feats = None



## === cell 18
if CFG.LOAD_FEATURES_FROM is None:
    print(
        "Save train_feats.csv (engineered features only; TF-IDF kept sparse in-memory)"
    )
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv (engineered features only; TF-IDF computed from text)")
    train_feats = _read_csv(CFG.LOAD_FEATURES_FROM)



## === cell 19
print(train_feats.head())



## === cell 20
obj_cols = [
    c
    for c in train_feats.columns
    if c not in ["essay_id", "score"] and train_feats[c].dtype == "object"
]
if obj_cols:
    train_feats[obj_cols] = train_feats[obj_cols].apply(pd.to_numeric, errors="coerce")



## === cell 21
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import xgboost as xgb

print("XGBoost Version: ", xgb.__version__)



## === cell 22
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()

FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## === cell 23
def quadratic_weighted_kappa(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk




## === cell 24
def xgboost():
    all_oof = []
    all_true = []

    feat_df = train_feats.set_index("essay_id").loc[df_train["essay_id"].values]
    X_num = np.clip(
        feat_df[FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False), 0, 10000
    )
    X = sparse.hstack([sparse.csr_matrix(X_num), train_tfid], format="csr")
    y = feat_df[TARGET].to_numpy()

    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    for i, (train_index, valid_index) in enumerate(skf.split(X_num, y)):

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
            n_jobs=os.cpu_count(),
        )

        X_tr = X[train_index]
        y_tr = y[train_index]
        X_va = X[valid_index]
        y_va = y[valid_index]

        try:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=100,
                verbose=50,
            )
        except TypeError:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
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




## === cell 25
def _all_folds_exist(base: str) -> bool:
    return base is not None and all(
        _exists(f"{base}XGB_v{CFG.VER}_f{i}.pkl") for i in range(5)
    )


if _all_folds_exist(CFG.LOAD_MODELS_FROM):
    print("Using pre-trained models from:", CFG.LOAD_MODELS_FROM)
elif all(_exists(f"XGB_v{CFG.VER}_f{i}.pkl") for i in range(5)):
    print("Using local pre-trained models from working dir.")
else:
    print("Training XGBoost (no pre-trained models found)")
    xgboost()



## === cell 26
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
            "importance": model.feature_importances_[: len(FEATURES)],
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



## === cell 27
test_text = df_test["full_text"].tolist()
test_tfid = vectorizer.transform(test_text).tocsr()



## === cell 28
pass



## === cell 29
test_base = df_test[["essay_id", "full_text"]].copy()

test_feats1 = Paragraph_Features(test_base)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test_base)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test_base)
test_feats3 = Word_aggregation(test_feats3)



## === cell 30
sample_path = os.path.join(CFG.BASE_PATH, "sample_submission.csv")
sample_sub = _read_csv(sample_path)[["essay_id"]].copy()


def _dedup_sort(df_in: pd.DataFrame) -> pd.DataFrame:
    return (
        df_in.sort_values("essay_id")
        .drop_duplicates("essay_id", keep="first")
        .reset_index(drop=True)
    )


test_feats1 = _dedup_sort(test_feats1)
test_feats2 = _dedup_sort(test_feats2)
test_feats3 = _dedup_sort(test_feats3)

test_feats = sample_sub.merge(
    test_feats1, on="essay_id", how="left", validate="one_to_one"
)
test_feats = test_feats.merge(
    test_feats2, on="essay_id", how="left", validate="one_to_one"
)
test_feats = test_feats.merge(
    test_feats3, on="essay_id", how="left", validate="one_to_one"
)

print("Shape of test_feats:", test_feats.shape)
print(test_feats.head())

obj_cols_t = [
    c for c in test_feats.columns if c != "essay_id" and test_feats[c].dtype == "object"
]
if obj_cols_t:
    test_feats[obj_cols_t] = test_feats[obj_cols_t].apply(
        pd.to_numeric, errors="coerce"
    )

for col in FEATURES:
    if col not in test_feats.columns:
        test_feats[col] = 0

feat_test_aligned = test_feats.set_index("essay_id").loc[df_test["essay_id"].values]
X_num_test = np.clip(
    feat_test_aligned[FEATURES].fillna(0).to_numpy(dtype=np.float32, copy=False),
    0,
    10000,
)
test_x = sparse.hstack([sparse.csr_matrix(X_num_test), test_tfid], format="csr")

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

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub[TARGET] = np.clip(pred, 1, 6).round().astype(int)

assert sub.columns.tolist() == ["essay_id", "score"]
assert len(sub) == len(_read_csv(os.path.join(CFG.BASE_PATH, "sample_submission.csv")))

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Wrote: submission.csv")

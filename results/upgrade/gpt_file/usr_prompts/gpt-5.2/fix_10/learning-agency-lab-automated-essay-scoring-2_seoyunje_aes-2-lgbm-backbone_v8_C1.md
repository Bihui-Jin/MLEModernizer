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
import re
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from scipy import sparse

import lightgbm as lgb
from lightgbm import early_stopping

import warnings

warnings.filterwarnings("ignore")

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        return x


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

DO_PLOTS = False
Clean = True


class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = "/kaggle/input/aes2-lgbm-backbone/"
    LOAD_FEATURES_FROM = "/kaggle/input/aes2-lgbm-backbone/train_feats_1.csv"
    LOAD_TEST_FEATURES_FROM = "/kaggle/input/aes2-lgbm-backbone/test_feats_1.csv"
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


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

print("LightGBM Version: ", lgb.__version__)




## === cell 1
def _path_exists(p: str | None) -> bool:
    return (p is not None) and os.path.exists(p)


def _ensure_dir_suffix(p: str | None) -> str | None:
    if p is None:
        return None
    if os.path.isdir(p) and (not p.endswith("/")):
        return p + "/"
    return p


def _autofind_backbone_assets() -> None:
    global CFG
    candidates = [
        "/kaggle/input/aes2-lgbm-backbone/",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/",
        "/kaggle/data/",
    ]

    if CFG.LOAD_MODELS_FROM:
        candidates = [CFG.LOAD_MODELS_FROM] + candidates
    candidates = [c for c in candidates if c is not None]

    found_dir = None
    for base in candidates:
        base = _ensure_dir_suffix(base)
        if base and os.path.isdir(base):
            if os.path.exists(os.path.join(base, f"LGB_v{CFG.VER}_f0.pkl")):
                found_dir = base
                break
    if found_dir is not None:
        CFG.LOAD_MODELS_FROM = found_dir
    else:
        CFG.LOAD_MODELS_FROM = _ensure_dir_suffix(CFG.LOAD_MODELS_FROM)

    if not _path_exists(CFG.LOAD_FEATURES_FROM):
        alt = os.path.join(CFG.BASE_PATH, f"train_feats_{CFG.VER}.csv")
        CFG.LOAD_FEATURES_FROM = alt if os.path.exists(alt) else None

    if not _path_exists(CFG.LOAD_TEST_FEATURES_FROM):
        alt = os.path.join(CFG.BASE_PATH, f"test_feats_{CFG.VER}.csv")
        CFG.LOAD_TEST_FEATURES_FROM = alt if os.path.exists(alt) else None


_autofind_backbone_assets()

CAN_LOAD_FEATURES = _path_exists(CFG.LOAD_FEATURES_FROM)
CAN_LOAD_MODELS = _path_exists(CFG.LOAD_MODELS_FROM) if CFG.LOAD_MODELS_FROM else False
CAN_LOAD_TEST_FEATURES = _path_exists(CFG.LOAD_TEST_FEATURES_FROM)

if not CAN_LOAD_FEATURES:
    CFG.LOAD_FEATURES_FROM = None
if not CAN_LOAD_TEST_FEATURES:
    CFG.LOAD_TEST_FEATURES_FROM = None
if not CAN_LOAD_MODELS:
    CFG.LOAD_MODELS_FROM = None

print("CAN_LOAD_FEATURES:", CAN_LOAD_FEATURES, "-> using", CFG.LOAD_FEATURES_FROM)
print(
    "CAN_LOAD_TEST_FEATURES:",
    CAN_LOAD_TEST_FEATURES,
    "-> using",
    CFG.LOAD_TEST_FEATURES_FROM,
)
print("CAN_LOAD_MODELS:", CAN_LOAD_MODELS, "-> using", CFG.LOAD_MODELS_FROM)



## === cell 2
df_train = pd.read_csv(
    CFG.BASE_PATH + "train.csv",
    usecols=["essay_id", "full_text", "score"],
    dtype={"essay_id": "string", "full_text": "string", "score": "int8"},
)
print("Shape of Train: ", df_train.shape)
display(df_train.head())

df_test = pd.read_csv(
    CFG.BASE_PATH + "test.csv",
    usecols=["essay_id", "full_text"],
    dtype={"essay_id": "string", "full_text": "string"},
)
print("Shape of Test: ", df_test.shape)
display(df_test.head())



## === cell 3
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
    "I'll've": "I would have",
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
_html_re = re.compile(r"<.*?>")
_at_re = re.compile(r"@\w+")
_quote_digit_re = re.compile(r"'\d+")
_digit_re = re.compile(r"\d+")
_http_re = re.compile(r"http\w+")
_space_re = re.compile(r"\s+")
_dots_re = re.compile(r"\.+")
_commas_re = re.compile(r"\,+")
_nonword_re = re.compile(r"[^\w\s.,;:\"\"''?!]")


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    return _html_re.sub(r"", x)


def dataPreprocessing(x):
    x = str(x).lower()
    x = removeHTML(x)
    x = _at_re.sub("", x)
    x = _quote_digit_re.sub("", x)
    x = _digit_re.sub("", x)
    x = _http_re.sub("", x)
    x = _space_re.sub(" ", x)
    x = expandContractions(x)
    x = _dots_re.sub(".", x)
    x = _commas_re.sub(",", x)
    x = _nonword_re.sub("", x)
    return x.strip()




## === cell 4
_word_re = re.compile(r"[a-zA-Z]+")
_triple_re = re.compile(r"(.)\1\1")


@lru_cache(maxsize=1_000_000)
def _count_misspelled_words_cached(text: str) -> int:
    text = str(text)
    words = _word_re.findall(text.lower())
    if not words:
        return 0
    vowels = set("aeiou")
    miss = 0
    for w in words:
        if len(w) >= 18:
            miss += 1
            continue
        if (len(w) >= 5) and (not any(ch in vowels for ch in w)):
            miss += 1
            continue
        if _triple_re.search(w):
            miss += 1
            continue
    return miss


def count_misspelled_words(text: str) -> int:
    return _count_misspelled_words_cached(str(text))


def _prime_misspell_cache(unique_texts: pd.Series) -> None:
    arr = unique_texts.astype(str).unique()
    for t in arr:
        _count_misspelled_words_cached(t)




## === cell 5
def _group_sum_bool(keys_code: np.ndarray, cond: np.ndarray, n_keys: int) -> np.ndarray:
    out = np.zeros(n_keys, dtype=np.int64)
    np.add.at(out, keys_code, cond.astype(np.int64))
    return out


def _add_quantiles(
    out_df: pd.DataFrame, g: pd.core.groupby.generic.DataFrameGroupBy, feats: list[str]
) -> pd.DataFrame:
    q = g[feats].quantile([0.25, 0.75]).unstack(level=-1)
    for feat in feats:
        out_df[f"{feat}_q1"] = q[(feat, 0.25)]
        out_df[f"{feat}_q3"] = q[(feat, 0.75)]
    return out_df


paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]

sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]
word_features = ["word_len"]




## === cell 6
def _build_paragraph_df(df_in: pd.DataFrame) -> pd.DataFrame:
    tmp = df_in[["essay_id", "full_text"]]
    tmp = tmp.copy()
    tmp["paragraph"] = tmp["full_text"].astype(str).str.split("\n\n")
    tmp = tmp.explode("paragraph", ignore_index=False)
    tmp["paragraph"] = tmp["paragraph"].map(dataPreprocessing)

    par = tmp["paragraph"].astype(str)
    tmp["paragraph_len"] = par.str.len().astype(np.int64)
    tmp["paragraph_comma_cnt"] = par.str.count(",", flags=0).astype(np.int64)
    tmp["paragraph_sentence_cnt"] = (par.str.count(r"\.") + 1).astype(np.int64)
    tmp["paragraph_word_cnt"] = (par.str.count(" ") + 1).astype(np.int64)

    _prime_misspell_cache(par)
    tmp["paragraph_misspelled_cnt"] = par.map(count_misspelled_words).astype(np.int64)
    return tmp[["essay_id", "paragraph", *paragraph_features]]


def Paragraph_aggregation_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
    tmp = _build_paragraph_df(df_in)
    keys = tmp["essay_id"].astype("string")
    codes, uniques = pd.factorize(keys, sort=True)
    n_keys = len(uniques)

    out = pd.DataFrame(index=uniques.astype("string"))

    par_len = tmp["paragraph_len"].to_numpy()
    sent_cnt = tmp["paragraph_sentence_cnt"].to_numpy()
    word_cnt = tmp["paragraph_word_cnt"].to_numpy()
    comma_cnt = tmp["paragraph_comma_cnt"].to_numpy()
    miss_cnt = tmp["paragraph_misspelled_cnt"].to_numpy()

    for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]:
        out[f"paragraph_{i}_cnt"] = _group_sum_bool(codes, par_len >= i, n_keys)
    for i in [100, 200]:
        out[f"paragraph_{i}_cnt_v2"] = _group_sum_bool(codes, par_len <= i, n_keys)

    out["short_paragraph_cnt"] = _group_sum_bool(
        codes, (par_len <= 300) & (par_len > 100), n_keys
    )
    out["mid_paragraph_cnt"] = _group_sum_bool(
        codes, (par_len <= 500) & (par_len > 300), n_keys
    )
    out["long_paragraph_cnt"] = _group_sum_bool(
        codes, (par_len <= 700) & (par_len > 500), n_keys
    )

    for i in [2, 4, 6, 8, 10]:
        out[f"paragraph_sentence_{i}_cnt"] = _group_sum_bool(
            codes, sent_cnt >= i, n_keys
        )

    out["short_paragraph_sentence_cnt"] = _group_sum_bool(
        codes, (sent_cnt <= 4) & (sent_cnt > 2), n_keys
    )
    out["mid_paragraph_sentence_cnt"] = _group_sum_bool(
        codes, (sent_cnt <= 8) & (sent_cnt > 4), n_keys
    )
    out["long_paragraph_sentence_cnt"] = _group_sum_bool(
        codes, (sent_cnt <= 10) & (sent_cnt > 8), n_keys
    )

    for i in [20, 40, 60, 90, 120]:
        out[f"paragraph_word_{i}_cnt"] = _group_sum_bool(codes, word_cnt >= i, n_keys)

    out["short_paragraph_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 40) & (word_cnt > 20), n_keys
    )
    out["mid_paragraph_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 90) & (word_cnt > 40), n_keys
    )
    out["long_paragraph_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 120) & (word_cnt > 90), n_keys
    )

    for i in [1, 2, 3, 4, 5]:
        out[f"paragraph_comma_{i}_cnt"] = _group_sum_bool(codes, comma_cnt >= i, n_keys)

    for i in [4, 8, 12, 16]:
        out[f"paragraph_misspelled_{i}_cnt"] = _group_sum_bool(
            codes, miss_cnt >= i, n_keys
        )
    for i in [2, 4]:
        out[f"paragraph_misspelled_{i}_cnt_v2"] = _group_sum_bool(
            codes, miss_cnt <= i, n_keys
        )

    out["short_paragraph_misspelled_cnt"] = _group_sum_bool(
        codes, (miss_cnt <= 8) & (miss_cnt > 4), n_keys
    )
    out["mid_paragraph_misspelled_cnt"] = _group_sum_bool(
        codes, (miss_cnt <= 12) & (miss_cnt > 8), n_keys
    )
    out["long_paragraph_misspelled_cnt"] = _group_sum_bool(
        codes, (miss_cnt <= 16) & (miss_cnt > 12), n_keys
    )

    out["paragraph_cnt"] = np.bincount(codes, minlength=n_keys).astype(np.int64)

    g = tmp.groupby(keys, sort=True)
    agg_basic = g[paragraph_features].agg(["max", "mean", "min", "std", "sum"])
    agg_basic.columns = [f"{c[0]}_{c[1]}" for c in agg_basic.columns]
    out = out.join(agg_basic, how="left")

    out = _add_quantiles(out, g, paragraph_features)

    out = (
        out.reset_index()
        .rename(columns={"index": "essay_id"})
        .sort_values("essay_id")
        .reset_index(drop=True)
    )
    return out




## === cell 7
def _build_sentence_df(df_in: pd.DataFrame, cleaned_col: str) -> pd.DataFrame:
    tmp = df_in[["essay_id", cleaned_col]].copy()
    tmp["sentence"] = tmp[cleaned_col].astype(str).str.split(".")
    tmp = tmp.explode("sentence", ignore_index=False)

    sent = tmp["sentence"].astype(str)
    tmp["sentence_len"] = sent.str.len().astype(np.int64)
    tmp = tmp[tmp["sentence_len"] > 3].copy()

    sent = tmp["sentence"].astype(str)

    _prime_misspell_cache(sent)
    tmp["sentence_misspelled_cnt"] = sent.map(count_misspelled_words).astype(np.int64)

    tmp["only_sentence_len"] = (
        sent.str.replace(" ", "", regex=False).str.len().astype(np.int64)
    )
    tmp["sentence_word_cnt"] = (sent.str.count(" ") + 1).astype(np.int64)
    return tmp[["essay_id", "sentence", "only_sentence_len", *sentence_features]]


def Sentence_aggregation_from_df(df_in: pd.DataFrame, cleaned_col: str) -> pd.DataFrame:
    tmp = _build_sentence_df(df_in, cleaned_col=cleaned_col)
    keys = tmp["essay_id"].astype("string")
    codes, uniques = pd.factorize(keys, sort=True)
    n_keys = len(uniques)

    out = pd.DataFrame(index=uniques.astype("string"))

    sent_len = tmp["sentence_len"].to_numpy()
    only_len = tmp["only_sentence_len"].to_numpy()
    word_cnt = tmp["sentence_word_cnt"].to_numpy()

    for i in [40, 60, 70, 80, 100, 120, 140]:
        out[f"sentence_{i}_cnt"] = _group_sum_bool(codes, sent_len >= i, n_keys)
    for i in [10, 20, 30]:
        out[f"sentence_{i}_cnt_v2"] = _group_sum_bool(codes, sent_len <= i, n_keys)

    out["short_sentence_cnt"] = _group_sum_bool(
        codes, (sent_len <= 70) & (sent_len > 40), n_keys
    )
    out["mid_sentence_cnt"] = _group_sum_bool(
        codes, (sent_len <= 100) & (sent_len > 70), n_keys
    )
    out["long_sentence_cnt"] = _group_sum_bool(
        codes, (sent_len <= 140) & (sent_len > 100), n_keys
    )

    for i in [40, 60, 80, 100, 120]:
        out[f"only_sentence_{i}_cnt"] = _group_sum_bool(codes, only_len >= i, n_keys)

    out["short_only_sentence_cnt"] = _group_sum_bool(
        codes, (only_len <= 60) & (only_len > 40), n_keys
    )
    out["mid_only_sentence_cnt"] = _group_sum_bool(
        codes, (only_len <= 100) & (only_len > 60), n_keys
    )
    out["long_only_sentence_cnt"] = _group_sum_bool(
        codes, (only_len <= 120) & (only_len > 100), n_keys
    )

    for i in [10, 15, 20, 25]:
        out[f"sentence_word_{i}_cnt"] = _group_sum_bool(codes, word_cnt >= i, n_keys)

    out["short_sentence_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 15) & (word_cnt > 10), n_keys
    )
    out["mid_sentence_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 20) & (word_cnt > 15), n_keys
    )
    out["long_sentence_word_cnt"] = _group_sum_bool(
        codes, (word_cnt <= 25) & (word_cnt > 20), n_keys
    )

    out["sentence_cnt"] = np.bincount(codes, minlength=n_keys).astype(np.int64)

    g = tmp.groupby(keys, sort=True)
    agg_basic = g[sentence_features].agg(["max", "mean", "min", "std", "sum"])
    agg_basic.columns = [f"{c[0]}_{c[1]}" for c in agg_basic.columns]
    out = out.join(agg_basic, how="left")

    out = _add_quantiles(out, g, sentence_features)

    out = (
        out.reset_index()
        .rename(columns={"index": "essay_id"})
        .sort_values("essay_id")
        .reset_index(drop=True)
    )

    denom = out["sentence_cnt"].replace(0, np.nan)
    for i in [40, 60, 70, 80, 100, 120, 140]:
        out[f"sentence_{i}_cnt_ratio"] = out[f"sentence_{i}_cnt"] / denom
    out["short_sentence_cnt_ratio"] = out["short_sentence_cnt"] / denom
    out["mid_sentence_cnt_ratio"] = out["mid_sentence_cnt"] / denom
    out["long_sentence_cnt_ratio"] = out["long_sentence_cnt"] / denom

    return out




## === cell 8
def _build_word_df(df_in: pd.DataFrame, cleaned_col: str) -> pd.DataFrame:
    tmp = df_in[["essay_id", cleaned_col]].copy()
    tmp["word"] = tmp[cleaned_col].astype(str).str.split(" ")
    tmp = tmp.explode("word", ignore_index=False)
    w = tmp["word"].astype(str)
    tmp["word_len"] = w.str.len().astype(np.int64)
    tmp = tmp[tmp["word_len"] > 0].copy()
    return tmp[["essay_id", "word", *word_features]]


def Word_aggregation_from_df(df_in: pd.DataFrame, cleaned_col: str) -> pd.DataFrame:
    tmp = _build_word_df(df_in, cleaned_col=cleaned_col)
    keys = tmp["essay_id"].astype("string")
    codes, uniques = pd.factorize(keys, sort=True)
    n_keys = len(uniques)

    out = pd.DataFrame(index=uniques.astype("string"))

    wlen = tmp["word_len"].to_numpy()

    for i in [3, 4, 5, 6, 7, 8, 10]:
        out[f"word_{i}_cnt"] = _group_sum_bool(codes, wlen >= i, n_keys)
    for i in [1, 2, 3]:
        out[f"word_{i}_cnt_v2"] = _group_sum_bool(codes, wlen <= i, n_keys)

    out["short_word_cnt"] = _group_sum_bool(codes, (wlen <= 4) & (wlen > 2), n_keys)
    out["mid_word_cnt"] = _group_sum_bool(codes, (wlen <= 6) & (wlen > 4), n_keys)
    out["long_word_cnt"] = _group_sum_bool(codes, (wlen <= 10) & (wlen > 6), n_keys)

    out["word_cnt"] = np.bincount(codes, minlength=n_keys).astype(np.int64)

    g = tmp.groupby(keys, sort=True)
    agg_basic = g[word_features].agg(["max", "mean", "min", "std", "sum"])
    agg_basic.columns = [f"{c[0]}_{c[1]}" for c in agg_basic.columns]
    out = out.join(agg_basic, how="left")

    out = _add_quantiles(out, g, word_features)

    out = (
        out.reset_index()
        .rename(columns={"index": "essay_id"})
        .sort_values("essay_id")
        .reset_index(drop=True)
    )

    denom_word = out["word_cnt"].replace(0, np.nan)
    for i in [3, 4, 5, 6, 7, 8, 10]:
        out[f"word_{i}_cnt_ratio"] = out[f"word_{i}_cnt"] / denom_word
    for i in [1, 2, 3]:
        out[f"word_{i}_cnt_v2_ratio"] = out[f"word_{i}_cnt_v2"] / denom_word

    denom2 = out["word_2_cnt_v2"].replace(0, np.nan)
    denom3 = out["word_3_cnt_v2"].replace(0, np.nan)
    for i in [3, 4, 5, 6, 7, 8, 10]:
        out[f"word_{i}_pre2_ratio"] = out[f"word_{i}_cnt"] / denom2
        out[f"word_{i}_pre3_ratio"] = out[f"word_{i}_cnt"] / denom3

    for i in [1, 2, 3]:
        denom = out[f"word_{i}_cnt_v2"].replace(0, np.nan)
        out[f"short_word_ratio_{i}"] = out["short_word_cnt"] / denom
        out[f"mid_word_ratio_{i}"] = out["mid_word_cnt"] / denom
        out[f"long_word_ratio_{i}"] = out["long_word_cnt"] / denom

    return out




## === cell 9
df_train_clean = df_train[["essay_id", "full_text"]].copy()
df_train_clean["full_text_clean"] = df_train_clean["full_text"].map(dataPreprocessing)

df_test_clean = df_test[["essay_id", "full_text"]].copy()
df_test_clean["full_text_clean"] = df_test_clean["full_text"].map(dataPreprocessing)

train_texts = df_train_clean["full_text_clean"].tolist()
test_texts = df_test_clean["full_text_clean"].tolist()

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

train_tfid = vectorizer.fit_transform(train_texts)
train_cnt = vectorizer_cnt.fit_transform(train_texts)
test_tfid = vectorizer.transform(test_texts)
test_cnt = vectorizer_cnt.transform(test_texts)

clean_memory()



## === cell 10
if CFG.LOAD_FEATURES_FROM is not None:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)
else:
    t0 = time.time()
    base_train_df = df_train_clean[["essay_id", "full_text", "full_text_clean"]].copy()
    train_feats1 = Paragraph_aggregation_from_df(base_train_df)
    train_feats2 = Sentence_aggregation_from_df(
        base_train_df, cleaned_col="full_text_clean"
    )
    train_feats3 = Word_aggregation_from_df(
        base_train_df, cleaned_col="full_text_clean"
    )
    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left").merge(
        train_feats3, on="essay_id", how="left"
    )
    train_feats["score"] = df_train["score"].values
    print(f"Train handcrafted features time: {time.time()-t0:.1f}s")
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)

display(train_feats.head())



## === cell 11
a = 2.948
b = 1.092


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




## === cell 12
categorical_columns = train_feats.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
FEATURES = [
    c
    for c in train_feats.columns
    if c not in (categorical_columns + ["score", "essay_id"])
]
TARGET = "score"

print("Num handcrafted FEATURES:", len(FEATURES))

X_hand = np.clip(train_feats[FEATURES].fillna(0).to_numpy(dtype=np.float32), 0, 10000)
_train_y_all = train_feats[TARGET].to_numpy(dtype=np.float32) - a

X_train_all = sparse.hstack(
    [sparse.csr_matrix(X_hand), train_tfid, train_cnt], format="csr"
)
del X_hand
clean_memory()
print("X_train_all shape:", X_train_all.shape)




## === cell 13
def lightgbm_train():
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
    y_strat = train_feats[TARGET]

    for i, (train_index, valid_index) in enumerate(
        skf.split(np.zeros(X_train_all.shape[0]), y_strat)
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

        train_x = X_train_all[train_index]
        train_y = _train_y_all[train_index]
        valid_x = X_train_all[valid_index]
        valid_y = _train_y_all[valid_index]

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
    print("CV Score for LightGBM = ", cv)

    if DO_PLOTS:
        cm = confusion_matrix(
            all_true, np.clip(all_oof, 1, 6).round(), labels=[x for x in range(1, 7)]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
        )
        disp.plot()
        import matplotlib.pyplot as plt

        plt.show()




## === cell 14
if CFG.LOAD_MODELS_FROM is None:
    print("Training LightGBM")
    lightgbm_train()
else:
    print("Using preloaded models from:", CFG.LOAD_MODELS_FROM)



## === cell 15
model_path0 = None
if CFG.LOAD_MODELS_FROM:
    cand = f"{CFG.LOAD_MODELS_FROM}LGB_v{CFG.VER}_f0.pkl"
    if os.path.exists(cand):
        model_path0 = cand
else:
    cand = f"LGB_v{CFG.VER}_f0.pkl"
    if os.path.exists(cand):
        model_path0 = cand

if model_path0 is not None and DO_PLOTS:
    model = pickle.load(open(model_path0, "rb"))
    df_importance = pd.DataFrame(
        {
            "features_name": FEATURES,
            "importance": model.feature_importances_[: len(FEATURES)],
        }
    ).sort_values(by="importance", ascending=False)
    import matplotlib.pyplot as plt

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
else:
    print("Skip feature importance plot.")



## === cell 16
t0 = time.time()
if CFG.LOAD_TEST_FEATURES_FROM is not None:
    print("Load test_feats.csv")
    test_feats = pd.read_csv(CFG.LOAD_TEST_FEATURES_FROM)
else:
    base_test_df = df_test_clean[["essay_id", "full_text", "full_text_clean"]].copy()
    test_feats1 = Paragraph_aggregation_from_df(base_test_df)
    test_feats2 = Sentence_aggregation_from_df(
        base_test_df, cleaned_col="full_text_clean"
    )
    test_feats3 = Word_aggregation_from_df(base_test_df, cleaned_col="full_text_clean")
    test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left").merge(
        test_feats3, on="essay_id", how="left"
    )
print(f"Test handcrafted features ready in: {time.time()-t0:.1f}s")

test_feats["essay_id"] = test_feats["essay_id"].astype("string")
test_feats = test_feats.sort_values("essay_id").reset_index(drop=True)
df_test_sorted = df_test.copy()
df_test_sorted["essay_id"] = df_test_sorted["essay_id"].astype("string")
df_test_sorted = df_test_sorted.sort_values("essay_id").reset_index(drop=True)

print("Shape of test handcrafted feats:", test_feats.shape)
display(test_feats.head())

X_hand_test = np.clip(
    test_feats[FEATURES].fillna(0).to_numpy(dtype=np.float32), 0, 10000
)
X_test = sparse.hstack(
    [
        sparse.csr_matrix(X_hand_test),
        test_tfid[df_test_sorted.index],
        test_cnt[df_test_sorted.index],
    ],
    format="csr",
)
del X_hand_test
clean_memory()
print("X_test shape:", X_test.shape)



## === cell 17
n_folds = 15
pred_sum = np.zeros(X_test.shape[0], dtype=np.float64)

for i in range(n_folds):
    print(f"Fold {i+1}")
    if CFG.LOAD_MODELS_FROM:
        model_path = f"{CFG.LOAD_MODELS_FROM}LGB_v{CFG.VER}_f{i}.pkl"
    else:
        model_path = f"LGB_v{CFG.VER}_f{i}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}. If you intended to train, ensure training completed."
        )

    model = pickle.load(open(model_path, "rb"))
    pred_sum += model.predict(X_test) + a
    del model
    clean_memory()

pred1 = pred_sum / n_folds

sub = pd.DataFrame({"essay_id": df_test_sorted["essay_id"].values})
sub["score"] = np.clip(pred1, 1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
display(sub.head())
print("Wrote: submission.csv")

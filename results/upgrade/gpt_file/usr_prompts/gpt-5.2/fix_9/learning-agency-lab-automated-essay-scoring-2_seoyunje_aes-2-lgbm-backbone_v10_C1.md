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
import polars as pl  # kept

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

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass




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


_html_re = re.compile(r"<.*?>")
_re_at = re.compile(r"@\w+")
_re_apnum = re.compile(r"'\d+")
_re_num = re.compile(r"\d+")
_re_http = re.compile(r"http\w+")
_re_space = re.compile(r"\s+")
_re_dots = re.compile(r"\.+")
_re_commas = re.compile(r"\,+")
_re_keep = re.compile(r'[^\w\s.,;:""\'\'?!]')


def removeHTML(x):
    return _html_re.sub(r"", x)  # html -> ''


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
from concurrent.futures import ThreadPoolExecutor

paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
]


def _safe_div(n, d):
    return (n / d) if d else 0.0


_token_clean_re = re.compile(r"[^a-z]")


def _tokenize_words_lower(text: str):
    return text.split()


def _build_token_misspell_cache(texts, max_vocab=400_000):
    vocab = set()
    for t in texts:
        for w in _tokenize_words_lower(t):
            if w:
                vocab.add(w)
                if len(vocab) >= max_vocab:
                    break
        if len(vocab) >= max_vocab:
            break

    try:
        from spellchecker import SpellChecker  # type: ignore

        sp = SpellChecker()
        miss = sp.unknown(vocab)
        cache = {w: (1 if w in miss else 0) for w in vocab}
    except Exception:
        cache = {}
        for w in vocab:
            ww = _token_clean_re.sub("", w)
            if ww == "":
                cache[w] = 0
            elif len(ww) > 18:
                cache[w] = 1
            elif not ww.isalpha():
                cache[w] = 1
            else:
                cache[w] = 0
    return cache


def _misspelled_cnt_from_cache(text: str, cache: dict) -> int:
    if not text:
        return 0
    get = cache.get
    fallback = count_misspelled_words

    seen = set()
    cnt = 0
    for w in text.split():
        if not w or w in seen:
            continue
        seen.add(w)
        v = get(w)
        if v is None:
            vv = fallback(w)
            v = 1 if vv > 0 else 0
            cache[w] = v
        cnt += v
    return cnt


def _agg_stats(values):
    if not values:
        return dict(max=0.0, mean=0.0, min=0.0, std=0.0, sum=0.0, q1=0.0, q3=0.0)
    arr = np.asarray(values, dtype=np.float64)
    return dict(
        max=float(arr.max()),
        mean=float(arr.mean()),
        min=float(arr.min()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        sum=float(arr.sum()),
        q1=float(np.quantile(arr, 0.25)),
        q3=float(np.quantile(arr, 0.75)),
    )


_GLOBAL_TOKEN_MISS_CACHE = None


def _init_worker_token_cache(token_cache):
    global _GLOBAL_TOKEN_MISS_CACHE
    _GLOBAL_TOKEN_MISS_CACHE = token_cache


def _compute_one_essay_features(args):
    essay_id, full_text_pre = args
    token_miss_cache = _GLOBAL_TOKEN_MISS_CACHE

    miss_cnt = _misspelled_cnt_from_cache
    safe_div = _safe_div

    paragraphs = full_text_pre.split("\n\n")
    p_len = []
    p_comma = []
    p_miss = []
    p_sent = []
    p_word = []
    p_cnts_len_ge = {
        i: 0 for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
    }
    p_cnts_len_le = {i: 0 for i in [100, 200]}
    p_sent_ge = {i: 0 for i in [2, 4, 6, 8, 10]}
    p_word_ge = {i: 0 for i in [20, 40, 60, 90, 120]}
    p_comma_ge = {i: 0 for i in [1, 2, 3, 4, 5]}
    p_miss_ge = {i: 0 for i in [4, 8, 12, 16]}
    p_miss_le = {i: 0 for i in [2, 4]}

    short_p = mid_p = long_p = 0
    short_ps = mid_ps = long_ps = 0
    short_pw = mid_pw = long_pw = 0
    short_pm = mid_pm = long_pm = 0

    for pp in paragraphs:
        l = len(pp)
        c = pp.count(",")
        m = miss_cnt(pp, token_miss_cache)
        s_cnt = len(pp.split("."))
        w_cnt = len(pp.split())

        p_len.append(l)
        p_comma.append(c)
        p_miss.append(m)
        p_sent.append(s_cnt)
        p_word.append(w_cnt)

        for i in p_cnts_len_ge:
            if l >= i:
                p_cnts_len_ge[i] += 1
        for i in p_cnts_len_le:
            if l <= i:
                p_cnts_len_le[i] += 1
        if (l <= 300) and (l > 100):
            short_p += 1
        if (l <= 500) and (l > 300):
            mid_p += 1
        if (l <= 700) and (l > 500):
            long_p += 1

        for i in p_sent_ge:
            if s_cnt >= i:
                p_sent_ge[i] += 1
        if (s_cnt <= 4) and (s_cnt > 2):
            short_ps += 1
        if (s_cnt <= 8) and (s_cnt > 4):
            mid_ps += 1
        if (s_cnt <= 10) and (s_cnt > 8):
            long_ps += 1

        for i in p_word_ge:
            if w_cnt >= i:
                p_word_ge[i] += 1
        if (w_cnt <= 40) and (w_cnt > 20):
            short_pw += 1
        if (w_cnt <= 90) and (w_cnt > 40):
            mid_pw += 1
        if (w_cnt <= 120) and (w_cnt > 90):
            long_pw += 1

        for i in p_comma_ge:
            if c >= i:
                p_comma_ge[i] += 1

        for i in p_miss_ge:
            if m >= i:
                p_miss_ge[i] += 1
        for i in p_miss_le:
            if m <= i:
                p_miss_le[i] += 1
        if (m <= 8) and (m > 4):
            short_pm += 1
        if (m <= 12) and (m > 8):
            mid_pm += 1
        if (m <= 16) and (m > 12):
            long_pm += 1

    out = {"essay_id": essay_id}
    for i, v in p_cnts_len_ge.items():
        out[f"paragraph_{i}_cnt"] = v
    for i, v in p_cnts_len_le.items():
        out[f"paragraph_{i}_cnt_v2"] = v
    out["short_paragraph_cnt"] = short_p
    out["mid_paragraph_cnt"] = mid_p
    out["long_paragraph_cnt"] = long_p
    for i, v in p_sent_ge.items():
        out[f"paragraph_sentence_{i}_cnt"] = v
    out["short_paragraph_sentence_cnt"] = short_ps
    out["mid_paragraph_sentence_cnt"] = mid_ps
    out["long_paragraph_sentence_cnt"] = long_ps
    for i, v in p_word_ge.items():
        out[f"paragraph_word_{i}_cnt"] = v
    out["short_paragraph_word_cnt"] = short_pw
    out["mid_paragraph_word_cnt"] = mid_pw
    out["long_paragraph_word_cnt"] = long_pw
    for i, v in p_comma_ge.items():
        out[f"paragraph_comma_{i}_cnt"] = v
    for i, v in p_miss_ge.items():
        out[f"paragraph_misspelled_{i}_cnt"] = v
    for i, v in p_miss_le.items():
        out[f"paragraph_misspelled_{i}_cnt_v2"] = v
    out["short_paragraph_misspelled_cnt"] = short_pm
    out["mid_paragraph_misspelled_cnt"] = mid_pm
    out["long_paragraph_misspelled_cnt"] = long_pm

    out["paragraph_cnt"] = len(paragraphs)

    stats = _agg_stats(p_len)
    for k, v in stats.items():
        out[f"paragraph_len_{k}"] = v
    stats = _agg_stats(p_sent)
    for k, v in stats.items():
        out[f"paragraph_sentence_cnt_{k}"] = v
    stats = _agg_stats(p_word)
    for k, v in stats.items():
        out[f"paragraph_word_cnt_{k}"] = v
    stats = _agg_stats(p_comma)
    for k, v in stats.items():
        out[f"paragraph_comma_cnt_{k}"] = v
    stats = _agg_stats(p_miss)
    for k, v in stats.items():
        out[f"paragraph_misspelled_cnt_{k}"] = v

    sentences = full_text_pre.split(".")
    s_len = []
    s_word = []
    s_miss = []
    only_s_len = []

    s_len_ge = {i: 0 for i in [40, 60, 70, 80, 100, 120, 140]}
    s_len_le = {i: 0 for i in [10, 20, 30]}
    short_s = mid_s = long_s = 0

    only_len_ge = {i: 0 for i in [40, 60, 80, 100, 120]}
    short_only = mid_only = long_only = 0

    s_word_ge = {i: 0 for i in [10, 15, 20, 25]}
    short_sw = mid_sw = long_sw = 0

    sent_cnt = 0
    for s in sentences:
        l = len(s)
        if l <= 3:
            continue
        sent_cnt += 1
        m = miss_cnt(s, token_miss_cache)
        ol = len(s.replace(" ", ""))
        wc = len(s.split())

        s_len.append(l)
        s_word.append(wc)
        s_miss.append(m)
        only_s_len.append(ol)

        for i in s_len_ge:
            if l >= i:
                s_len_ge[i] += 1
        for i in s_len_le:
            if l <= i:
                s_len_le[i] += 1
        if (l <= 70) and (l > 40):
            short_s += 1
        if (l <= 100) and (l > 70):
            mid_s += 1
        if (l <= 140) and (l > 100):
            long_s += 1

        for i in only_len_ge:
            if ol >= i:
                only_len_ge[i] += 1
        if (ol <= 60) and (ol > 40):
            short_only += 1
        if (ol <= 100) and (ol > 60):
            mid_only += 1
        if (ol <= 120) and (ol > 100):
            long_only += 1

        for i in s_word_ge:
            if wc >= i:
                s_word_ge[i] += 1
        if (wc <= 15) and (wc > 10):
            short_sw += 1
        if (wc <= 20) and (wc > 15):
            mid_sw += 1
        if (wc <= 25) and (wc > 20):
            long_sw += 1

    for i, v in s_len_ge.items():
        out[f"sentence_{i}_cnt"] = v
    for i, v in s_len_le.items():
        out[f"sentence_{i}_cnt_v2"] = v
    out["short_sentence_cnt"] = short_s
    out["mid_sentence_cnt"] = mid_s
    out["long_sentence_cnt"] = long_s
    for i, v in only_len_ge.items():
        out[f"only_sentence_{i}_cnt"] = v
    out["short_only_sentence_cnt"] = short_only
    out["mid_only_sentence_cnt"] = mid_only
    out["long_only_sentence_cnt"] = long_only
    for i, v in s_word_ge.items():
        out[f"sentence_word_{i}_cnt"] = v
    out["short_sentence_word_cnt"] = short_sw
    out["mid_sentence_word_cnt"] = mid_sw
    out["long_sentence_word_cnt"] = long_sw
    out["sentence_cnt"] = sent_cnt

    stats = _agg_stats(s_len)
    for k, v in stats.items():
        out[f"sentence_len_{k}"] = v
    stats = _agg_stats(s_word)
    for k, v in stats.items():
        out[f"sentence_word_cnt_{k}"] = v
    stats = _agg_stats(s_miss)
    for k, v in stats.items():
        out[f"sentence_misspelled_cnt_{k}"] = v

    for i in [40, 60, 70, 80, 100, 120, 140]:
        out[f"sentence_{i}_cnt_ratio"] = safe_div(out[f"sentence_{i}_cnt"], sent_cnt)
    out["short_sentence_cnt_ratio"] = safe_div(out["short_sentence_cnt"], sent_cnt)
    out["mid_sentence_cnt_ratio"] = safe_div(out["mid_sentence_cnt"], sent_cnt)
    out["long_sentence_cnt_ratio"] = safe_div(out["long_sentence_cnt"], sent_cnt)

    words = full_text_pre.split()
    w_len = []
    w_cnt = 0
    w_ge = {i: 0 for i in [3, 4, 5, 6, 7, 8, 10]}
    w_le = {i: 0 for i in [1, 2, 3]}
    short_w = mid_w = long_w = 0

    for w in words:
        l = len(w)
        if l <= 0:
            continue
        w_cnt += 1
        w_len.append(l)
        for i in w_ge:
            if l >= i:
                w_ge[i] += 1
        for i in w_le:
            if l <= i:
                w_le[i] += 1
        if (l <= 4) and (l > 2):
            short_w += 1
        if (l <= 6) and (l > 4):
            mid_w += 1
        if (l <= 10) and (l > 6):
            long_w += 1

    for i, v in w_ge.items():
        out[f"word_{i}_cnt"] = v
    for i, v in w_le.items():
        out[f"word_{i}_cnt_v2"] = v
    out["short_word_cnt"] = short_w
    out["mid_word_cnt"] = mid_w
    out["long_word_cnt"] = long_w
    out["word_cnt"] = w_cnt

    stats = _agg_stats(w_len)
    for k, v in stats.items():
        out[f"word_len_{k}"] = v

    for i in [3, 4, 5, 6, 7, 8, 10]:
        out[f"word_{i}_cnt_ratio"] = safe_div(out[f"word_{i}_cnt"], w_cnt)
    for i in [1, 2, 3]:
        out[f"word_{i}_cnt_v2_ratio"] = safe_div(out[f"word_{i}_cnt_v2"], w_cnt)
    denom2 = out.get("word_2_cnt_v2", 0)
    denom3 = out.get("word_3_cnt_v2", 0)
    for i in [3, 4, 5, 6, 7, 8, 10]:
        out[f"word_{i}_pre2_ratio"] = safe_div(out[f"word_{i}_cnt"], denom2)
        out[f"word_{i}_pre3_ratio"] = safe_div(out[f"word_{i}_cnt"], denom3)
    for i in [1, 2, 3]:
        d = out.get(f"word_{i}_cnt_v2", 0)
        out[f"short_word_ratio_{i}"] = safe_div(out["short_word_cnt"], d)
        out[f"mid_word_ratio_{i}"] = safe_div(out["mid_word_cnt"], d)
        out[f"long_word_ratio_{i}"] = safe_div(out["long_word_cnt"], d)

    return out


def _compute_features_fast(pl_df: pl.DataFrame) -> pd.DataFrame:
    essay_ids = pl_df["essay_id"].to_list()
    texts = pl_df["full_text_pre"].to_list()

    token_cache = _build_token_misspell_cache(texts)

    n_workers = min(N_THREADS, os.cpu_count() or 1)
    chunks = list(zip(essay_ids, texts))

    _init_worker_token_cache(token_cache)

    if n_workers <= 1:
        rows = [
            _compute_one_essay_features(x)
            for x in tqdm(chunks, desc="Feature engineering")
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rows = list(
                tqdm(
                    ex.map(_compute_one_essay_features, chunks, chunksize=1024),
                    total=len(chunks),
                    desc="Feature engineering",
                )
            )

    df = pd.DataFrame(rows).sort_values("essay_id").reset_index(drop=True)
    return df


sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]
word_features = ["word_len"]


def Paragraph_Features(x, paragraph_col: str = "paragraph"):
    return x


def Paragraph_aggregation(x):
    raise RuntimeError("Paragraph_aggregation should not be called in optimized path")


def Sentence_Features(x, text_col: str = "full_text"):
    return x


def Sentence_aggregation(x):
    raise RuntimeError("Sentence_aggregation should not be called in optimized path")


def Word_Features(x, text_col: str = "full_text"):
    return x


def Word_aggregation(x):
    raise RuntimeError("Word_aggregation should not be called in optimized path")




## === cell 12
print("Precomputing full_text_pre for train/test (single pass preprocessing)")

_pre_cache_train = "/kaggle/working/full_text_pre_train.pkl"
_pre_cache_test = "/kaggle/working/full_text_pre_test.pkl"

if os.path.exists(_pre_cache_train) and os.path.exists(_pre_cache_test):
    df_train["full_text_pre"] = pickle.load(open(_pre_cache_train, "rb"))
    df_test["full_text_pre"] = pickle.load(open(_pre_cache_test, "rb"))
else:
    df_train["full_text_pre"] = [
        dataPreprocessing(x)
        for x in tqdm(df_train["full_text"].tolist(), desc="Preprocess train")
    ]
    df_test["full_text_pre"] = [
        dataPreprocessing(x)
        for x in tqdm(df_test["full_text"].tolist(), desc="Preprocess test")
    ]
    pickle.dump(df_train["full_text_pre"].tolist(), open(_pre_cache_train, "wb"))
    pickle.dump(df_test["full_text_pre"].tolist(), open(_pre_cache_test, "wb"))

train = pl.from_pandas(df_train[["essay_id", "full_text", "score", "full_text_pre"]])
test = pl.from_pandas(df_test[["essay_id", "full_text", "full_text_pre"]])

train = train.with_columns(
    pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph_pre")
)
test = test.with_columns(
    pl.col("full_text_pre").str.split(by="\n\n").alias("paragraph_pre")
)

clean_memory()



## === cell 13
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

_train_texts = df_train["full_text_pre"].tolist()
train_tfid = vectorizer.fit_transform(_train_texts)



## === cell 14
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

train_cnt = vectorizer_cnt.fit_transform(_train_texts)



## === cell 15
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



## === cell 16
_train_feat_cache = f"/kaggle/working/train_feats_cached_v{CFG.VER}.csv"

if CFG.LOAD_FEATURES_FROM is None and os.path.exists(_train_feat_cache):
    train_feats = pd.read_csv(_train_feat_cache)
elif CFG.LOAD_FEATURES_FROM is None:
    train_feats = _compute_features_fast(train.select(["essay_id", "full_text_pre"]))
    train_feats["score"] = df_train["score"].values
    train_feats.to_csv(_train_feat_cache, index=False)
else:
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)

display(train_feats.head())
print("train_feats shape (engineered only):", train_feats.shape)



## === cell 17
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping

print("LightGBM Version: ", lgb.__version__)




## === cell 18
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



## === cell 19
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
    [sparse.csr_matrix(X_eng), train_tfid, train_cnt], format="csr"
).tocsr()
y = train_feats[TARGET].to_numpy(dtype=np.float32, copy=False) - a

clean_memory()




## === cell 20
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
            force_col_wise=True,
            feature_pre_filter=False,
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




## === cell 21
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



## === cell 22
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



## === cell 23
_test_texts = df_test["full_text_pre"].tolist()
test_tfid = vectorizer.transform(_test_texts)



## === cell 24
test_cnt = vectorizer_cnt.transform(_test_texts)



## === cell 25
_test_feat_cache = f"/kaggle/working/test_feats_cached_v{CFG.VER}.csv"

if os.path.exists(_test_feat_cache):
    test_feats = pd.read_csv(_test_feat_cache)
else:
    test_feats = _compute_features_fast(test.select(["essay_id", "full_text_pre"]))
    test_feats.to_csv(_test_feat_cache, index=False)

print("Shape of test_feats (engineered only):", test_feats.shape)
display(test_feats.head())



## === cell 26
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
    [sparse.csr_matrix(X_test_eng), test_tfid, test_cnt], format="csr"
).tocsr()

print(
    "Missing engineered added to test:",
    len(missing_eng),
    " Extra engineered dropped from test:",
    len(extra_eng),
)
clean_memory()



## === cell 27
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

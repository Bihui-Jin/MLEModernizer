# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

RANDOM_STATE = 42
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(os.cpu_count() or 4))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # safe no-op if already patched
except Exception:
    pass

import shutil
import gc
import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"

stopwords = "english"
cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

np.random.seed(RANDOM_STATE)



## === cell 1
_ = None



## === cell 2
_ = None




## === cell 3
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    out_csv = os.path.join(OUTPUT_DIR, filename.replace(".zip", ""))
    if os.path.exists(out_csv):
        return
    try:
        shutil.unpack_archive(
            filename=os.path.join(DATA_DIR, filename),
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")




## === cell 4
for csv_name, zip_name in [
    ("train.csv", "train.csv.zip"),
    ("test.csv", "test.csv.zip"),
    ("sample_submission.csv", "sample_submission.csv.zip"),
]:
    if not os.path.exists(os.path.join(DATA_DIR, csv_name)):
        unpack_zipfile(filename=zip_name)




## === cell 5
def read_csv_fast(name, usecols=None, dtype=None):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        path2 = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path2):
            path = path2
        else:
            raise FileNotFoundError(
                f"Could not find {name} in {DATA_DIR} or {OUTPUT_DIR}"
            )
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)


train_df = read_csv_fast(
    "train.csv",
    usecols=["id", "comment_text"] + cols,
    dtype={"id": "string", "comment_text": "string", **{c: "int8" for c in cols}},
)
test_df = read_csv_fast(
    "test.csv",
    usecols=["id", "comment_text"],
    dtype={"id": "string", "comment_text": "string"},
)
sample_sub = read_csv_fast("sample_submission.csv", dtype={"id": "string"})



## === cell 6
test_labels = None
test_labels_info = None



## === cell 7
(train_df.shape, test_df.shape, sample_sub.shape)



## === cell 8
label_counts = train_df[cols].sum(axis=0).astype(int)
label_counts



## === cell 9
pass



## === cell 10
(test_df.shape,)



## === cell 11
corpus_train = train_df["comment_text"].fillna("").to_numpy(dtype=object)
corpus_train[:2]



## === cell 12
target_train = train_df[cols].to_numpy()
target_train[:2]



## === cell 13
corpus_test = test_df["comment_text"].fillna("").to_numpy(dtype=object)
corpus_test[:2]



## === cell 14
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    analyzer="word",
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    dtype=np.float32,
)

vectorizer_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    dtype=np.float32,
)



## === cell 15
from scipy.sparse import hstack, save_npz, load_npz


def _vectorizer_fingerprint(vec: TfidfVectorizer) -> str:
    params = vec.get_params(deep=True)
    blob = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(blob).hexdigest()


def _file_stat_fp(path: str) -> str:
    st = os.stat(path)
    blob = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}".encode()
    return hashlib.md5(blob).hexdigest()


def _resolve_path(name: str) -> str:
    p1 = os.path.join(DATA_DIR, name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find {name} in {DATA_DIR} or {OUTPUT_DIR}")


cache_dir = os.path.join(OUTPUT_DIR, "tfidf_cache")
os.makedirs(cache_dir, exist_ok=True)

word_fp = _vectorizer_fingerprint(vectorizer)
char_fp = _vectorizer_fingerprint(vectorizer_char)

train_path = _resolve_path("train.csv")
test_path = _resolve_path("test.csv")
data_fp = f"tr{_file_stat_fp(train_path)}__te{_file_stat_fp(test_path)}"

path_w_tr = os.path.join(cache_dir, f"w{word_fp}_{data_fp}_Xtr.npz")
path_w_te = os.path.join(cache_dir, f"w{word_fp}_{data_fp}_Xte.npz")
path_c_tr = os.path.join(cache_dir, f"c{char_fp}_{data_fp}_Xtr.npz")
path_c_te = os.path.join(cache_dir, f"c{char_fp}_{data_fp}_Xte.npz")
path_vec_w = os.path.join(cache_dir, f"w{word_fp}_{data_fp}_vec.pkl")
path_vec_c = os.path.join(cache_dir, f"c{char_fp}_{data_fp}_vec.pkl")

need_word = not (
    os.path.exists(path_w_tr)
    and os.path.exists(path_w_te)
    and os.path.exists(path_vec_w)
)
need_char = not (
    os.path.exists(path_c_tr)
    and os.path.exists(path_c_te)
    and os.path.exists(path_vec_c)
)

if not need_word:
    Xw_tr = load_npz(path_w_tr)
    Xw_te = load_npz(path_w_te)
    with open(path_vec_w, "rb") as f:
        vectorizer = pickle.load(f)
else:
    Xw_tr = vectorizer.fit_transform(corpus_train)
    Xw_te = vectorizer.transform(corpus_test)
    save_npz(path_w_tr, Xw_tr)
    save_npz(path_w_te, Xw_te)
    with open(path_vec_w, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

if not need_char:
    Xc_tr = load_npz(path_c_tr)
    Xc_te = load_npz(path_c_te)
    with open(path_vec_c, "rb") as f:
        vectorizer_char = pickle.load(f)
else:
    Xc_tr = vectorizer_char.fit_transform(corpus_train)
    Xc_te = vectorizer_char.transform(corpus_test)
    save_npz(path_c_tr, Xc_tr)
    save_npz(path_c_te, Xc_te)
    with open(path_vec_c, "wb") as f:
        pickle.dump(vectorizer_char, f, protocol=pickle.HIGHEST_PROTOCOL)

features_train = hstack([Xw_tr, Xc_tr], format="csr")
features_test = hstack([Xw_te, Xc_te], format="csr")

del Xw_tr, Xw_te, Xc_tr, Xc_te
del corpus_train, corpus_test
gc.collect()

(features_train.shape, features_test.shape)



## === cell 16
gc.collect()



## === cell 17
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="saga",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=-1,
)



## === cell 18
classifier.fit(features_train, target_train)



## === cell 19
test_ids = test_df["id"].to_numpy(dtype=object)
test_ids[:2]



## === cell 20
pass



## === cell 21
pass



## === cell 22
proba_predictions_test = classifier.predict_proba(features_test)



## === cell 23
sample_sub.head()



## === cell 24
submission = pd.DataFrame(proba_predictions_test, columns=cols)
submission.insert(0, "id", test_ids)
submission = submission.reindex(columns=sample_sub.columns)
submission.head()



## === cell 25
sample_sub.info()



## === cell 26
submission.info()



## === cell 27
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")



## === cell 28
with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

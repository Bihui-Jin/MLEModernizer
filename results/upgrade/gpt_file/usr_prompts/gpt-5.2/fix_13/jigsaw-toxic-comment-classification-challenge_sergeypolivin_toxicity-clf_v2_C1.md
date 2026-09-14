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
corpus_train = train_df["comment_text"].fillna("").to_numpy()
corpus_train[:2]



## === cell 12
target_train = train_df[cols].to_numpy()
target_train[:2]



## === cell 13
corpus_test = test_df["comment_text"].fillna("").to_numpy()
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
from scipy.sparse import hstack, vstack, save_npz, load_npz


def _vectorizer_fingerprint(vec: TfidfVectorizer) -> str:
    params = vec.get_params(deep=True)
    blob = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(blob).hexdigest()


def _array_fingerprint(arr: np.ndarray, n_bytes: int = 2_000_000) -> str:
    """
    Speed/correctness: robust cache key that changes if data changes.
    Uses a deterministic sample of bytes from start+end to avoid hashing the full corpus.
    """
    if arr.size == 0:
        return "empty"
    b0 = str(arr[0]).encode("utf-8", "ignore")
    b1 = str(arr[-1]).encode("utf-8", "ignore")
    mid = str(arr[arr.size // 2]).encode("utf-8", "ignore")
    blob = b0 + b"\n" + mid + b"\n" + b1
    return hashlib.md5(blob[:n_bytes]).hexdigest()


cache_dir = os.path.join(OUTPUT_DIR, "tfidf_cache")
os.makedirs(cache_dir, exist_ok=True)

word_fp = _vectorizer_fingerprint(vectorizer)
char_fp = _vectorizer_fingerprint(vectorizer_char)
data_fp = f"tr{len(corpus_train)}_{_array_fingerprint(corpus_train)}__te{len(corpus_test)}_{_array_fingerprint(corpus_test)}"

cache_prefix = f"w{word_fp}_c{char_fp}_{data_fp}"
path_train = os.path.join(cache_dir, f"{cache_prefix}_Xtr.npz")
path_test = os.path.join(cache_dir, f"{cache_prefix}_Xte.npz")
path_vec = os.path.join(cache_dir, f"{cache_prefix}_vec.pkl")
path_vec_char = os.path.join(cache_dir, f"{cache_prefix}_vec_char.pkl")

if (
    os.path.exists(path_train)
    and os.path.exists(path_test)
    and os.path.exists(path_vec)
    and os.path.exists(path_vec_char)
):
    features_train = load_npz(path_train)
    features_test = load_npz(path_test)
    with open(path_vec, "rb") as f:
        vectorizer = pickle.load(f)
    with open(path_vec_char, "rb") as f:
        vectorizer_char = pickle.load(f)
else:
    vectorizer.fit(corpus_train)
    vectorizer_char.fit(corpus_train)

    combined = np.concatenate([corpus_train, corpus_test], axis=0)

    X_word_all = vectorizer.transform(combined)
    X_char_all = vectorizer_char.transform(combined)

    X_all = hstack([X_word_all, X_char_all], format="csr")

    n_tr = len(corpus_train)
    features_train = X_all[:n_tr]
    features_test = X_all[n_tr:]

    del combined, X_word_all, X_char_all, X_all
    gc.collect()

    features_train.sort_indices()
    features_test.sort_indices()

    save_npz(path_train, features_train)
    save_npz(path_test, features_test)
    with open(path_vec, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_vec_char, "wb") as f:
        pickle.dump(vectorizer_char, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    warm_start=True,
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=1,
)



## === cell 18
classifier.fit(features_train, target_train)



## === cell 19
test_ids = test_df["id"].astype(str).to_numpy()
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

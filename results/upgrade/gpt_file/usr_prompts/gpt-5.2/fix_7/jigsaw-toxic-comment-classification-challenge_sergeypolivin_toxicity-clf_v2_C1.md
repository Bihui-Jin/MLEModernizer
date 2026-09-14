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
import shutil

import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # safe no-op if already patched
except Exception:
    pass

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

try:
    nltk.download("stopwords", quiet=True)
    stopwords = list(nltk_stopwords.words("english"))
except Exception:
    stopwords = None  # let TfidfVectorizer handle default if needed

cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 1
os.listdir(DATA_DIR)



## === cell 2
os.listdir(OUTPUT_DIR)




## === cell 3
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    out_csv = os.path.join(OUTPUT_DIR, filename.replace(".zip", ""))
    if os.path.exists(out_csv):
        return
    try:
        shutil.unpack_archive(
            filename=DATA_DIR + filename,
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
os.listdir(OUTPUT_DIR)




## === cell 6
def _find_file_recursive(root_dir, filename):
    for r, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(r, filename)
    return None


def read_csv_fallback(name, usecols=None, dtype=None):
    in_path = os.path.join(DATA_DIR, name)
    out_path = os.path.join(OUTPUT_DIR, name)

    if os.path.exists(in_path):
        return pd.read_csv(in_path, usecols=usecols, dtype=dtype)
    if os.path.exists(out_path):
        return pd.read_csv(out_path, usecols=usecols, dtype=dtype)

    found = _find_file_recursive(DATA_DIR, name)
    if found is not None:
        return pd.read_csv(found, usecols=usecols, dtype=dtype)

    found = _find_file_recursive(OUTPUT_DIR, name)
    if found is not None:
        return pd.read_csv(found, usecols=usecols, dtype=dtype)

    raise FileNotFoundError(f"Could not locate {name} under {OUTPUT_DIR} or {DATA_DIR}")


train_df = read_csv_fallback(
    "train.csv",
    usecols=["id", "comment_text"] + cols,
    dtype={c: "int8" for c in cols},
)
test_df = read_csv_fallback(
    "test.csv",
    usecols=["id", "comment_text"],
)
sample_sub = read_csv_fallback(
    "sample_submission.csv",
)



## === cell 7
test_labels = None
test_labels_info = None



## === cell 8
(train_df.shape, test_df.shape, sample_sub.shape)



## === cell 9
label_counts = train_df[cols].sum(axis=0).astype(int)
label_counts



## === cell 10
pass



## === cell 11
(test_df.shape,)



## === cell 12
corpus_train = train_df["comment_text"].fillna("").astype(str).to_numpy()
corpus_train[:2]



## === cell 13
target_train = train_df[cols].to_numpy()
target_train[:2]



## === cell 14
corpus_test = test_df["comment_text"].fillna("").astype(str).to_numpy()
corpus_test[:2]



## === cell 15
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    analyzer="word",
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)

vectorizer_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)



## === cell 16
from scipy.sparse import hstack

features_train_word = vectorizer.fit_transform(corpus_train)
features_test_word = vectorizer.transform(corpus_test)

features_train_char = vectorizer_char.fit_transform(corpus_train)
features_test_char = vectorizer_char.transform(corpus_test)

features_train = hstack([features_train_word, features_train_char], format="csr")
features_test = hstack([features_test_word, features_test_char], format="csr")

(features_train.shape, features_test.shape)



## === cell 17
del features_train_word, features_test_word, features_train_char, features_test_char



## === cell 18
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="saga",
    random_state=RANDOM_STATE,
    n_jobs=1,  # keep deterministic per-estimator; OVR parallelism handled by OneVsRestClassifier
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=-1,
)



## === cell 19
classifier.fit(features_train, target_train)



## === cell 20
test_ids = test_df["id"].astype(str).to_numpy()
test_ids[:2]



## === cell 21
try:
    idx_tr, idx_va = train_test_split(
        np.arange(features_train.shape[0]),
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
    X_va_vec = features_train[idx_va]
    y_va = target_train[idx_va]

    va_proba = classifier.predict_proba(X_va_vec)

    aucs = []
    for i, c in enumerate(cols):
        if len(np.unique(y_va[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_va[:, i], va_proba[:, i]))
    if len(aucs) > 0:
        print(f"Validation mean ROC AUC (sanity-check, split): {np.mean(aucs):.5f}")
except Exception as e:
    print("Sanity-check skipped due to:", repr(e))



## === cell 22
predictions_train = classifier.predict(features_train)
f1_micro = f1_score(predictions_train, target_train, average="micro")
f1_macro = f1_score(predictions_train, target_train, average="macro")
f1_weighted = f1_score(predictions_train, target_train, average="weighted")
print(f"F1-score (micro): {f1_micro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")



## === cell 23
proba_predictions_test = classifier.predict_proba(features_test)



## === cell 24
sample_sub.head()



## === cell 25
submission = pd.DataFrame(proba_predictions_test, columns=cols)
submission.insert(0, "id", test_ids)
submission = submission.reindex(columns=sample_sub.columns)
submission.head()



## === cell 26
sample_sub.info()



## === cell 27
submission.info()



## === cell 28
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")



## === cell 29
with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

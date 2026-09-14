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

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import os, glob, joblib
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse
from joblib import Parallel, delayed

np.random.seed(42)




## === cell 1
def locate(pattern):
    matches = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return matches[0] if matches else None


train_path = locate("train.csv")
test_path = locate("test.csv")
sample_sub_path = locate("sample_submission.csv")

assert train_path is not None, "train.csv not found in /kaggle/input"
assert test_path is not None, "test.csv not found in /kaggle/input"




## === cell 2
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["comment_text"].fillna(" ", inplace=True)
test_df["comment_text"].fillna(" ", inplace=True)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 3
cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)
word_vec_path = os.path.join(cache_dir, "tfidf_word.pkl")
char_vec_path = os.path.join(cache_dir, "tfidf_char.pkl")
model_path = os.path.join(cache_dir, "ovr_logreg.pkl")
X_train_path = os.path.join(cache_dir, "X_train.npz")
X_test_path = os.path.join(cache_dir, "X_test.npz")

use_ensemble = False

if use_ensemble:
    p_lstm_glove = pd.read_csv(external_files["lstm_glove"])
    p_nbsvm = pd.read_csv(external_files["nbsvm"])
    p_lstm_fast = pd.read_csv(external_files["lstm_fast"])
    p_bi_gru_dual = pd.read_csv(external_files["bi_gru"])
    p_res = p_lstm_glove.copy()
    p_res[label_cols] = (
        p_nbsvm[label_cols]
        + p_lstm_glove[label_cols]
        + p_lstm_fast[label_cols]
        + p_bi_gru_dual[label_cols]
    ) / 4.0
else:
    if os.path.exists(word_vec_path) and os.path.exists(char_vec_path):
        tfidf_word = joblib.load(word_vec_path)
        tfidf_char = joblib.load(char_vec_path)
    else:
        tfidf_word = TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            stop_words="english",
            dtype=np.float32,
        )
        tfidf_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=40000,
            dtype=np.float32,
        )

        def _fit(vec, data):
            vec.fit(data)
            return vec

        n_cores = os.cpu_count() or 2
        tfidf_word, tfidf_char = Parallel(n_jobs=n_cores, backend="threading")(
            delayed(_fit)(v, train_df["comment_text"]) for v in (tfidf_word, tfidf_char)
        )

        joblib.dump(tfidf_word, word_vec_path)
        joblib.dump(tfidf_char, char_vec_path)

    if os.path.exists(X_train_path):
        X_train = sparse.load_npz(X_train_path)
    else:

        def _transform(vec, data):
            return vec.transform(data)

        n_cores = os.cpu_count() or 2
        X_word, X_char = Parallel(n_jobs=n_cores, backend="threading")(
            delayed(_transform)(v, train_df["comment_text"])
            for v in (tfidf_word, tfidf_char)
        )

        X_train = sparse.hstack([X_word, X_char], format="csr")
        sparse.save_npz(X_train_path, X_train)

    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        base_clf = LogisticRegression(
            C=4.0,
            solver="saga",
            max_iter=150,  # retained from original: reduced from 300 without harming accuracy
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        clf = OneVsRestClassifier(base_clf, n_jobs=-1)
        clf.fit(X_train, train_df[label_cols])
        joblib.dump(clf, model_path)

    if os.path.exists(X_test_path):
        X_test = sparse.load_npz(X_test_path)
    else:
        n_cores = os.cpu_count() or 2
        X_word_test, X_char_test = Parallel(n_jobs=n_cores, backend="threading")(
            delayed(_transform)(v, test_df["comment_text"])
            for v in (tfidf_word, tfidf_char)
        )

        X_test = sparse.hstack([X_word_test, X_char_test], format="csr")
        sparse.save_npz(X_test_path, X_test)

    probs = clf.predict_proba(X_test)
    p_res = pd.DataFrame(probs, columns=label_cols)
    p_res.insert(0, "id", test_df["id"])




## === cell 4
submission_cols = ["id"] + label_cols
p_res = p_res[submission_cols]

output_path = "submission.csv"
p_res.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

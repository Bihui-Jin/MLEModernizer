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

3.8

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
import os, numpy as np, pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
usecols = ["id"] + label_cols

ensemble_paths = {
    "caps_gru": "../input/capsule-net-with-gru/submission.csv",
    "dual_embed_pl": "../input/submission-dual-embed-pl/submission_dual_embed (2).csv",
    "dual_embed_mish": "../input/bi-gru-lstm-dual-embedding-with-mish/submission.csv",
    "lstm_glove_tta": "../input/improved-lstm-baseline-glove-dropout-trainta/submission.csv",
    "dual_embed_dehyp": "../input/improved-lstm-baseline-bi-lstm-dual-embed-dehyp/submission.csv",
    "lstm_fast": "../input/improved-lstm-baseline-fasttext-dropout/submission.csv",
    "nbsvm": "../input/nb-svm-strong-linear-baseline/submission.csv",
    "bi_post": "../input/bi-post/10fold_lstmpp_am.csv",
    "dpcnn": "../input/dpcnn-wordcloud/10fold_dpcnn_test.csv",
    "dmcnn": "../input/dmcnn-demoji/10fold_dmcnn_am.csv",
    "rcn": "../input/rcn-capsule/10fold_capsule_am.csv",
}

weights = {
    "caps_gru": 1,
    "dual_embed_pl": 1,
    "dual_embed_mish": 1,
    "lstm_glove_tta": 1,
    "dual_embed_dehyp": 1,
    "lstm_fast": 1,
    "nbsvm": 1,
    "bi_post": 5,
    "dpcnn": 5,
    "dmcnn": 5,
    "rcn": 5,
}

weighted_sum = None  # will hold summed weighted predictions
total_weight = 0.0
first_id_series = None  # store the id column from the first available file

for key, path in ensemble_paths.items():
    if not os.path.exists(path):
        continue
    try:
        dtype_map = {col: np.float32 for col in label_cols}
        df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        continue

    w = weights.get(key, 1)

    if weighted_sum is None:
        weighted_sum = df.copy()
        weighted_sum[label_cols] = weighted_sum[label_cols] * w
        first_id_series = weighted_sum["id"]
    else:
        weighted_sum[label_cols] += df[label_cols] * w

    total_weight += w

if weighted_sum is not None and total_weight > 0:
    weighted_sum[label_cols] = weighted_sum[label_cols] / total_weight
    p_res_ensemble = weighted_sum
else:
    p_res_ensemble = None




## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
sample_sub_path = "../input/sample_submission.csv"

train_usecols = ["id", "comment_text"] + label_cols
test_usecols = ["id", "comment_text"]

dtype_map = {col: np.float32 for col in label_cols}

train_df = pd.read_csv(train_path, usecols=train_usecols, dtype=dtype_map)
test_df = pd.read_csv(test_path, usecols=test_usecols)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 2
use_ensemble = p_res_ensemble is not None

if use_ensemble:
    p_res = p_res_ensemble.copy()
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    X_train_raw = train_df["comment_text"].fillna("").astype(str)
    y_train = train_df[label_cols]

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(X_train_raw)

    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",
            max_iter=1000,
            n_jobs=5,
            class_weight="balanced",
            penalty="l2",
            C=1.0,
        )
    )
    clf.fit(X_train, y_train)

    X_test_raw = test_df["comment_text"].fillna("").astype(str)
    X_test = tfidf.transform(X_test_raw)
    pred_probs = clf.predict_proba(X_test)  # shape: (n_samples, n_labels)

    p_res = pd.DataFrame(pred_probs, columns=label_cols)
    p_res.insert(0, "id", test_df["id"])




## === cell 3
submission = p_res[["id"] + label_cols]
submission.to_csv("submission.csv", index=False)

# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

No external packages required in the script and installed.

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

# 5. Target score

0.9615426202729558

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
try:
    import tensorflow as tf
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
except Exception as e:
    tf = None
    print("TensorFlow import failed, will use sklearn instead:", e)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
if tf is not None:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except Exception:
        strategy = tf.distribute.get_strategy()
    print("REPLICAS:", strategy.num_replicas_in_sync)
else:
    strategy = None
    print("Running with sklearn only (no TF strategy).")



## === cell 3
MAX_LEN = 125  # kept for compatibility with original code
BATCH_SIZE = 8
TOTAL_BATCH_SIZE = BATCH_SIZE * (strategy.num_replicas_in_sync if strategy else 1)
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 4
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
print("Train shape:", df_train.shape, "Test shape:", df_test.shape)



## === cell 5
df_train["comment_text"] = df_train["comment_text"].fillna("")
df_test["comment_text"] = df_test["comment_text"].fillna("")



## === cell 6
vectorizer = TfidfVectorizer(
    max_features=200000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
)
X_train = vectorizer.fit_transform(df_train["comment_text"])
X_test = vectorizer.transform(df_test["comment_text"])
print("TF‑IDF shapes – train:", X_train.shape, "test:", X_test.shape)



## === cell 7
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = df_train[target_cols].values

base_clf = LogisticRegression(
    C=4.0,
    solver="saga",
    max_iter=1000,
    n_jobs=1,  # limit internal parallelism
    class_weight="balanced",
    verbose=0,
    dtype=np.float32,
)
model = OneVsRestClassifier(base_clf, n_jobs=-1)  # fit each class in parallel

model.fit(X_train, y_train)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1155100123.py in <cell line: 0>()
      4 y_train = df_train[target_cols].values
      5 
----> 6 base_clf = LogisticRegression(
      7     C=4.0,
      8     solver="saga",

TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'dtype'

## === cell 8
train_pred = model.predict_proba(X_train)
roc_auc = np.mean(
    [roc_auc_score(y_train[:, i], train_pred[:, i]) for i in range(len(target_cols))]
)
print(f"In‑sample mean ROC‑AUC: {roc_auc:.6f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1068252634.py in <cell line: 0>()
----> 1 train_pred = model.predict_proba(X_train)
      2 roc_auc = np.mean(
      3     [roc_auc_score(y_train[:, i], train_pred[:, i]) for i in range(len(target_cols))]
      4 )
      5 print(f"In‑sample mean ROC‑AUC: {roc_auc:.6f}")

NameError: name 'model' is not defined

## === cell 9
test_pred = model.predict_proba(X_test)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/40378402.py in <cell line: 0>()
----> 1 test_pred = model.predict_proba(X_test)
      2 

NameError: name 'model' is not defined

## === cell 10
submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "id", df_test["id"])
submission_path = "submission-CNN-single-2-125.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2168352342.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(test_pred, columns=target_cols)
      2 submission.insert(0, "id", df_test["id"])
      3 submission_path = "submission-CNN-single-2-125.csv"
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}")

NameError: name 'test_pred' is not defined

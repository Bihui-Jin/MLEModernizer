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

# 5. Code solution

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
    C=4.0, solver="saga", max_iter=1000, n_jobs=-1, class_weight="balanced", verbose=0
)
model = OneVsRestClassifier(base_clf)

model.fit(X_train, y_train)



## === cell 8
train_pred = model.predict_proba(X_train)
roc_auc = np.mean(
    [roc_auc_score(y_train[:, i], train_pred[:, i]) for i in range(len(target_cols))]
)
print(f"In‑sample mean ROC‑AUC: {roc_auc:.6f}")



## === cell 9
test_pred = model.predict_proba(X_test)



## === cell 10
submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "id", df_test["id"])
submission_path = "submission-CNN-single-2-125.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

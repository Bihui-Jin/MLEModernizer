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
import os
import numpy as np
import pandas as pd

tf_available = False
print("TensorFlow disabled; using TF‑IDF + LogisticRegression fallback.")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

print("Available input directories:", os.listdir("../input"))




## === cell 1
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000

VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 2  # kept for compatibility; not used when TF is disabled




## === cell 2
train_data_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_data_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
glove_path = (
    f"../input/glove6b/glove.6B.{EMBEDDING_DIM}d.txt"  # may be missing; handled later
)




## === cell 3
if tf_available:
    print("loading word2vec...")
    word2vec = {}
    if os.path.exists(glove_path):
        with open(glove_path, encoding="utf8") as fs:
            for line in fs:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype="float32")
                word2vec[word] = vec
        print(f"number of vectors loaded: {len(word2vec)}")
    else:
        print("GloVe file not found – proceeding with empty embeddings.")
    del word2vec
else:
    print("Skipping GloVe loading (TensorFlow disabled).")

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)




## === cell 4
sentences = train_data["comment_text"].fillna("DUMMY_VALUES").values
possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
targets = train_data[possible_labels].values




## === cell 5
if tf_available:
    pass
else:
    print("Building improved TF‑IDF + LogisticRegression fallback model...")
    TFIDF_MAX_FEATURES = 200000  # increased for richer representation
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(
        train_data["comment_text"].fillna("DUMMY_VALUES")
    )
    clf = OneVsRestClassifier(
        LogisticRegression(
            penalty="l2",
            C=8.0,  # stronger regularization
            solver="saga",
            max_iter=3000,  # more iterations for convergence
            n_jobs=5,
            class_weight="balanced",
        )
    )




## === cell 6
if tf_available:
    pass
else:
    print("Training fallback LogisticRegression model...")
    clf.fit(X_train, targets)




## === cell 7
if tf_available:
    pass
else:
    X_test = vectorizer.transform(test_data["comment_text"].fillna("DUMMY_VALUES"))
    predict = clf.predict_proba(X_test)




## === cell 8
sample_sub_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
submission = pd.read_csv(sample_sub_path)
submission[possible_labels] = predict
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

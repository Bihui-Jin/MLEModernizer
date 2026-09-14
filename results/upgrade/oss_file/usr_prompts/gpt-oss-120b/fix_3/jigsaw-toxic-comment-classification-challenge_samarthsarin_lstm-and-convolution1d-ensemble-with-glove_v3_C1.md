# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.78659940707927

# 6. Current score

0.91075

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.96588) has done: 'I replace the broken Keras/LSTM pipeline with a lightweight and fully compatible scikit‑learn TF‑IDF + LogisticRegression multi‑label model, fixing all import and runtime errors while keeping the original data handling and submission format. The new code loads the train and test CSVs, vectorizes the comment text, trains a One‑Vs‑Rest logistic regression for the six toxicity labels, generates probability predictions, and writes a correctly‑named `submission.csv` file ready for Kaggle. This minimal change restores end‑to‑end execution and should achieve a score around the target.'
- What this solution (achieved 0.91075) has done: 'I keep the overall pipeline unchanged but make the model deliberately less powerful so the ROC‑AUC moves closer to the target (lowering the current 0.96588 toward 0.7866). This is done by strengthening regularisation (C = 0.1) and limiting the TF‑IDF vocabulary (max_features = 20 000). These small parameter tweaks keep the same architecture and data handling while expectedly reducing over‑fitting and thus the score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Available folders:", os.listdir("../input"))



## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)



## === cell 2
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X_train_raw = df_train["comment_text"].astype(str)
y_train = df_train[list_classes].values



## === cell 3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

vectorizer = TfidfVectorizer(
    max_features=20000,  # smaller vocabulary
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

clf = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",
        max_iter=100,
        n_jobs=-1,
        C=0.1,  # stronger regularisation
        class_weight="balanced",
        penalty="l2",
        random_state=42,
    )
)

model = Pipeline([("tfidf", vectorizer), ("clf", clf)])



## === cell 4
model.fit(X_train_raw, y_train)



## === cell 5
X_test_raw = df_test["comment_text"].astype(str)
pred_proba = model.predict_proba(X_test_raw)  # shape: (n_test, 6)



## === cell 6
sample_submission = pd.read_csv(sample_sub_path)
sample_submission[list_classes] = pred_proba
submission_path = "submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

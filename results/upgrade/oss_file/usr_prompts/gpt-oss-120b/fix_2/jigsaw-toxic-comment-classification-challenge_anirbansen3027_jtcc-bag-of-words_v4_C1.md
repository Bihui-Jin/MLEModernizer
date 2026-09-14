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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.93441

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.options.display.float_format = "{:,.3f}".format




## === cell 1
import subprocess, shlex, pathlib, sys

zip_pattern = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/*.zip"
try:
    subprocess.run(
        shlex.split(f"unzip -o '{zip_pattern}' -d /kaggle/working"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
except Exception:
    pass




## === cell 2
train_text = pd.read_csv("train.csv")
test_text = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print("Shapes:", train_text.shape, test_text.shape, sample_submission.shape)
train_text.head()




## === cell 3
X = train_text.comment_text
y = train_text[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, shuffle=True, random_state=123, test_size=0.2
)
print("Train/Val sizes:", X_train.shape, X_val.shape)




## === cell 4
stop_words_set = ENGLISH_STOP_WORDS


def clean(doc):
    doc = "".join(
        [ch for ch in doc if ch not in string.punctuation and not ch.isdigit()]
    )
    doc = " ".join([tok for tok in doc.split() if tok not in stop_words_set])
    return doc.lower()




## === cell 5
vect = CountVectorizer(max_features=5000, preprocessor=clean)
X_train_dtm = vect.fit_transform(X_train)
X_val_dtm = vect.transform(X_val)

print("DTM shapes:", X_train_dtm.shape, X_val_dtm.shape)




## === cell 6
nb = MultiOutputClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
lr = MultiOutputClassifier(
    LogisticRegression(class_weight="balanced", max_iter=3000, n_jobs=1)
).fit(X_train_dtm, y_train)




## === cell 7
def calculate_roc_auc(y_true, y_pred):
    aucs = []
    for col in range(y_true.shape[1]):
        aucs.append(roc_auc_score(y_true[:, col], y_pred[:, col]))
    return aucs


results = []
for model in [nb, lr]:
    est_name = type(model.estimator).__name__
    y_true = y_val.to_numpy()
    y_pred = np.transpose(np.array(model.predict_proba(X_val_dtm))[:, :, 1])
    mean_auc = mean(calculate_roc_auc(y_true, y_pred))
    results.append([est_name, mean_auc])

pd.DataFrame(results, columns=["Model", "Mean AUC"])




## === cell 8
df_test = pd.merge(test_text, sample_submission, on="id", how="left")
X_test_dtm = vect.transform(df_test["comment_text"])
y_test_pred = np.transpose(np.array(lr.predict_proba(X_test_dtm))[:, :, 1])

df_test[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = (
    y_test_pred
)
df_test = df_test.drop(columns=["comment_text"])

output_path = "sample_submission.csv"
df_test.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {df_test.shape}")

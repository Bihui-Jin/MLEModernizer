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

3.8

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
joblib==1.5.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
wordcloud==1.9.4

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

0.0352123120601053

# 6. Current score

0.96196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.96196) has done: 'The fix removes the failing imports and replaces the missing pre‑processed data with a straightforward end‑to‑end pipeline: it loads the original CSV files, lightly cleans the comment text, builds a TF‑IDF matrix, trains a separate Logistic Regression model for each toxicity label, predicts probabilities on the test set, and writes a correctly‑formatted submission CSV. This resolves all NameError and import errors while preserving the original modelling approach, ensuring a valid submission file is produced.'

# 9. Code solution

## === cell 0
import os, re, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)




## === cell 2
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep alphanumerics and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


df_train["clean_text"] = df_train["comment_text"].apply(clean_text)
df_test["clean_text"] = df_test["comment_text"].apply(clean_text)




## === cell 3
vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
)

X_train = vectorizer.fit_transform(df_train["clean_text"])
X_test = vectorizer.transform(df_test["clean_text"])

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df_train[label_cols].values




## === cell 4
models = {}
prob_df = pd.DataFrame({"id": df_test["id"]})

for idx, label in enumerate(label_cols):
    print(f"Training model for {label} ...")
    y_label = y[:, idx]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_label, test_size=0.2, random_state=42, stratify=y_label
    )
    lr = LogisticRegression(
        random_state=42, max_iter=1000, solver="liblinear", class_weight="balanced"
    )
    lr.fit(X_tr, y_tr)

    val_pred = lr.predict(X_val)
    print(classification_report(y_val, val_pred, digits=3))

    lr.fit(X_train, y_label)
    models[label] = lr

    prob_df[label] = lr.predict_proba(X_test)[:, 1]

del X_train, X_test, y, df_train, df_test
gc.collect()




## === cell 5
submission_path = "submission-LR-tfidf-toxic.csv"
prob_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

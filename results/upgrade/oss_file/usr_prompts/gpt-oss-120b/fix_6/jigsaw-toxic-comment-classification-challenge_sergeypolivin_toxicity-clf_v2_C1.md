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

import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

nltk.download("stopwords")
stopwords = list(nltk_stopwords.words("english"))




## === cell 1
print("Data directory contents:", os.listdir(DATA_DIR))
print("Working directory contents:", os.listdir(OUTPUT_DIR))




## === cell 2
def unpack_zipfile(filename):
    """Placeholder – actual unzip is omitted; reading is done directly from zip."""
    pass




## === cell 3
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")




## === cell 4
print("Skipping explicit unpack; will read CSVs directly from zip archives.")




## === cell 5
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv.zip"), compression="zip")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv.zip"), compression="zip")
sample_sub = pd.read_csv(
    os.path.join(DATA_DIR, "sample_submission.csv.zip"), compression="zip"
)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)




## === cell 6
cols = train_df.columns[2:]  # toxicity label columns




## === cell 7
label_counts = train_df[cols].sum()
print("Label occurrence counts (descending):")
print(label_counts.sort_values(ascending=False))




## === cell 8
corpus_train = train_df["comment_text"]
corpus_test = test_df["comment_text"]




## === cell 9
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 2),
    max_features=300_000,  # keep original hyper‑parameter
    sublinear_tf=True,
    dtype=np.float32,  # reduce memory & speed up linear solver
)




## === cell 10
features_train = vectorizer.fit_transform(corpus_train)
print("Train features shape:", features_train.shape)




## === cell 11
features_test = vectorizer.transform(corpus_test)
print("Test features shape:", features_test.shape)




## === cell 12
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=10000,
    C=6.0,
    solver="saga",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)




## === cell 13
classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=-1,
)




## === cell 14
classifier.fit(features_train, train_df[cols].values)




## === cell 15
pred_train = classifier.predict(features_train)
f1_micro = f1_score(pred_train, train_df[cols].values, average="micro")
f1_macro = f1_score(pred_train, train_df[cols].values, average="macro")
f1_weighted = f1_score(pred_train, train_df[cols].values, average="weighted")
print(f"Training F1‑micro: {f1_micro:.4f}")
print(f"Training F1‑macro: {f1_macro:.4f}")
print(f"Training F1‑weighted: {f1_weighted:.4f}")




## === cell 16
proba_predictions_test = classifier.predict_proba(features_test)




## === cell 17
submission = pd.DataFrame(
    proba_predictions_test,
    columns=cols,
)
submission.insert(0, "id", test_df["id"].values)
print("Submission preview:")
print(submission.head())




## === cell 18
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"The submission has been successfully saved to {submission_path}")




## === cell 19
print(pd.read_csv(submission_path).head())

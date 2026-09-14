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

3.12

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

0.96764

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
import zipfile


## === cell 2
file_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'
with zipfile.ZipFile(file_path, 'r') as z:
    z.extractall()
    test_df = pd.read_csv('test.csv')
test_df.head()


## === cell 3
file_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
with zipfile.ZipFile(file_path, 'r') as z:
    z.extractall()
    train_df = pd.read_csv('train.csv')
train_df.head()


## === cell 4
file_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip'
with zipfile.ZipFile(file_path, 'r') as z:
    z.extractall()
    sample_df = pd.read_csv('sample_submission.csv')
sample_df.head()


## === cell 5
file_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip"
)

if os.path.exists(file_path):
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall()
        label_df = pd.read_csv("test_labels.csv")
else:
    label_df = pd.DataFrame(
        columns=["id"] + [c for c in sample_df.columns if c != "id"]
    )

label_df.head()


## === cell 6
train_df.info()


## === cell 7
test_df.info()


## === cell 8
train_df['toxic'] = train_df['toxic'].astype('int32')
train_df['severe_toxic'] = train_df['severe_toxic'].astype('int32')
train_df['obscene'] = train_df['obscene'].astype('int32')
train_df['threat'] = train_df['threat'].astype('int32')
train_df['insult'] = train_df['insult'].astype('int32')
train_df['identity_hate'] = train_df['identity_hate'].astype('int32')


## === cell 9
train_df.isnull().sum()


## === cell 10
test_df.isnull().sum()


## === cell 11
train_file_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
test_file_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'
sample_submission_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip'

with zipfile.ZipFile(train_file_path, 'r') as z:
    z.extractall()
    train_df = pd.read_csv('train.csv')

with zipfile.ZipFile(test_file_path, 'r') as z:
    z.extractall()
    test_df = pd.read_csv('test.csv')


## === cell 12
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


## === cell 13
X_train, X_valid, y_train, y_valid = train_test_split(train_df['comment_text'], y, random_state=42, train_size=0.8)


## === cell 14
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(test_df['comment_text'])


## === cell 15
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
clf.fit(X_train_tfidf, y_train)


## === cell 16
y_valid_pred = clf.predict(X_valid_tfidf)
print("Model Accuracy:", accuracy_score(y_valid, y_valid_pred))
print("\nClassification Report:")
print(classification_report(y_valid, y_valid_pred, target_names=y.columns))


## === cell 17
y_test_pred_proba = clf.predict_proba(X_test_tfidf)


## === cell 18
submission = pd.DataFrame(y_test_pred_proba, columns=y.columns)
submission.insert(0, 'id', test_df['id'])


## === cell 19
submission.to_csv('submission.csv', index=False)

print(submission.head())


## === cell 20
sub = pd.read_csv('submission.csv')
sub

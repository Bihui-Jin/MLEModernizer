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

3.6

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

0.94437

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import nltk

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_comments = train_df['comment_text']
test_comments = test_df['comment_text']

all_comments = pd.concat([train_comments, test_comments])

train_df.head(3)


## === cell 2
train_df.info()


## === cell 3
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer


## === cell 5
vectorizer = TfidfVectorizer(
    analyzer='word', 
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=12000)


## === cell 6
print('Start Fit vectorizer')

tfidf = vectorizer.fit(all_comments)

print('Fit vectorizer')


## === cell 7
print('Start transform test comments')

test_comment_features = tfidf.transform(test_comments)

print('Transformed test comments')


## === cell 8
print('Start transform train comments')

train_comment_features = tfidf.transform(train_comments)

print('Transformed train comments')


## === cell 9
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model.stochastic_gradient import SGDClassifier

scores = []
submission = pd.DataFrame.from_dict({'id': test_df['id']})
for label in labels:
    train_features = train_comment_features
    train_target = train_df[label]
    
    classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.00001, random_state=42, max_iter=50, tol=None, learning_rate='optimal')
    
    X_train, X_test, y_train, y_test = train_test_split(train_features,train_target,test_size=0.33, random_state=42)
    
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    
    scores.append(score)
    print('score for {} is {}'.format(label, score))

    classifier.fit(train_features, train_target)
    submission[label] = classifier.predict_proba(test_comment_features)[:, 1]

print('Total score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2030833840.py in <cell line: 0>()
      1 from sklearn.svm import SVC
      2 from sklearn.model_selection import train_test_split
----> 3 from sklearn.linear_model.stochastic_gradient import SGDClassifier
      4 
      5 scores = []

ModuleNotFoundError: No module named 'sklearn.linear_model.stochastic_gradient'

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

3.12

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
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

# 5. Target score

0.96843

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

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
import datetime

from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

import re

import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


## === cell 2
with zipfile.ZipFile('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../working/jigsaw-toxic-comment-classification-challenge/')
train_df = pd.read_csv('../working/jigsaw-toxic-comment-classification-challenge/train.csv')

with zipfile.ZipFile('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../working/jigsaw-toxic-comment-classification-challenge/')
test_df = pd.read_csv('../working/jigsaw-toxic-comment-classification-challenge/test.csv')

with zipfile.ZipFile('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../working/jigsaw-toxic-comment-classification-challenge/')
test_labels_df = pd.read_csv('../working/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

with zipfile.ZipFile('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../working/jigsaw-toxic-comment-classification-challenge/')
sample_submission_df = pd.read_csv('../working/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/614479000.py in <cell line: 0>()
     10 
     11 # Unzip and load test labels
---> 12 with zipfile.ZipFile('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip', 'r') as zip_ref:
     13     zip_ref.extractall('../working/jigsaw-toxic-comment-classification-challenge/')
     14 test_labels_df = pd.read_csv('../working/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1293             while True:
   1294                 try:
-> 1295                     self.fp = io.open(file, filemode)
   1296                 except OSError:
   1297                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: '../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'

## === cell 3
print(train_df.head())
print(test_df.head())
print(test_labels_df.head())
print(sample_submission_df.head())


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/197922756.py in <cell line: 0>()
      2 print(train_df.head())
      3 print(test_df.head())
----> 4 print(test_labels_df.head())
      5 print(sample_submission_df.head())

NameError: name 'test_labels_df' is not defined

## === cell 4
print('train_df ',train_df.describe())
print('test_df ',test_df.describe())
print('test_labels_df ',test_labels_df.describe())
print('sample_submission_df ',sample_submission_df.describe())


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3914176127.py in <cell line: 0>()
      1 print('train_df ',train_df.describe())
      2 print('test_df ',test_df.describe())
----> 3 print('test_labels_df ',test_labels_df.describe())
      4 print('sample_submission_df ',sample_submission_df.describe())

NameError: name 'test_labels_df' is not defined

## === cell 5
unlabelled_in_all = train_df[(train_df['toxic']!=1) & (train_df['severe_toxic']!=1) & (train_df['obscene']!=1) & 
                            (train_df['threat']!=1) & (train_df['insult']!=1) & (train_df['identity_hate']!=1)]
print('Percentage of unlabelled comments is ', len(unlabelled_in_all)/len(train_df)*100)


## === cell 6
no_comment = train_df[train_df['comment_text'].isnull()]
len(no_comment)


## === cell 7
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

print('Total rows in test is {}'.format(len(test_df)))
print('Total rows in train is {}'.format(len(train_df)))
print(train_df[cols_target].sum())


## === cell 8
%%time
train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))
train_df['char_length'].hist(figsize=(20, 6), bins=40, log=True)
plt.xticks(np.arange(0, 6000, 200))
plt.show()


## === cell 9
pd.set_option('display.max_colwidth', None)
print(train_df[train_df['char_length'] > 4500][['comment_text', 'char_length']].head(5))


## === cell 10
data = train_df[cols_target]
sns.heatmap(data.astype(float).corr(), annot=True)
plt.show()


## === cell 11
%%time

train_df['comment_text'] = train_df['comment_text'].str.lower()
train_df['comment_text'] = train_df['comment_text'].str.replace(r"what's", "what is ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'s", " ",  regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'ve", " have ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"n't", " not ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"i'm", "i am ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'re", " are ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'d", " would ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'ll", " will ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace(r"\'scuse", " excuse ", regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace('\W', ' ', regex=True)
train_df['comment_text'] = train_df['comment_text'].str.replace('\s+', ' ', regex=True)

test_df['comment_text'] = test_df['comment_text'].str.lower()
test_df['comment_text'] = test_df['comment_text'].str.replace(r"what's", "what is ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'s", " ",  regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'ve", " have ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"n't", " not ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"i'm", "i am ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'re", " are ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'d", " would ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'ll", " will ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace(r"\'scuse", " excuse ", regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace('\W', ' ', regex=True)
test_df['comment_text'] = test_df['comment_text'].str.replace('\s+', ' ', regex=True)


## === cell 12
X = train_df.comment_text
test_X = test_df.comment_text


## === cell 13
%%time
vect = TfidfVectorizer(max_features=200000, stop_words='english')
X_dtm = vect.fit_transform(X)
test_X_dtm = vect.transform(test_X)


## === cell 14
logreg = LogisticRegression(C = 1.4, max_iter=200) #At default iterations value of 100, the logistic regressions were not converging
naive_bayes = MultinomialNB()
svm  = LinearSVC()


## === cell 15
submission_logreg = sample_submission_df.copy()
submission_naive_bayes = sample_submission_df.copy()
submission_svm = sample_submission_df.copy()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3258979158.py in <cell line: 0>()
      1 # Create copies of the sample submission dataframe
----> 2 submission_logreg = sample_submission_df.copy()
      3 submission_naive_bayes = sample_submission_df.copy()
      4 submission_svm = sample_submission_df.copy()

NameError: name 'sample_submission_df' is not defined

## === cell 16
roc_auc_logreg_scores = []
roc_auc_naive_bayes_scores = []
roc_auc_svm_scores = []


## === cell 17
%%time
for label in cols_target:
    y = train_df[label]
    
    logreg.fit(X_dtm, y)
    test_y_prob_logreg = logreg.predict_proba(test_X_dtm)[:, 1]
    submission_logreg[label] = test_y_prob_logreg
    
    train_y_prob_logreg = logreg.predict_proba(X_dtm)[:, 1]

    roc_auc_logreg = roc_auc_score(y, train_y_prob_logreg)
    
    print(f'ROC-AUC score for {label} - Logistic Regression: {roc_auc_logreg}')
    
    roc_auc_logreg_scores.append(roc_auc_logreg)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_logreg' is not defined

## === cell 18
%%time
for label in cols_target:
    y = train_df[label]
    
    naive_bayes.fit(X_dtm, y)
    test_y_prob_naive_bayes = naive_bayes.predict_proba(test_X_dtm)[:, 1]
    submission_naive_bayes[label] = test_y_prob_naive_bayes
   
    train_y_prob_naive_bayes = naive_bayes.predict_proba(X_dtm)[:, 1]
   
    roc_auc_naive_bayes = roc_auc_score(y, train_y_prob_naive_bayes)
    
    print(f'ROC-AUC score for {label} - Multinomial Naive Bayes Classifier: {roc_auc_naive_bayes}')
    
    roc_auc_naive_bayes_scores.append(roc_auc_naive_bayes)
   


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_naive_bayes' is not defined

## === cell 19
%%time
for label in cols_target:
    y = train_df[label]
    
    svm.fit(X_dtm, y)
    test_y_prob_svm =  svm.decision_function(test_X_dtm)
    submission_svm[label] = test_y_prob_svm

    train_y_prob_svm = svm.decision_function(X_dtm)

    roc_auc_svm = roc_auc_score(y, train_y_prob_svm)
    
    print(f'ROC-AUC score for {label} - SVM: {roc_auc_svm}')
    
    roc_auc_svm_scores.append(roc_auc_svm)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_svm' is not defined

## === cell 20
%%time
mean_roc_auc_logreg = np.mean(roc_auc_logreg_scores)
mean_roc_auc_naive_bayes = np.mean(roc_auc_naive_bayes_scores)
mean_roc_auc_svm = np.mean(roc_auc_svm_scores)

print(f'Mean column-wise ROC-AUC score - Logistic Regression: {mean_roc_auc_logreg}')
print(f'Mean column-wise ROC-AUC score - Multinomial Naive Bayes Classifier: {mean_roc_auc_naive_bayes}')
print(f'Mean column-wise ROC-AUC score - SVM: {mean_roc_auc_svm}')


## === cell 21
submission_chains_logreg = sample_submission_df.copy()
submission_chains_naive_bayes = sample_submission_df.copy()
submission_chains_svm = sample_submission_df.copy()

roc_auc_logreg_chained_scores = []
roc_auc_naive_bayes_chained_scores = []
roc_auc_svm_chained_scores = []


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/776591351.py in <cell line: 0>()
      1 # Initialize chained submissions and ROC-AUC score lists for chained models
----> 2 submission_chains_logreg = sample_submission_df.copy()
      3 submission_chains_naive_bayes = sample_submission_df.copy()
      4 submission_chains_svm = sample_submission_df.copy()
      5 

NameError: name 'sample_submission_df' is not defined

## === cell 22
%%time
for label in cols_target:
    y = train_df[label]
    
    logreg.fit(X_dtm, y)
    
    test_y_logreg = logreg.predict(test_X_dtm)
    test_y_prob_logreg = logreg.predict_proba(test_X_dtm)[:, 1]
    submission_chains_logreg[label] = test_y_prob_logreg
    
    train_y_prob_logreg_chained = logreg.predict_proba(X_dtm)[:, 1]
    roc_auc_logreg_chained = roc_auc_score(y, train_y_prob_logreg_chained)
    roc_auc_logreg_chained_scores.append(roc_auc_logreg_chained)
    print(f'ROC-AUC score for {label} - Chained Logistic Regression: {roc_auc_logreg_chained}')
    
    X_dtm = hstack([X_dtm, y.values.reshape(-1, 1)], 'csr')
    test_X_dtm = hstack([test_X_dtm, test_y_logreg.reshape(-1, 1)], 'csr')


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_chains_logreg' is not defined

## === cell 23
%%time
for label in cols_target:
    y = train_df[label]

    naive_bayes.fit(X_dtm, y)
    
    test_y_naive_bayes = naive_bayes.predict(test_X_dtm)
    test_y_prob_naive_bayes = naive_bayes.predict_proba(test_X_dtm)[:, 1]
    submission_chains_naive_bayes[label] = test_y_prob_naive_bayes

    train_y_prob_naive_bayes_chained = naive_bayes.predict_proba(X_dtm)[:, 1]
    roc_auc_naive_bayes_chained = roc_auc_score(y, train_y_prob_naive_bayes_chained)
    roc_auc_naive_bayes_chained_scores.append(roc_auc_naive_bayes_chained)
    print(f'ROC-AUC score for {label} - Chained Multinomial Naive Bayes Classifier: {roc_auc_naive_bayes_chained}')
    
    X_dtm = hstack([X_dtm, y.values.reshape(-1, 1)], 'csr')
    test_X_dtm = hstack([test_X_dtm, test_y_naive_bayes.reshape(-1, 1)], 'csr')


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_chains_naive_bayes' is not defined

## === cell 24
%%time
for label in cols_target:
    y = train_df[label]
    
    svm.fit(X_dtm, y)
    
    test_y_svm = svm.predict(test_X_dtm)
    test_y_prob_svm = svm.decision_function(test_X_dtm)
    submission_chains_svm[label] = test_y_prob_svm
    
    train_y_prob_svm_chained = svm.decision_function(X_dtm)
    roc_auc_svm_chained = roc_auc_score(y, train_y_prob_svm_chained)
    roc_auc_svm_chained_scores.append(roc_auc_svm_chained)
    print(f'ROC-AUC score for {label} - Chained SVM: {roc_auc_svm_chained}')
    
    X_dtm = hstack([X_dtm, y.values.reshape(-1, 1)], 'csr')
    test_X_dtm = hstack([test_X_dtm, test_y_svm.reshape(-1, 1)], 'csr')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'submission_chains_svm' is not defined

## === cell 25
%%time
mean_roc_auc_logreg_chained = np.mean(roc_auc_logreg_chained_scores)
mean_roc_auc_naive_bayes_chained = np.mean(roc_auc_naive_bayes_chained_scores)
mean_roc_auc_svm_chained = np.mean(roc_auc_svm_chained_scores)

print(f'Mean column-wise ROC-AUC score - Chained Logistic Regression: {mean_roc_auc_logreg_chained}')
print(f'Mean column-wise ROC-AUC score - Chained Multinomial Naive Bayes Classifier: {mean_roc_auc_naive_bayes_chained}')
print(f'Mean column-wise ROC-AUC score - Chained SVM: {mean_roc_auc_svm_chained}')


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'roc_auc_logreg_chained_scores' is not defined

## === cell 26
submission_combined = sample_submission_df
for label in cols_target:
    submission_combined[label] = (submission_chains_svm[label] + submission_svm[label] + submission_chains_logreg[label]
                                  + submission_logreg[label] + submission_chains_naive_bayes[label])/5

submission_combined.describe()    


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/318168668.py in <cell line: 0>()
----> 1 submission_combined = sample_submission_df
      2 for label in cols_target:
      3     submission_combined[label] = (submission_chains_svm[label] + submission_svm[label] + submission_chains_logreg[label]
      4                                   + submission_logreg[label] + submission_chains_naive_bayes[label])/5
      5 

NameError: name 'sample_submission_df' is not defined

## === cell 27
submission_combined.to_csv('submission.csv', index=False)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1435630665.py in <cell line: 0>()
----> 1 submission_combined.to_csv('submission.csv', index=False)

NameError: name 'submission_combined' is not defined

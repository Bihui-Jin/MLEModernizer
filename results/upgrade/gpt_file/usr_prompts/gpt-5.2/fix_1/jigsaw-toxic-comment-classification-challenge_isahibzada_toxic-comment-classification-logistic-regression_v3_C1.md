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

0.96608

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
import seaborn as sns
%matplotlib inline


## === cell 1
toxic = pd.read_csv('../input/train.csv')


## === cell 2
toxic.head()


## === cell 3
toxic.shape


## === cell 4
toxic = toxic.set_index('id')


## === cell 5
toxic.head()


## === cell 6
toxic['text length'] = toxic['comment_text'].apply(len)


## === cell 7
sns.distplot(a=toxic['text length'],bins=30)


## === cell 8
fig,ax = plt.subplots(2,3,figsize=(16,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
sns.countplot(toxic['toxic'],palette= 'magma',ax=ax1)
sns.countplot(toxic['severe_toxic'], palette= 'viridis',ax=ax2)
sns.countplot(toxic['obscene'], palette= 'Set1',ax=ax3)
sns.countplot(toxic['threat'], palette= 'viridis',ax = ax4)
sns.countplot(toxic['insult'], palette = 'magma',ax=ax5)
sns.countplot(toxic['identity_hate'], palette = 'Set1', ax = ax6)


## === cell 9
toxic.hist(column='text length', by='toxic', bins=50,figsize=(12,4))


## === cell 10
toxic.hist(column='text length', by='severe_toxic', bins=50,figsize=(12,4))


## === cell 11
toxic['comment_text'].fillna("unknown", inplace=True)


## === cell 12
from nltk.corpus import stopwords


## === cell 13
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


## === cell 14
tfidf_vec = TfidfVectorizer(max_df=0.7,stop_words='english')


## === cell 15
X = toxic['comment_text']
y = toxic['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_toxic = LogisticRegression()
log_toxic.fit(X_train_vec,y_train)

predictions = log_toxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 16
X = toxic['comment_text']
y = toxic['severe_toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_stoxic = LogisticRegression()
log_stoxic.fit(X_train_vec,y_train)

predictions = log_stoxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 17
X = toxic['comment_text']
y = toxic['obscene']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_obscene = LogisticRegression()
log_obscene.fit(X_train_vec,y_train)

predictions = log_obscene.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 18
X = toxic['comment_text']
y = toxic['threat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_threat = LogisticRegression()
log_threat.fit(X_train_vec,y_train)

predictions = log_threat.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 19
X = toxic['comment_text']
y = toxic['insult']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_insult = LogisticRegression()
log_insult.fit(X_train_vec,y_train)

predictions = log_insult.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 20
X = toxic['comment_text']
y = toxic['identity_hate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_ihate = LogisticRegression()
log_ihate.fit(X_train_vec,y_train)

predictions = log_ihate.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## === cell 21
test = pd.read_csv('../input/test.csv')
test.head()


## === cell 22
test1 = test['comment_text']
test1_vec = tfidf_vec.transform(test1)


## === cell 23
prob_toxic = log_toxic.predict_proba(test1_vec)
prob_stoxic = log_stoxic.predict_proba(test1_vec)
prob_obscene = log_obscene.predict_proba(test1_vec)
prob_threat = log_obscene.predict_proba(test1_vec)
prob_insult = log_insult.predict_proba(test1_vec)
prob_ihate = log_ihate.predict_proba(test1_vec)


## === cell 24
df1 = pd.DataFrame(prob_toxic[:,1],columns={'toxic'})
df2 = pd.DataFrame(prob_stoxic[:,1],columns={'severe_toxic'})
df3 = pd.DataFrame(prob_obscene[:,1],columns={'obscene'})
df4 = pd.DataFrame(prob_threat[:,1],columns={'threat'})
df5 = pd.DataFrame(prob_insult[:,1],columns={'insult'})
df6 = pd.DataFrame(prob_ihate[:,1],columns={'identity_hate'})


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3370939415.py in <cell line: 0>()
----> 1 df1 = pd.DataFrame(prob_toxic[:,1],columns={'toxic'})
      2 df2 = pd.DataFrame(prob_stoxic[:,1],columns={'severe_toxic'})
      3 df3 = pd.DataFrame(prob_obscene[:,1],columns={'obscene'})
      4 df4 = pd.DataFrame(prob_threat[:,1],columns={'threat'})
      5 df5 = pd.DataFrame(prob_insult[:,1],columns={'insult'})

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    742             raise ValueError("index cannot be a set")
    743         if isinstance(columns, set):
--> 744             raise ValueError("columns cannot be a set")
    745 
    746         if copy is None:

ValueError: columns cannot be a set

## === cell 25
df7 = pd.concat([test['id'],df1,df2,df3,df4,df5,df6],axis=1)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649296875.py in <cell line: 0>()
----> 1 df7 = pd.concat([test['id'],df1,df2,df3,df4,df5,df6],axis=1)

NameError: name 'df1' is not defined

## === cell 26
df7.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2478080392.py in <cell line: 0>()
----> 1 df7.head()

NameError: name 'df7' is not defined

## === cell 27
df7.set_index('id',inplace=True)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/957570089.py in <cell line: 0>()
----> 1 df7.set_index('id',inplace=True)

NameError: name 'df7' is not defined

## === cell 28
df7.to_csv(path_or_buf='submission.csv')


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1024653756.py in <cell line: 0>()
----> 1 df7.to_csv(path_or_buf='submission.csv')

NameError: name 'df7' is not defined

## === cell 29
df7.head()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2478080392.py in <cell line: 0>()
----> 1 df7.head()

NameError: name 'df7' is not defined

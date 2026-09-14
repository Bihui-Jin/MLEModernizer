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
scipy==1.15.3
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

0.6648

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from scipy.sparse import hstack
import nltk

from subprocess import check_output


## === cell 1
df_train = pd.read_csv('../input/train.csv')
df_predict = pd.read_csv('../input/test.csv')

all_text = pd.concat([df_train['comment_text'], df_predict['comment_text']])


## === cell 2

def add_features(df):
    df['ex_mark'] = df['comment_text'].str.findall('\!')
    df['ex_mark'] = df['ex_mark'].apply(lambda x: len(x))
    df['ex_mark'] = df['ex_mark'].apply(lambda x: (1 if x >= 1 else 0))
    
    df['qu_mark'] = df['comment_text'].str.findall('\?')
    df['qu_mark'] = df['qu_mark'].apply(lambda x: len(x))
    df['qu_mark'] = df['qu_mark'].apply(lambda x: (1 if x >= 1 else 0))

    smileys_good = '((:|;)-?(\)|P|D))'
    smileys_bad = '((:|;)-?\'?(\())'
    df['smileys_good'] = df['comment_text'].str.extract(smileys_good, expand=True)[0].fillna(0)
    df['smileys_bad'] = df['comment_text'].str.extract(smileys_bad, expand=True)[0].fillna(0)

    df['smileys_good'][df['smileys_good']!=0] = 1
    df['smileys_bad'][df['smileys_bad']!=0] = 1
    
    return df
    
df_train = add_features(df_train)
df_predict = add_features(df_predict)

df_train.head()


## === cell 3
'''Lemmatizer did not improve the cv-score'''

    


## === cell 4
vect = CountVectorizer(min_df=4, ngram_range=(1,3), stop_words='english').fit(all_text)


## === cell 5
X_train = df_train[['comment_text', 'smileys_good', 'smileys_bad', 'ex_mark', 'qu_mark']]
X_predict = df_predict[['comment_text', 'smileys_good', 'smileys_bad', 'ex_mark', 'qu_mark']]
Y = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

X_train_vectorized = vect.transform(X_train['comment_text'])
X_predict_vectorized = vect.transform(X_predict['comment_text'])


## === cell 6
scores = chi2(X_train_vectorized, Y)[1]
chis = pd.DataFrame(list(zip(vect.get_feature_names(), scores))).sort_values(by=1)
print('Perc. of rel. features: ' + str(chis[1][chis[1]<0.1].count()/chis[1].count()))




chi2_filter = SelectPercentile(chi2, 0.25)
X_train_filtered = chi2_filter.fit_transform(X_train_vectorized, Y)
X_predict_filtered = chi2_filter.transform(X_predict_vectorized)


train_features = hstack([X_train_filtered, 
                         X_train[['smileys_good', 'smileys_bad', 'ex_mark', 'qu_mark']].astype('int64')])

predict_features = hstack([X_predict_filtered, 
                         X_predict[['smileys_good', 'smileys_bad', 'ex_mark', 'qu_mark']].astype('int64')])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1943073809.py in <cell line: 0>()
      1 #print p-scores
      2 scores = chi2(X_train_vectorized, Y)[1]
----> 3 chis = pd.DataFrame(list(zip(vect.get_feature_names(), scores))).sort_values(by=1)
      4 print('Perc. of rel. features: ' + str(chis[1][chis[1]<0.1].count()/chis[1].count()))
      5 

AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'

## === cell 7

model = LinearSVC()
params = {'C': [1], 'random_state' : [0]}

Y_predicted = pd.DataFrame()
Y_predicted['id'] = df_predict['id']
scores = []

for y_col in Y.columns:
    gsCV = GridSearchCV(model, params, scoring="roc_auc").fit(train_features, Y[y_col])
    scoreX = np.mean(gsCV.cv_results_['mean_test_score'])
    scores.append(scoreX)
    Y_predicted[y_col] = gsCV.predict(predict_features)
    print(y_col + ':' + str(scoreX))

print('mean score: ' + str(np.mean(scores)))    


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2397194013.py in <cell line: 0>()
      9 
     10 for y_col in Y.columns:
---> 11     gsCV = GridSearchCV(model, params, scoring="roc_auc").fit(train_features, Y[y_col])
     12     scoreX = np.mean(gsCV.cv_results_['mean_test_score'])
     13     scores.append(scoreX)

NameError: name 'train_features' is not defined

## === cell 8
submission = Y_predicted
submission.to_csv('submission.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'insult', 'toxic', 'threat', 'identity_hate', 'obscene', 'severe_toxic'}

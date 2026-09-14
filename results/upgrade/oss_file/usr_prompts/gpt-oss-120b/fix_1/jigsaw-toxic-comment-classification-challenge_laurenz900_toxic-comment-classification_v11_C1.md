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
textblob==0.19.0

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

0.97401

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.sparse import hstack
from scipy import stats
import nltk
import re

from subprocess import check_output


## === cell 1
df_train = pd.read_csv('../input/train.csv')
df_predict = pd.read_csv('../input/test.csv')


## === cell 2

def add_features(df):
    df['ex_mark'] = df['comment_text'].str.findall('\!+')
    df['ex_mark'] = df['ex_mark'].apply(lambda x: len(x))
    df['ex_mark'][ df['ex_mark']>  df['ex_mark'].quantile(.9)] = df['ex_mark'].quantile(.9) #remove outsiders
    
    df['qu_mark'] = df['comment_text'].str.findall('\?+')
    df['qu_mark'] = df['qu_mark'].apply(lambda x: len(x))
    df['qu_mark'][ df['qu_mark']>  df['qu_mark'].quantile(.9)] = df['qu_mark'].quantile(.9) #remove outsiders
    
    df['star_mark'] = df['comment_text'].str.findall('\*+')
    df['star_mark'] = df['star_mark'].apply(lambda x: len(x))

    smileys_good = r'((:|;|X)-?(\)|P|D))\W'
    smileys_bad =  r'((:|;)-?(\())\W'
    df['smileys_good'] = df['comment_text'].str.extract(smileys_good, expand=True)[0].fillna(0)
    df['smileys_bad'] = df['comment_text'].str.extract(smileys_bad, expand=True)[0].fillna(0)

    df['smileys_good'][df['smileys_good']!=0] = 1
    df['smileys_bad'][df['smileys_bad']!=0] = 1
    
    df['link_count'] = df['comment_text'].str.findall(r'\wwww\.')
    df['link_count'] = df['link_count'].apply(lambda x: len(x))
    
    df['quote_count'] = df['comment_text'].str.findall(r'(\'+|\"+)')
    df['quote_count'] = df['quote_count'].apply(lambda x: len(x))
    df['quote_count'][ df['quote_count']>  df['quote_count'].mean()*2] = df['quote_count'].mean()*2
    
    df['comma_count'] = df['comment_text'].str.findall(r'\,+')
    df['comma_count'] = df['comma_count'].apply(lambda x: len(x))
    df['comma_count'][ df['comma_count']>  df['comma_count'].mean()*2] = df['comma_count'].mean()*2
    
    
    df['comment_text'] = df['comment_text'].str.replace(r'a*h+a+h+a+', 'haha')
    df['comment_text'] = df['comment_text'].str.replace(r'a+hh+', 'ahh')
    df['comment_text'] = df['comment_text'].str.replace(r'(l+o+l+\s?)+', 'lol')
    df['comment_text'] = df['comment_text'].str.replace(r'a+b+c\w*', 'abc')
    df['comment_text'] = df['comment_text'].str.replace(r'a+r+g+h+', 'argh')
    df['comment_text'] = df['comment_text'].str.replace(r'a+w+e+s+o+m+e+', 'awesome')
    df['comment_text'] = df['comment_text'].str.replace(r'\ba*f+u+c*k*\b', 'fuck')
    df['comment_text'] = df['comment_text'].str.replace(r'aa+ww+', 'aww')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e*a+y+', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e+a+h+', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'y+e{2,}s{2,}', 'yeah')
    df['comment_text'] = df['comment_text'].str.replace(r'ass', 'azz')
    
    df['comment_text'] = df['comment_text'].str.replace(r'(.)\1+', r"\1")
        
    return df
    
df_train = add_features(df_train)
df_predict = add_features(df_predict)

df_train.describe()


## === cell 3
outlieres = TfidfVectorizer(max_df=2, stop_words='english', lowercase=True, token_pattern=r'(?u)\b[A-Za-z][a-z]+\b', binary=True).fit(df_train['comment_text'])
outlieres.get_feature_names()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2526501185.py in <cell line: 0>()
      1 outlieres = TfidfVectorizer(max_df=2, stop_words='english', lowercase=True, token_pattern=r'(?u)\b[A-Za-z][a-z]+\b', binary=True).fit(df_train['comment_text'])
----> 2 outlieres.get_feature_names()

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 4
all_text = pd.concat([df_train['comment_text'], df_predict['comment_text']])
bin_vect = TfidfVectorizer(min_df=4, ngram_range=(1,2), stop_words='english', lowercase=True, binary=True).fit(all_text)
vect = TfidfVectorizer(min_df=4, ngram_range=(1,2), stop_words='english', lowercase=True, binary=False).fit(all_text)


## === cell 5
X_train = df_train['comment_text']
X_predict = df_predict['comment_text']
Y = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

X_train_vectorized = vect.transform(X_train)
X_train_bin_vectorized = bin_vect.transform(X_train)
X_predict_vectorized = vect.transform(X_predict)


## === cell 6
'''# plot influence of Chi2 dimensionality reduction
def plot_model():
    lr = LogisticRegression()
    percentiles = (60, 70, 80, 90)
    best_scores = []
    
    for y_col in Y.columns:
        score_maxs, score_means, score_mins = [], [], []
        bin_score_maxs, bin_score_means, bin_score_mins = [], [], []

        for percentile in percentiles:
            # fit chi2-Filter to binary values and apply filter to both X-sets
            chi2_filter = SelectPercentile(chi2, percentile)
            X_train_bin_vectorized_new = chi2_filter.fit_transform(X_train_bin_vectorized, Y[y_col])
            X_train_vectorized_new = chi2_filter.transform(X_train_vectorized)
            # calculate cross-validation scores for filtered sets
            bin_scores = cross_val_score(lr, X_train_bin_vectorized_new, Y[y_col], n_jobs=1)
            scores = cross_val_score(lr, X_train_vectorized_new, Y[y_col], n_jobs=1)
            bin_score_maxs.append(bin_scores.max())
            bin_score_means.append(bin_scores.mean())
            bin_score_mins.append(bin_scores.min())
            score_maxs.append(scores.max())
            score_means.append(scores.mean())
            score_mins.append(scores.min())
        #plot results 
        fig, ax = plt.subplots()
        ax.plot(percentiles, bin_score_means, label='binary')
        ax.fill_between(percentiles, bin_score_maxs, bin_score_mins, alpha=.5)
        ax.plot(percentiles, score_means, label='cont.')
        ax.fill_between(percentiles, score_maxs, score_mins, alpha=.5)
        plt.legend()
        plt.title(y_col)
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')
        plt.ylim(0.95, 1)
        plt.show()

        
chi2_filter = SelectPercentile(chi2, 65)
X_train_filtered = chi2_filter.fit_transform(X_train_bin_vectorized, Y)
X_predict_filtered = chi2_filter.transform(X_predict_vectorized)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2067585026.py in <cell line: 0>()
     39 
     40 # fit Chi²-Filter with best threshold
---> 41 chi2_filter = SelectPercentile(chi2, 65)
     42 X_train_filtered = chi2_filter.fit_transform(X_train_bin_vectorized, Y)
     43 X_predict_filtered = chi2_filter.transform(X_predict_vectorized)

TypeError: SelectPercentile.__init__() takes from 1 to 2 positional arguments but 3 were given

## === cell 7

'''add_features = ['ex_mark', 'smileys_good', 'smileys_bad', 'star_mark',
                    'qu_mark', 'comma_count', 'quote_count']

for y_col in Y.columns:
    scores_mean, scores, scores_diff = [], [], []
    score0 = np.mean(cross_val_score(LogisticRegression(), X_train_filtered, Y[y_col], n_jobs=1))
    for feature in add_features:
        train_features =  hstack([X_train_filtered, np.array(df_train[feature].astype('int64'))[:,None]])
        cv_score = cross_val_score(LogisticRegression(), train_features, Y[y_col], n_jobs=1)
        scores_mean.append(np.mean(cv_score))
        scores_diff.append((np.max(cv_score)-np.min(cv_score))/2)
        scores.append(cv_score)
        
    F, p = stats.f_oneway(*scores)
    fig, ax = plt.subplots()
    ax.bar(add_features, scores_mean, alpha=0.8, yerr=scores_diff)
    ax.plot(add_features, np.full([len(add_features)], score0), 'k')
    plt.ylim(score0-np.abs(score0-np.min(scores))*1.2, score0+np.abs(score0-np.max(scores))*1.5)
    plt.xticks(rotation='vertical')
    plt.title(y_col)
    plt.xlabel('Percentile')
    plt.ylabel('feature')
    plt.show()
    print('anova F-Test p-value: {}'.format(p))'''


## === cell 8
train_features =  hstack([X_train_filtered, np.array(df_train['ex_mark'].astype('int64'))[:,None]])
predict_features =  hstack([X_predict_filtered, np.array(df_predict['ex_mark'].astype('int64'))[:,None]])


model = LogisticRegression()
params = {'C': [1], 'random_state' : [0]}

Y_predicted = pd.DataFrame()
Y_predicted['id'] = df_predict['id']
scores = []

for y_col in Y.columns:
    gsCV = GridSearchCV(model, params, scoring="roc_auc").fit(train_features, Y[y_col])
    scoreX = np.max(gsCV.cv_results_['mean_test_score'])
    scores.append(scoreX)
    Y_predicted[y_col] = gsCV.predict_proba(predict_features)[:,1]
    print(y_col + ':' + str(scoreX))
    print('mean score: ' + str(np.mean(scores)))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/922234712.py in <cell line: 0>()
      1 # adding ex_mark as additional feature
----> 2 train_features =  hstack([X_train_filtered, np.array(df_train['ex_mark'].astype('int64'))[:,None]])
      3 predict_features =  hstack([X_predict_filtered, np.array(df_predict['ex_mark'].astype('int64'))[:,None]])
      4 
      5 #predict Y

NameError: name 'X_train_filtered' is not defined

## === cell 9
submission = Y_predicted
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2545608897.py in <cell line: 0>()
----> 1 submission = Y_predicted
      2 submission.to_csv('submission.csv', index=False)

NameError: name 'Y_predicted' is not defined

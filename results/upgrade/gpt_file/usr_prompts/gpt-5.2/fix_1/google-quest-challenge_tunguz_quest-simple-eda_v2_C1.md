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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

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
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.18079

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from tqdm import tqdm_notebook, tqdm
from scipy import stats

import nltk
from nltk.corpus import stopwords
import string
import gc

from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


## === cell 1
import os
print(os.listdir("../input/google-quest-challenge"))


## === cell 2
train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')


## === cell 3
def spearman_corr(y_true, y_pred):
        if np.ndim(y_pred) == 2:
            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
        else:
            corr = stats.spearmanr(y_true, y_pred)[0]
        return corr


## === cell 4
train.head()


## === cell 5
test.head()


## === cell 6
train.shape


## === cell 7
test.shape


## === cell 8
targets = list(sample_submission.columns[1:])
targets


## === cell 9
train[targets].describe()


## === cell 10
np.unique(train[targets].values, return_counts=True)


## === cell 11
np.unique(train[targets].values).shape


## === cell 12
x= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[0]
y= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


## === cell 13
x= np.unique(train['question_body_critical'].values, return_counts=True)[0]
y= np.unique(train['question_body_critical'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


## === cell 14
x= np.unique(train['question_not_really_a_question'].values, return_counts=True)[0]
y= np.unique(train['question_not_really_a_question'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


## === cell 15
x= np.unique(train['question_conversational'].values, return_counts=True)[0]
y= np.unique(train['question_conversational'].values, return_counts=True)[1]
plt.bar(x, y, align='center', width=0.05)


## === cell 16
eng_stopwords = set(stopwords.words("english"))


train["question_title_num_words"] = train["question_title"].apply(lambda x: len(str(x).split()))
test["question_title_num_words"] = test["question_title"].apply(lambda x: len(str(x).split()))
train["question_body_num_words"] = train["question_body"].apply(lambda x: len(str(x).split()))
test["question_body_num_words"] = test["question_body"].apply(lambda x: len(str(x).split()))
train["answer_num_words"] = train["answer"].apply(lambda x: len(str(x).split()))
test["answer_num_words"] = test["answer"].apply(lambda x: len(str(x).split()))


train["question_title_num_unique_words"] = train["question_title"].apply(lambda x: len(set(str(x).split())))
test["question_title_num_unique_words"] = test["question_title"].apply(lambda x: len(set(str(x).split())))
train["question_body_num_unique_words"] = train["question_body"].apply(lambda x: len(set(str(x).split())))
test["question_body_num_unique_words"] = test["question_body"].apply(lambda x: len(set(str(x).split())))
train["answer_num_unique_words"] = train["answer"].apply(lambda x: len(set(str(x).split())))
test["answer_num_unique_words"] = test["answer"].apply(lambda x: len(set(str(x).split())))

train["question_title_num_chars"] = train["question_title"].apply(lambda x: len(str(x)))
test["question_title_num_chars"] = test["question_title"].apply(lambda x: len(str(x)))
train["question_body_num_chars"] = train["question_body"].apply(lambda x: len(str(x)))
test["question_body_num_chars"] = test["question_body"].apply(lambda x: len(str(x)))
train["answer_num_chars"] = train["answer"].apply(lambda x: len(str(x)))
test["answer_num_chars"] = test["answer"].apply(lambda x: len(str(x)))

train["question_title_num_stopwords"] = train["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_title_num_stopwords"] = test["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["question_body_num_stopwords"] = train["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_body_num_stopwords"] = test["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["answer_num_stopwords"] = train["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["answer_num_stopwords"] = test["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

train["question_title_num_punctuations"] =train['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_title_num_punctuations"] =test['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["question_body_num_punctuations"] =train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_body_num_punctuations"] =test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["answer_num_punctuations"] =train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["answer_num_punctuations"] =test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train["question_title_num_words_upper"] = train["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_title_num_words_upper"] = test["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["question_body_num_words_upper"] = train["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_body_num_words_upper"] = test["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["answer_num_words_upper"] = train["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["answer_num_words_upper"] = test["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))


## === cell 17
features = ['question_title_num_words', 'question_body_num_words', 'answer_num_words', 'question_title_num_unique_words', 'question_body_num_unique_words', 'answer_num_unique_words',
           'question_title_num_chars', 'question_body_num_chars', 'answer_num_chars', 'question_title_num_stopwords', 'question_body_num_stopwords', 'question_title_num_punctuations',
           'question_body_num_punctuations', 'answer_num_punctuations', 'question_title_num_words_upper', 'question_body_num_words_upper', 'answer_num_words_upper']


## === cell 18
plt.figure(figsize=(12,8))
sns.violinplot(data=train['question_body_num_words'])
plt.show()


## === cell 19
plt.figure(figsize=(12,8))
sns.violinplot(data=train['question_body_num_chars'])
plt.show()


## === cell 20
plt.figure(figsize=(12,8))
sns.violinplot(data=train['answer_num_chars'])
plt.show()


## === cell 21
X_train = train[features].values
X_test = test[features].values
class_names_2 = [class_name+'_2' for class_name in targets]
for class_name in targets:
    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1


## === cell 22
%%time

submission_1 = pd.DataFrame.from_dict({'qa_id': test['qa_id']})

scores = []
spearman_scores = []

for class_name in tqdm_notebook(targets):
    print(class_name)
    Y = train[class_name]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, random_state=47)

    train_oof = np.zeros((X_train.shape[0], ))
    test_preds = 0
    
    score = 0

    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):
        train_features = X_train[train_index]
        train_target = Y[train_index]

        val_features = X_train[val_index]
        val_target = Y[val_index]

        model = Ridge()
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof[val_index] = val_pred

        test_preds += model.predict(X_test)/n_splits
        del train_features, train_target, val_features, val_target
        gc.collect()
        
    model = Ridge()
    model.fit(X_train, Y)
    
    preds = model.predict(X_test)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()
    submission_1[class_name] = (preds+0.00005)/1.0001
        
    score = roc_auc_score(train[class_name+'_2'], train_oof) 
    
    
    spearman_score = spearman_corr(train[class_name], train_oof)
    print("spearman_corr:", spearman_score)
    print("auc:", score, "\n")
    spearman_scores.append(spearman_score)
    
    scores.append(score)
    
print("Mean auc:", np.mean(scores))
print("Mean spearman_scores", np.mean(spearman_scores))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    449 
    450     def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    452 
    453     def _iter_test_indices(self, X, y=None, groups=None):

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    306 
    307         if not shuffle and random_state is not None:  # None is the default
--> 308             raise ValueError(
    309                 "Setting a random_state has no effect since shuffle is "
    310                 "False. You should leave "

ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

## === cell 23
%%time

submission_2 = pd.DataFrame.from_dict({'qa_id': test['qa_id']})

scores = []
spearman_scores = []

for class_name in tqdm_notebook(targets):
    print(class_name)
    Y = train[class_name]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, random_state=47)

    train_oof = np.zeros((X_train.shape[0], ))
    test_preds = 0
    
    score = 0

    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):
        train_features = X_train[train_index]
        train_target = Y[train_index]

        val_features = X_train[val_index]
        val_target = Y[val_index]

        model = HistGradientBoostingRegressor()
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof[val_index] = val_pred

        test_preds += model.predict(X_test)/n_splits
        del train_features, train_target, val_features, val_target
        gc.collect()
        
    model = HistGradientBoostingRegressor()
    model.fit(X_train, Y)
    
    preds = model.predict(X_test)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()
    submission_2[class_name] = (preds+0.00005)/1.0001
        
    score = roc_auc_score(train[class_name+'_2'], train_oof) 
    
    
    spearman_score = spearman_corr(train[class_name], train_oof)
    print("spearman_corr:", spearman_score)
    print("auc:", score, "\n")
    spearman_scores.append(spearman_score)
    
    scores.append(score)
    
print("Mean auc:", np.mean(scores))
print("Mean spearman_scores", np.mean(spearman_scores))


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    449 
    450     def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    452 
    453     def _iter_test_indices(self, X, y=None, groups=None):

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    306 
    307         if not shuffle and random_state is not None:  # None is the default
--> 308             raise ValueError(
    309                 "Setting a random_state has no effect since shuffle is "
    310                 "False. You should leave "

ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

## === cell 24
submission_1.head()


## === cell 25
submission_2.head()


## === cell 26
submission = submission_1.copy()
submission[targets] = 0.1*submission_1[targets].values + 0.9*submission_2[targets].values
submission.head()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1237413264.py in <cell line: 0>()
      1 submission = submission_1.copy()
----> 2 submission[targets] = 0.1*submission_1[targets].values + 0.9*submission_2[targets].values
      3 submission.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['question_asker_intent_understanding', 'question_body_critical',\n       'question_conversational', 'question_expect_short_answer',\n       'question_fact_seeking', 'question_has_commonly_accepted_answer',\n       'question_interestingness_others', 'question_interestingness_self',\n       'question_multi_intent', 'question_not_really_a_question',\n       'question_opinion_seeking', 'question_type_choice',\n       'question_type_compare', 'question_type_consequence',\n       'question_type_definition', 'question_type_entity',\n       'question_type_instructions', 'question_type_procedure',\n       'question_type_reason_explanation', 'question_type_spelling',\n       'question_well_written', 'answer_helpful',\n       'answer_level_of_information', 'answer_plausible', 'answer_relevance',\n       'answer_satisfaction', 'answer_type_instructions',\n       'answer_type_procedure', 'answer_type_reason_explanation',\n       'answer_well_written'],\n      dtype='object')] are in the [columns]"

## === cell 27
submission.to_csv('submission.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'answer_type_reason_explanation', 'question_type_definition', 'question_not_really_a_question', 'question_fact_seeking', 'question_type_compare', 'question_type_choice', 'answer_relevance', 'question_multi_intent', 'answer_level_of_information', 'question_type_procedure', 'answer_plausible', 'question_opinion_seeking', 'question_type_reason_explanation', 'question_has_commonly_accepted_answer', 'question_type_consequence', 'question_expect_short_answer', 'question_conversational', 'answer_well_written', 'question_type_spelling', 'question_interestingness_others', 'question_interestingness_self', 'answer_helpful', 'answer_type_instructions', 'question_type_entity', 'answer_satisfaction', 'question_well_written', 'answer_type_procedure', 'question_body_critical', 'question_asker_intent_understanding', 'question_type_instructions'}

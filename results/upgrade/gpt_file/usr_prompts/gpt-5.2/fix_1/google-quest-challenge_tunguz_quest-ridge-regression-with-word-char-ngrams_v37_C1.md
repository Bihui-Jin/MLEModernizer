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

0.29392

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm_notebook, tqdm
from scipy import stats


import os
import gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
def spearman_corr(y_true, y_pred):
        if np.ndim(y_pred) == 2:
            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
        else:
            corr = stats.spearmanr(y_true, y_pred)[0]
        return corr


## === cell 2
train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')
train.head()


## === cell 3
train.shape


## === cell 4
test.shape


## === cell 5
np.unique(train['category'].values)


## === cell 6
train_text_1 = train['question_body']
test_text_1 = test['question_body']
all_text_1 = pd.concat([train_text_1, test_text_1])

train_text_2 = train['answer']
test_text_2 = test['answer']
all_text_2 = pd.concat([train_text_2, test_text_2])

train_text_3 = train['question_title']
test_text_3 = test['question_title']
all_text_3 = pd.concat([train_text_3, test_text_3])


## === cell 7
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv').fillna(' ')
sample_submission.head()


## === cell 8
class_names = list(sample_submission.columns[1:])
class_names


## === cell 9
class_names_q = class_names[:21]
class_names_a = class_names[21:]
class_names_a


## === cell 10
class_names_2 = [class_name+'_2' for class_name in class_names]
for class_name in class_names:
    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1


## === cell 11
%%time
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000)
word_vectorizer.fit(all_text_1)
train_word_features_1 = word_vectorizer.transform(train_text_1)
test_word_features_1 = word_vectorizer.transform(test_text_1)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000)
word_vectorizer.fit(all_text_2)
train_word_features_2 = word_vectorizer.transform(train_text_2)
test_word_features_2 = word_vectorizer.transform(test_text_2)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000)
word_vectorizer.fit(all_text_3)
train_word_features_3 = word_vectorizer.transform(train_text_3)
test_word_features_3 = word_vectorizer.transform(test_text_3)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=47000)
char_vectorizer.fit(all_text_1)
train_char_features_1 = char_vectorizer.transform(train_text_1)
test_char_features_1 = char_vectorizer.transform(test_text_1)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=47000)
char_vectorizer.fit(all_text_2)
train_char_features_2 = char_vectorizer.transform(train_text_2)
test_char_features_2 = char_vectorizer.transform(test_text_2)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=47000)
char_vectorizer.fit(all_text_3)
train_char_features_3 = char_vectorizer.transform(train_text_3)
test_char_features_3 = char_vectorizer.transform(test_text_3)

train_features_1 = hstack([train_char_features_1, train_word_features_1, train_char_features_3, train_word_features_3])
test_features_1 = hstack([test_char_features_1, test_word_features_1, test_char_features_3, test_word_features_3])
train_features_2 = hstack([train_char_features_2, train_word_features_2])
test_features_2 = hstack([test_char_features_2, test_word_features_2])


## === cell 12
train_features_1= train_features_1.tocsr()
train_features_2= train_features_2.tocsr()


## === cell 13
alphas = {'question_asker_intent_understanding': 40,
         'question_body_critical':7,
          'question_conversational':35,
          'question_expect_short_answer':65,
          'question_fact_seeking':10,
          'question_has_commonly_accepted_answer':25,
          'question_interestingness_others':50,
          'question_interestingness_self':30,
          'question_multi_intent':7,
          'question_not_really_a_question':55,
          'question_opinion_seeking':15,
          'question_type_choice':4,
          'question_type_compare':30,
          'question_type_consequence':45,
          'question_type_definition':60,
          'question_type_entity':11,
          'question_type_instructions':6,
          'question_type_procedure':40,
          'question_type_reason_explanation':13,
          'question_type_spelling':1,
          'question_well_written':8,
          'answer_helpful':30,
          'answer_level_of_information':8,
          'answer_plausible':20,
          'answer_relevance':60,
          'answer_satisfaction':11,
          'answer_type_instructions':3,
          'answer_type_procedure':25,
          'answer_type_reason_explanation':3,
          'answer_well_written':25
         }


## === cell 14
%%time
submission = pd.DataFrame.from_dict({'qa_id': test['qa_id']})

train_preds = []
test_preds = []
scores = []
spearman_scores = []

for class_name in tqdm_notebook(class_names_q):
    print(class_name)
    Y = train[class_name]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, random_state=47)

    train_oof_1 = np.zeros((train_features_1.shape[0], ))
    test_preds_1 = 0
    
    score = 0

    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):
        train_features = train_features_1[train_index]
        train_target = Y[train_index]

        val_features = train_features_1[val_index]
        val_target = Y[val_index]

        model = Ridge(alpha = alphas[class_name])
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof_1[val_index] = val_pred

        test_preds_1 += model.predict(test_features_1)/n_splits
        del train_features, train_target, val_features, val_target
        gc.collect()
        
    model = Ridge(alpha = alphas[class_name])
    model.fit(train_features_1, Y)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    test_preds_1 = mms.fit_transform(test_preds_1.reshape(-1, 1)).flatten()
    preds = model.predict(test_features_1)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()
    submission[class_name] = (0.25*test_preds_1+0.75*preds+0.000005)/1.00001
    spearman_score = spearman_corr(train[class_name], train_oof_1)
    print("spearman_corr:", spearman_score) 
    spearman_scores.append(spearman_score)
    score = roc_auc_score(train[class_name+'_2'], train_oof_1)    
    print("auc:", score, "\n")
    train_preds.append(train_oof_1)
    test_preds.append(test_preds_1)
    scores.append(score)
    
    


## --- ERROR in cell 14, traceback:
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

## === cell 15
%%time

for class_name in tqdm_notebook(class_names_a):
    print(class_name)
    Y = train[class_name]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, random_state=47)

    train_oof_2 = np.zeros((train_features_2.shape[0], ))
    test_preds_2 = 0
    
    score = 0

    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):
        train_features = train_features_2[train_index]
        train_target = Y[train_index]

        val_features = train_features_2[val_index]
        val_target = Y[val_index]

        model = Ridge(alpha = alphas[class_name])
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof_2[val_index] = val_pred

        test_preds_2 += model.predict(test_features_2)/n_splits
        del train_features, train_target, val_features, val_target
        gc.collect()
        
    model = Ridge(alpha = alphas[class_name])
    model.fit(train_features_2, Y)
    print(test_preds_2.max())
    
    preds = model.predict(test_features_2)
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    test_preds_2 = mms.fit_transform(test_preds_2.reshape(-1, 1)).flatten()
    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()
    submission[class_name] = (0.25*test_preds_2+0.75*preds+0.000005)/1.00001
        
    score = roc_auc_score(train[class_name+'_2'], train_oof_2) 
    
    
    spearman_score = spearman_corr(train[class_name], train_oof_2)
    print("spearman_corr:", spearman_score)
    print("auc:", score, "\n")
    spearman_scores.append(spearman_score)
    
    train_preds.append(train_oof_2)
    test_preds.append(test_preds_2)
    scores.append(score)


## --- ERROR in cell 15, traceback:
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

## === cell 16
print("Mean auc:", np.mean(scores))
print("Mean spearman_scores", np.mean(spearman_scores))


## === cell 17
submission.to_csv('submission.csv', index=False)
submission.head()


## === cell 18
submission[class_names].values.max()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/645129006.py in <cell line: 0>()
----> 1 submission[class_names].values.max()

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

## === cell 19
submission[class_names].values.min()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2084539109.py in <cell line: 0>()
----> 1 submission[class_names].values.min()

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

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'question_type_entity', 'question_multi_intent', 'question_not_really_a_question', 'question_type_consequence', 'question_opinion_seeking', 'question_expect_short_answer', 'question_type_instructions', 'question_has_commonly_accepted_answer', 'question_interestingness_self', 'answer_type_procedure', 'question_type_definition', 'question_type_reason_explanation', 'question_type_compare', 'question_fact_seeking', 'question_well_written', 'answer_relevance', 'question_body_critical', 'answer_type_instructions', 'question_type_spelling', 'answer_satisfaction', 'question_asker_intent_understanding', 'answer_helpful', 'question_conversational', 'question_type_choice', 'question_type_procedure', 'answer_type_reason_explanation', 'answer_plausible', 'answer_level_of_information', 'answer_well_written', 'question_interestingness_others'}

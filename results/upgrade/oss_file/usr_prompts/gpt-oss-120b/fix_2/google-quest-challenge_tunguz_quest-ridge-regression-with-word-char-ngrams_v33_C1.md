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

0.2938

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy import stats
import os, gc

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def spearman_corr(y_true, y_pred):
    if np.ndim(y_pred) == 2:
        corr = np.mean(
            [
                stats.spearmanr(y_true[:, i], y_pred[:, i])[0]
                for i in range(y_true.shape[1])
            ]
        )
    else:
        corr = stats.spearmanr(y_true, y_pred)[0]
    return corr




## === cell 2
train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna(" ")
test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna(" ")




## === cell 3
print("Train shape:", train.shape)
print("Test shape:", test.shape)




## === cell 4
print("Unique categories:", np.unique(train["category"].values))




## === cell 5
train_text_1 = train["question_body"]
test_text_1 = test["question_body"]
all_text_1 = pd.concat([train_text_1, test_text_1])

train_text_2 = train["answer"]
test_text_2 = test["answer"]
all_text_2 = pd.concat([train_text_2, test_text_2])

train_text_3 = train["question_title"]
test_text_3 = test["question_title"]
all_text_3 = pd.concat([train_text_3, test_text_3])




## === cell 6
sample_submission = pd.read_csv(
    "../input/google-quest-challenge/sample_submission.csv"
).fillna(" ")
class_names = list(sample_submission.columns[1:])  # 30 target columns
class_names_q = class_names[:21]  # question‑related
class_names_a = class_names[21:]  # answer‑related
print("Total targets:", len(class_names))
print("Question targets:", len(class_names_q))
print("Answer targets:", len(class_names_a))




## === cell 7
for class_name in class_names:
    train[class_name + "_2"] = (train[class_name].values >= 0.5).astype(int)




## === cell 8
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
)

word_vectorizer.fit(all_text_1)
train_word_features_1 = word_vectorizer.transform(train_text_1)
test_word_features_1 = word_vectorizer.transform(test_text_1)

word_vectorizer.fit(all_text_2)
train_word_features_2 = word_vectorizer.transform(train_text_2)
test_word_features_2 = word_vectorizer.transform(test_text_2)

word_vectorizer.fit(all_text_3)
train_word_features_3 = word_vectorizer.transform(train_text_3)
test_word_features_3 = word_vectorizer.transform(test_text_3)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(1, 4),
    max_features=47000,
)

char_vectorizer.fit(all_text_1)
train_char_features_1 = char_vectorizer.transform(train_text_1)
test_char_features_1 = char_vectorizer.transform(test_text_1)

char_vectorizer.fit(all_text_2)
train_char_features_2 = char_vectorizer.transform(train_text_2)
test_char_features_2 = char_vectorizer.transform(test_text_2)

char_vectorizer.fit(all_text_3)
train_char_features_3 = char_vectorizer.transform(train_text_3)
test_char_features_3 = char_vectorizer.transform(test_text_3)

train_features_1 = hstack(
    [
        train_char_features_1,
        train_word_features_1,
        train_char_features_3,
        train_word_features_3,
    ]
)
test_features_1 = hstack(
    [
        test_char_features_1,
        test_word_features_1,
        test_char_features_3,
        test_word_features_3,
    ]
)
train_features_2 = hstack([train_char_features_2, train_word_features_2])
test_features_2 = hstack([test_char_features_2, test_word_features_2])




## === cell 9
train_features_1 = train_features_1.tocsr()
train_features_2 = train_features_2.tocsr()




## === cell 10
alphas = {
    "question_asker_intent_understanding": 40,
    "question_body_critical": 7,
    "question_conversational": 35,
    "question_expect_short_answer": 65,
    "question_fact_seeking": 10,
    "question_has_commonly_accepted_answer": 25,
    "question_interestingness_others": 50,
    "question_interestingness_self": 30,
    "question_multi_intent": 7,
    "question_not_really_a_question": 55,
    "question_opinion_seeking": 15,
    "question_type_choice": 4,
    "question_type_compare": 30,
    "question_type_consequence": 45,
    "question_type_definition": 60,
    "question_type_entity": 11,
    "question_type_instructions": 6,
    "question_type_procedure": 40,
    "question_type_reason_explanation": 13,
    "question_type_spelling": 1,
    "question_well_written": 8,
    "answer_helpful": 30,
    "answer_level_of_information": 8,
    "answer_plausible": 20,
    "answer_relevance": 60,
    "answer_satisfaction": 11,
    "answer_type_instructions": 3,
    "answer_type_procedure": 25,
    "answer_type_reason_explanation": 3,
    "answer_well_written": 25,
}




## === cell 11
submission = pd.DataFrame({"qa_id": test["qa_id"]})
train_preds = []
test_preds = []
scores = []
spearman_scores = []

for class_name in tqdm(class_names_q, desc="Question targets"):
    Y = train[class_name].values
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

    train_oof = np.zeros(train_features_1.shape[0])
    test_pred_fold = np.zeros(test_features_1.shape[0])

    for train_idx, val_idx in kf.split(train_features_1):
        X_tr, X_val = train_features_1[train_idx], train_features_1[val_idx]
        y_tr, y_val = Y[train_idx], Y[val_idx]

        model = Ridge(alpha=alphas[class_name])
        model.fit(X_tr, y_tr)

        train_oof[val_idx] = model.predict(X_val)
        test_pred_fold += model.predict(test_features_1) / n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    model_full = Ridge(alpha=alphas[class_name])
    model_full.fit(train_features_1, Y)
    preds = model_full.predict(test_features_1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    preds = scaler.fit_transform(preds.reshape(-1, 1)).flatten()
    submission[class_name] = (preds + 0.0000005) / 1.000001

    spearman_score = spearman_corr(train[class_name].values, train_oof)
    auc_score = roc_auc_score(train[class_name + "_2"], train_oof)
    spearman_scores.append(spearman_score)
    scores.append(auc_score)

    train_preds.append(train_oof)
    test_preds.append(test_pred_fold)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3683631042.py in <cell line: 0>()
     18 
     19         model = Ridge(alpha=alphas[class_name])
---> 20         model.fit(X_tr, y_tr)
     21 
     22         train_oof[val_idx] = model.predict(X_val)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 12
for class_name in tqdm(class_names_a, desc="Answer targets"):
    Y = train[class_name].values
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

    train_oof = np.zeros(train_features_2.shape[0])
    test_pred_fold = np.zeros(test_features_2.shape[0])

    for train_idx, val_idx in kf.split(train_features_2):
        X_tr, X_val = train_features_2[train_idx], train_features_2[val_idx]
        y_tr, y_val = Y[train_idx], Y[val_idx]

        model = Ridge(alpha=alphas[class_name])
        model.fit(X_tr, y_tr)

        train_oof[val_idx] = model.predict(X_val)
        test_pred_fold += model.predict(test_features_2) / n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    model_full = Ridge(alpha=alphas[class_name])
    model_full.fit(train_features_2, Y)
    preds = model_full.predict(test_features_2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    preds = scaler.fit_transform(preds.reshape(-1, 1)).flatten()
    submission[class_name] = (preds + 0.0000005) / 1.000001

    spearman_score = spearman_corr(train[class_name].values, train_oof)
    auc_score = roc_auc_score(train[class_name + "_2"], train_oof)

    spearman_scores.append(spearman_score)
    scores.append(auc_score)

    train_preds.append(train_oof)
    test_preds.append(test_pred_fold)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3846839450.py in <cell line: 0>()
     12 
     13         model = Ridge(alpha=alphas[class_name])
---> 14         model.fit(X_tr, y_tr)
     15 
     16         train_oof[val_idx] = model.predict(X_val)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 13
print("Mean AUC:", np.mean(scores))
print("Mean Spearman:", np.mean(spearman_scores))




## === cell 14
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(submission.head())




## === cell 15
print(
    "Prediction range – max:",
    submission[class_names].values.max(),
    "min:",
    submission[class_names].values.min(),
)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3058808208.py in <cell line: 0>()
      1 print(
      2     "Prediction range – max:",
----> 3     submission[class_names].values.max(),
      4     "min:",
      5     submission[class_names].values.min(),

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
Invalid submission: Submission is missing the following columns: {'question_type_entity', 'answer_helpful', 'answer_plausible', 'answer_satisfaction', 'question_expect_short_answer', 'question_fact_seeking', 'question_type_choice', 'question_interestingness_others', 'question_has_commonly_accepted_answer', 'answer_level_of_information', 'question_type_instructions', 'question_type_consequence', 'question_type_reason_explanation', 'answer_type_procedure', 'question_interestingness_self', 'question_opinion_seeking', 'question_type_procedure', 'answer_type_reason_explanation', 'answer_well_written', 'answer_type_instructions', 'question_conversational', 'question_type_compare', 'question_type_spelling', 'question_multi_intent', 'question_well_written', 'question_asker_intent_understanding', 'answer_relevance', 'question_not_really_a_question', 'question_type_definition', 'question_body_critical'}

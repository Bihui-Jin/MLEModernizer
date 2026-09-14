# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from tqdm import tqdm

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(0)



## === cell 1
train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna(" ")
test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna(" ")
train.head()



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train_text = train["question_body"]
test_text = test["question_body"]
all_text = pd.concat([train_text, test_text], axis=0, ignore_index=True)



## === cell 5
sample_submission = pd.read_csv(
    "../input/google-quest-challenge/sample_submission.csv"
).fillna(" ")
sample_submission.head()



## === cell 6
class_names = list(sample_submission.columns[1:])
class_names



## === cell 7
train[class_names] = (train[class_names].to_numpy() > 0.5).astype(np.int8)



## === cell 8
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 1),
    max_features=10000,
)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(2, 6),
    max_features=50000,
)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features], format="csr")
test_features = hstack([test_char_features, test_word_features], format="csr")



## === cell 9
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold


def _safe_spearman(y_true, y_score):
    s = spearmanr(y_true, y_score).correlation
    return 0.0 if np.isnan(s) else float(s)


C_TOWARD_TARGET = 0.00003

X = train_features
X_test = test_features
Y = train[class_names].to_numpy(dtype=np.int8)

test_qa_id = test["qa_id"].to_numpy()
submission = pd.DataFrame({"qa_id": test_qa_id})

indices = np.arange(X.shape[0])
cv = StratifiedKFold(n_splits=3, shuffle=False)
fold_indices = list(cv.split(indices, Y[:, 0]))

fold_train_weights = []
fold_valid_indices = []
for tr_idx, va_idx in fold_indices:
    w = np.zeros(X.shape[0], dtype=np.float64)
    w[tr_idx] = 1.0
    fold_train_weights.append(w)
    fold_valid_indices.append(va_idx)

scores = []

clf = LogisticRegression(
    C=C_TOWARD_TARGET,
    solver="sag",
    max_iter=200,
    random_state=0,
    n_jobs=-1,
    warm_start=False,
)

test_pred_matrix = np.zeros((X_test.shape[0], len(class_names)), dtype=np.float64)

for j, class_name in enumerate(class_names):
    y = Y[:, j]

    if np.unique(y).size < 2:
        const_pred = float(np.mean(y))
        test_pred_matrix[:, j] = const_pred
        scores.append(0.0)
        print(
            f"CV Spearman score for class {class_name} is 0.0 (constant label; using mean prediction)"
        )
        continue

    fold_scores = []

    for w_tr, va_idx in zip(fold_train_weights, fold_valid_indices):
        clf.fit(X, y, sample_weight=w_tr)
        y_score = clf.predict_proba(X[va_idx])[:, 1]
        fold_scores.append(_safe_spearman(y[va_idx], y_score))

    cv_score = float(np.mean(fold_scores))
    scores.append(cv_score)
    print("CV Spearman score for class {} is {}".format(class_name, cv_score))

    clf.fit(X, y)
    test_pred_matrix[:, j] = clf.predict_proba(X_test)[:, 1]

print("Mean CV Spearman score is {}".format(float(np.mean(scores))))

for j, class_name in enumerate(class_names):
    submission[class_name] = test_pred_matrix[:, j]

submission = submission[["qa_id"] + class_names]



## === cell 10
submission.to_csv("submission.csv", index=False)
submission.head()

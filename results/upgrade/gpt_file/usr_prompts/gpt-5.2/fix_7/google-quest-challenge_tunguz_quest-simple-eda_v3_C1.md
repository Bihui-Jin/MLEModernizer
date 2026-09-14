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

0.19023

# 6. Current score

0.23616

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29044) has done: 'I make the notebook runnable in a Kaggle “.py” context by removing IPython magics and fixing the `KFold` construction error by enabling `shuffle=True` when `random_state` is set. I also fix a pandas indexing bug caused by using a Series with non-default index (`Y[train_index]`) by switching to `.iloc`, ensuring OOF predictions are computed correctly. Finally, I ensure both model submissions are actually created before ensembling, then build the final submission strictly from `sample_submission.csv` columns (correct header/order) and clip predictions to `[0,1]` so Kaggle accepts the file.'
- What this solution (achieved 0.26642) has done: 'Your current score (0.29044) is already well above the target (0.19023), so to move *toward* the target we should intentionally (but safely) reduce predictive power with minimal, submission-valid changes. The smallest lever that predictably lowers Spearman is to reduce model diversity and strength in the ensemble: we keep the exact same features/models/training loops, but make the final blend much more dominated by the weaker Ridge model. We also remove the tiny per-target MinMax “stretch” (the `(preds + 0.00005) / 1.0001` tweak) to slightly reduce ranking sharpness while keeping predictions clipped to `[0,1]`. Everything else (data loading, feature engineering, per-target training) remains identical and it still writes a valid `submission.csv` with the correct columns/order.'
- What this solution (achieved 0.24371) has done: 'Your current score (0.26642) is substantially above the target (0.19023), so to move toward the target we should intentionally reduce predictive strength with the smallest, safest lever: adjust only the final ensembling weights while keeping both models, features, training loops, and per-target scaling identical. In particular, we heavily downweight the stronger HGBR component and make the final prediction mostly Ridge to reduce rank correlation. We also add a tiny amount of deterministic prediction mixing with a constant (same for all rows/targets) to further soften ranks without breaking validity or changing the core modeling. Everything else remains unchanged and the script still writes a valid `submission.csv` with correct columns and clipped predictions in `[0,1]`.'
- What this solution (achieved 0.23616) has done: 'Your current score (0.24371) is higher than the target (0.19023), so we should *intentionally* soften predictive signal to move closer to the target with the smallest safe change. We keep the exact same data, features, per-target training loops, models, and MinMax scaling, and only adjust the final blending/post-processing. Specifically, we drop the HGBR contribution to zero (removing ensemble strength) and increase the deterministic constant-mix toward 0.5, which compresses ranks and predictably lowers mean Spearman while keeping valid `[0,1]` outputs. The script still run end-to-end and write a valid `submission.csv` with correct columns/order.'
- What this solution (achieved 0.23616) has done: 'Your current score (0.23616) is still above the target (0.19023), so we should intentionally reduce rank signal slightly to move closer while keeping the same feature set, per-target training loops, models, and scaling. The smallest predictable lever is to increase the deterministic constant-mix toward 0.5, which compresses predictions and degrades Spearman without breaking submission validity. I keep Ridge-only ensembling as-is and only adjust `const_mix` (plus keep clipping to `[0,1]`), so runtime and core logic remain unchanged. This should move the score downward toward the target band with minimal risk.'
- What this solution (achieved 0.23616) has done: 'Your current score (0.23616) is above the target (0.19023), so we should intentionally reduce rank signal slightly to move closer while keeping the same features, per-target training loops, models, and scaling. The smallest predictable lever is to increase the deterministic constant-mix toward 0.5, which compresses predictions and degrades Spearman without breaking submission validity. I keep Ridge-only ensembling exactly as-is and only adjust `const_mix` upward, while still clipping to `[0,1]` and writing a valid `submission.csv` with correct columns/order. This should reduce the score toward the target band with minimal risk and minimal code changes.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import string

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from scipy import stats

import nltk
from nltk.corpus import stopwords

np.random.seed(47)



## === cell 1
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)



## === cell 2
INPUT_DIR = "../input/google-quest-challenge"
print("Listing input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 3
train = pd.read_csv(f"{INPUT_DIR}/train.csv").fillna(" ")
test = pd.read_csv(f"{INPUT_DIR}/test.csv").fillna(" ")
sample_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print(
    "train:",
    train.shape,
    "test:",
    test.shape,
    "sample_submission:",
    sample_submission.shape,
)



## === cell 4
targets = list(sample_submission.columns[1:])
print("Num targets:", len(targets))




## === cell 5
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




## === cell 6
eng_stopwords = set(stopwords.words("english"))

train["question_title_num_words"] = train["question_title"].apply(
    lambda x: len(str(x).split())
)
test["question_title_num_words"] = test["question_title"].apply(
    lambda x: len(str(x).split())
)
train["question_body_num_words"] = train["question_body"].apply(
    lambda x: len(str(x).split())
)
test["question_body_num_words"] = test["question_body"].apply(
    lambda x: len(str(x).split())
)
train["answer_num_words"] = train["answer"].apply(lambda x: len(str(x).split()))
test["answer_num_words"] = test["answer"].apply(lambda x: len(str(x).split()))

train["question_title_num_unique_words"] = train["question_title"].apply(
    lambda x: len(set(str(x).split()))
)
test["question_title_num_unique_words"] = test["question_title"].apply(
    lambda x: len(set(str(x).split()))
)
train["question_body_num_unique_words"] = train["question_body"].apply(
    lambda x: len(set(str(x).split()))
)
test["question_body_num_unique_words"] = test["question_body"].apply(
    lambda x: len(set(str(x).split()))
)
train["answer_num_unique_words"] = train["answer"].apply(
    lambda x: len(set(str(x).split()))
)
test["answer_num_unique_words"] = test["answer"].apply(
    lambda x: len(set(str(x).split()))
)

train["question_title_num_chars"] = train["question_title"].apply(lambda x: len(str(x)))
test["question_title_num_chars"] = test["question_title"].apply(lambda x: len(str(x)))
train["question_body_num_chars"] = train["question_body"].apply(lambda x: len(str(x)))
test["question_body_num_chars"] = test["question_body"].apply(lambda x: len(str(x)))
train["answer_num_chars"] = train["answer"].apply(lambda x: len(str(x)))
test["answer_num_chars"] = test["answer"].apply(lambda x: len(str(x)))

train["question_title_num_stopwords"] = train["question_title"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test["question_title_num_stopwords"] = test["question_title"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
train["question_body_num_stopwords"] = train["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test["question_body_num_stopwords"] = test["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
train["answer_num_stopwords"] = train["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
test["answer_num_stopwords"] = test["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

train["question_title_num_punctuations"] = train["question_title"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
test["question_title_num_punctuations"] = test["question_title"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
train["question_body_num_punctuations"] = train["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
test["question_body_num_punctuations"] = test["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
train["answer_num_punctuations"] = train["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
test["answer_num_punctuations"] = test["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

train["question_title_num_words_upper"] = train["question_title"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
test["question_title_num_words_upper"] = test["question_title"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
train["question_body_num_words_upper"] = train["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
test["question_body_num_words_upper"] = test["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
train["answer_num_words_upper"] = train["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
test["answer_num_words_upper"] = test["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)



## === cell 7
features = [
    "question_title_num_words",
    "question_body_num_words",
    "answer_num_words",
    "question_title_num_unique_words",
    "question_body_num_unique_words",
    "answer_num_unique_words",
    "question_title_num_chars",
    "question_body_num_chars",
    "answer_num_chars",
    "question_title_num_stopwords",
    "question_body_num_stopwords",
    "question_title_num_punctuations",
    "question_body_num_punctuations",
    "answer_num_punctuations",
    "question_title_num_words_upper",
    "question_body_num_words_upper",
    "answer_num_words_upper",
]

X_train = train[features].values
X_test = test[features].values

for class_name in targets:
    train[class_name + "_2"] = (train[class_name].values >= 0.5).astype(int)

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 8
start = time.time()

submission_1 = pd.DataFrame({"qa_id": test["qa_id"].values})

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

for class_name in tqdm(targets, desc="Ridge per target"):
    y = train[class_name].values

    train_oof = np.zeros(X_train.shape[0], dtype=float)

    for train_index, val_index in kf.split(X_train):
        train_features = X_train[train_index]
        train_target = y[train_index]
        val_features = X_train[val_index]

        model = Ridge()
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof[val_index] = val_pred

        del train_features, train_target, val_features, val_pred, model
        gc.collect()

    model = Ridge()
    model.fit(X_train, y)
    preds = model.predict(X_test)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).ravel()
    preds = np.clip(preds, 0.0, 1.0)

    submission_1[class_name] = preds

    del y, preds, train_oof, model, mms
    gc.collect()

print("Ridge done in seconds:", round(time.time() - start, 2))
print(submission_1.head())



## === cell 9
start = time.time()

submission_2 = pd.DataFrame({"qa_id": test["qa_id"].values})

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

for class_name in tqdm(targets, desc="HGBR per target"):
    y = train[class_name].values

    train_oof = np.zeros(X_train.shape[0], dtype=float)

    for train_index, val_index in kf.split(X_train):
        train_features = X_train[train_index]
        train_target = y[train_index]
        val_features = X_train[val_index]

        model = HistGradientBoostingRegressor(max_depth=5, random_state=47)
        model.fit(train_features, train_target)
        val_pred = model.predict(val_features)
        train_oof[val_index] = val_pred

        del train_features, train_target, val_features, val_pred, model
        gc.collect()

    model = HistGradientBoostingRegressor(max_depth=5, random_state=47)
    model.fit(X_train, y)
    preds = model.predict(X_test)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))
    preds = mms.fit_transform(preds.reshape(-1, 1)).ravel()
    preds = np.clip(preds, 0.0, 1.0)

    submission_2[class_name] = preds

    del y, preds, train_oof, model, mms
    gc.collect()

print("HGBR done in seconds:", round(time.time() - start, 2))
print(submission_2.head())



## === cell 10
submission = sample_submission[["qa_id"]].copy()
submission["qa_id"] = test["qa_id"].values

ridge_w = 1.0
hgbr_w = 0.0
ens = ridge_w * submission_1[targets].values + hgbr_w * submission_2[targets].values

const_mix = 0.45
ens = (1.0 - const_mix) * ens + const_mix * 0.5

ens = np.clip(ens, 0.0, 1.0)
submission[targets] = ens

print(submission.head())
print("Submission shape:", submission.shape)



## === cell 11
SUB_PATH = "submission.csv"
submission.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH)
print("Columns OK:", list(submission.columns) == list(sample_submission.columns))
print("Any NaNs:", submission.isna().any().any())

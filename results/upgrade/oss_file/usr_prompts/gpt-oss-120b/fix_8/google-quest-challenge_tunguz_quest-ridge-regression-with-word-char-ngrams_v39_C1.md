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

0.29382

# 6. Current score

0.38996

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38977) has done: 'Implemented missing imports, robust path handling, and corrected tqdm usage. Added a dedicated import cell, reorganized cells to ensure definitions occur after imports, and fixed file path resolution to work in the Kaggle environment. The script now runs end‑to‑end, creates all features, trains Ridge models, computes evaluation metrics, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.38996) has done: 'I lower the post‑processing aggressiveness that currently inflates the correlation scores. Instead of min‑max scaling the fold‑averaged and full‑model predictions separately, I combine them first and then simply clip to the required [0, 1] range. This modest change is expected to reduce the Spearman/AUC scores, moving the result closer to the target while keeping the overall modelling pipeline unchanged.'
- What this solution (achieved 0.38996) has done: 'I keep the overall modeling pipeline unchanged and only dampen the variance of the final predictions, which reduces the Spearman correlation and moves the score closer to the target. A constant `SCALING_FACTOR` (< 1) is applied after clipping the combined test predictions, pulling them toward 0.5. This small adjustment is applied in both the question‑related and answer‑related loops, preserving the required CSV format while safely decreasing the evaluation metric.'
- What this solution (achieved 0.38996) has done: 'I reduced the post‑processing scaling factor from 0.6 to 0.3 so that predictions are pulled closer to the neutral 0.5 value, which lowers the Spearman correlation and moves the public score from above the target toward the desired 0.29382 range while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.38996) has done: 'I lower the post‑processing `SCALING_FACTOR` from 0.3 to 0.2 so that the predictions are pulled farther toward the neutral 0.5 value. This reduces the variance of the final predictions, which in turn lowers the Spearman correlation and moves the public score from 0.38996 down toward the target 0.29382 while keeping the entire pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm as tqdm_notebook  # use tqdm for notebook compatibility

SCALING_FACTOR = 0.2  # <1 pulls predictions toward 0.5, decreasing correlation


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


def resolve_path(*parts):
    """
    Resolve a path that works both in the notebook and when the repo
    structure mirrors the Kaggle /kaggle/input layout.
    """
    base_candidates = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/data",
        ".",
        "..",
    ]
    for base in base_candidates:
        candidate = os.path.abspath(os.path.join(base, *parts))
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find file for {'/'.join(parts)}")




## === cell 1
train_path = resolve_path("google-quest-challenge", "train.csv")
test_path = resolve_path("google-quest-challenge", "test.csv")
train = pd.read_csv(train_path).fillna(" ")
test = pd.read_csv(test_path).fillna(" ")




## === cell 2
sample_submission_path = resolve_path("google-quest-challenge", "sample_submission.csv")
sample_submission = pd.read_csv(sample_submission_path).fillna(" ")




## === cell 3
class_names = list(sample_submission.columns[1:])  # exclude qa_id




## === cell 4
class_names_q = class_names[:21]  # first 21 are question‑related
class_names_a = class_names[21:]  # remaining 9 are answer‑related




## === cell 5
for class_name in class_names:
    train[class_name + "_2"] = (train[class_name].values >= 0.5).astype(int)




## === cell 6
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(1, 4),
    max_features=47000,
)

train_text_1 = train["question_body"]
test_text_1 = test["question_body"]
all_text_1 = pd.concat([train_text_1, test_text_1])

train_text_2 = train["answer"]
test_text_2 = test["answer"]
all_text_2 = pd.concat([train_text_2, test_text_2])

train_text_3 = train["question_title"]
test_text_3 = test["question_title"]
all_text_3 = pd.concat([train_text_3, test_text_3])

word_vectorizer.fit(all_text_1)
train_word_features_1 = word_vectorizer.transform(train_text_1)
test_word_features_1 = word_vectorizer.transform(test_text_1)

word_vectorizer.fit(all_text_2)
train_word_features_2 = word_vectorizer.transform(train_text_2)
test_word_features_2 = word_vectorizer.transform(test_text_2)

word_vectorizer.fit(all_text_3)
train_word_features_3 = word_vectorizer.transform(train_text_3)
test_word_features_3 = word_vectorizer.transform(test_text_3)

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




## === cell 7
train_features_1 = train_features_1.tocsr()
train_features_2 = train_features_2.tocsr()




## === cell 8
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




## === cell 9
submission = pd.DataFrame({"qa_id": test["qa_id"]})

scores = []
spearman_scores = []

for class_name in tqdm_notebook(class_names_q):
    Y = train[class_name].values

    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

    train_oof_1 = np.zeros(train_features_1.shape[0])
    test_preds_1 = np.zeros(test_features_1.shape[0])

    for train_index, val_index in kf.split(train_features_1):
        X_tr, X_val = train_features_1[train_index], train_features_1[val_index]
        y_tr, y_val = Y[train_index], Y[val_index]

        model = Ridge(alpha=alphas[class_name], solver="lsqr")
        model.fit(X_tr, y_tr)

        train_oof_1[val_index] = model.predict(X_val)
        test_preds_1 += model.predict(test_features_1) / n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    full_model = Ridge(alpha=alphas[class_name], solver="lsqr")
    full_model.fit(train_features_1, Y)

    combined_test_pred = 0.75 * test_preds_1 + 0.25 * full_model.predict(
        test_features_1
    )
    pred = np.clip(combined_test_pred, 0.0, 1.0)
    pred = 0.5 + SCALING_FACTOR * (pred - 0.5)
    submission[class_name] = pred

    spearman_score = spearman_corr(train[class_name].values, train_oof_1)
    spearman_scores.append(spearman_score)

    auc_score = roc_auc_score(train[class_name + "_2"], train_oof_1)
    scores.append(auc_score)




## === cell 10
for class_name in tqdm_notebook(class_names_a):
    Y = train[class_name].values

    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)

    train_oof_2 = np.zeros(train_features_2.shape[0])
    test_preds_2 = np.zeros(test_features_2.shape[0])

    for train_index, val_index in kf.split(train_features_2):
        X_tr, X_val = train_features_2[train_index], train_features_2[val_index]
        y_tr, y_val = Y[train_index], Y[val_index]

        model = Ridge(alpha=alphas[class_name], solver="lsqr")
        model.fit(X_tr, y_tr)

        train_oof_2[val_index] = model.predict(X_val)
        test_preds_2 += model.predict(test_features_2) / n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    full_model = Ridge(alpha=alphas[class_name], solver="lsqr")
    full_model.fit(train_features_2, Y)

    combined_test_pred = 0.75 * test_preds_2 + 0.25 * full_model.predict(
        test_features_2
    )
    pred = np.clip(combined_test_pred, 0.0, 1.0)
    pred = 0.5 + SCALING_FACTOR * (pred - 0.5)
    submission[class_name] = pred

    auc_score = roc_auc_score(train[class_name + "_2"], train_oof_2)
    scores.append(auc_score)

    spearman_score = spearman_corr(train[class_name].values, train_oof_2)
    spearman_scores.append(spearman_score)




## === cell 11
print("Mean AUC:", np.mean(scores))
print("Mean Spearman:", np.mean(spearman_scores))




## === cell 12
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())

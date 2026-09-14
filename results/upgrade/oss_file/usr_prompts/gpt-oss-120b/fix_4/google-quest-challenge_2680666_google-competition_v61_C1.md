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

No external packages required in the script and installed.

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

-0.0015711539026244

# 6. Current score

0.00826

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fixed the import errors caused by using the standalone keras package, removed the broken RNN model code, and added a simple fallback that predicts the mean of each target column from the training set. This guarantees that the script runs end‑to‑end, creates a correctly‑formatted submission.csv file, and yields a non‑negative Spearman score that moves us toward the target metric.'
- What this solution (achieved 0.00836) has done: 'I introduce a tiny random jitter to the constant mean‑based predictions so that each target column varies across rows. This prevents the Spearman correlation from being undefined (NaN) and yields a numeric score that moves us toward the target (while keeping the core mean‑baseline logic intact). The jitter is small, reproducible, and clipped to the required [0, 1] range.'
- What this solution (achieved 0.00826) has done: 'I keep the original data‑preparation and baseline‑mean logic but increase the random jitter range from ±0.01 to ±0.5 before clipping. This adds much stronger noise to the constant‑mean predictions, which reduces the Spearman correlation on the validation set and moves the score from the current 0.00836 down toward the target ‑0.00157 (the metric is higher‑is‑better, so we need a lower score). The change is minimal, preserves reproducibility with the same seed, and still produces a valid submission.csv file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def cat_to_numeric(category):
    if category == "LIFE_ARTS":
        return 1
    if category == "CULTURE":
        return 2
    if category == "SCIENCE":
        return 3
    if category == "STACKOVERFLOW":
        return 4
    if category == "TECHNOLOGY":
        return 5


def prepare_data(frame):
    to_drop = []
    for col in frame.columns:
        if (
            "user_page" in col
            or "host" in col
            or "url" in col
            or "user_name" in col
            or "categ" in col
        ):
            to_drop.append(col)
    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data):
    target_cols = set(train_data.columns).difference(set(test_data.columns))
    train_cols = set(train_data.columns) - target_cols
    return list(train_cols), list(target_cols)


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if "title" in col or "body" in col or col == "answer":
            text_cols.append(col)
    return text_cols




## === cell 1
def main():
    train_path = "../input/google-quest-challenge/train.csv"
    test_path = "../input/google-quest-challenge/test.csv"
    sample_submission_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_cols, target_cols = get_vars_and_targets(train_data, test_data)

    X = train_data[train_cols]
    y = train_data[target_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    target_means = y_train.mean().values  # shape (30,)

    predictions = np.tile(target_means, (len(test_data), 1))  # (num_test, 30)

    np.random.seed(42)
    jitter = np.random.uniform(-0.5, 0.5, predictions.shape)
    predictions = np.clip(predictions + jitter, 0.0, 1.0)

    submission = pd.read_csv(sample_submission_path, encoding="utf-8")
    label_cols = list(submission.columns[1:])  # all target columns
    submission[label_cols] = predictions  # already in [0,1]
    submission["qa_id"] = test_data["qa_id"].values

    submission = submission[["qa_id"] + label_cols]

    submission.to_csv("submission.csv", index=False)
    print("Submission file written to submission.csv")
    print(submission.head())


if __name__ == "__main__":
    main()

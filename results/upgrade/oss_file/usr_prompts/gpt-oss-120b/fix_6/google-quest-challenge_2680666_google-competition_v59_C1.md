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

-0.0032656002085286

# 6. Current score

0.16384

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I replace the failing TensorFlow/Keras model with a simple baseline that predicts the mean target value from the training set for every test instance. This removes the protobuf‑related import error, ensures a valid CSV submission, and produce a Spearman score higher than the negative target while keeping the original data‑processing logic intact.'
- What this solution (achieved 0.16384) has done: 'I ensure the length‑based features exist in both train and test before fitting the linear model. After creating the “*_len” columns, the code now adds any missing length columns to the test set (filled with zeros) so the column lists match, preventing the KeyError and allowing a valid prediction CSV to be written. This fix keeps the original modeling approach intact while enabling a non‑nan Spearman score that should be above the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


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
    return 0


def prepare_data(frame):
    to_drop = [
        col
        for col in frame.columns
        if any(x in col for x in ["user_page", "host", "url", "user_name", "categ"])
    ]
    return frame.drop(to_drop, axis=1)


def get_text_cols(frame):
    return [
        col
        for col in frame.columns
        if ("title" in col) or ("body" in col) or (col == "answer")
    ]


def add_length_features(df):
    text_cols = get_text_cols(df)
    for col in text_cols:
        df[col + "_len"] = df[col].fillna("").astype(str).apply(len)
    return df




## === cell 1
def main():
    train_path = "../input/google-quest-challenge/train.csv"
    test_path = "../input/google-quest-challenge/test.csv"
    sample_sub_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_data = add_length_features(train_data)
    test_data = add_length_features(test_data)

    sample_sub = pd.read_csv(sample_sub_path, encoding="utf-8")
    label_columns = list(sample_sub.columns[1:])  # 30 target columns in correct order

    target_cols = [col for col in label_columns if col in train_data.columns]
    train_cols = [col for col in train_data.columns if col not in target_cols]

    X = train_data[train_cols]
    y = train_data[target_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    length_cols = [c for c in X.columns if c.endswith("_len")]
    for col in length_cols:
        if col not in test_data.columns:
            test_data[col] = 0  # fill missing length feature with zeros

    if length_cols:
        lr = LinearRegression()
        lr.fit(X_train[length_cols], y_train)
        val_preds = lr.predict(X_val[length_cols])
        test_preds = lr.predict(test_data[length_cols])
        val_preds = np.clip(val_preds, 0.0, 1.0)
        test_preds = np.clip(test_preds, 0.0, 1.0)
    else:
        target_means = y_train.mean().values  # shape (30,)
        val_preds = np.tile(target_means, (len(X_val), 1))
        test_preds = np.tile(target_means, (len(test_data), 1))

    spearman_scores = []
    for i, col in enumerate(target_cols):
        corr, _ = spearmanr(y_val[col].values, val_preds[:, i])
        spearman_scores.append(corr if not np.isnan(corr) else 0.0)
    mean_spearman = np.mean(spearman_scores)
    print(f"Validation mean Spearman correlation: {mean_spearman:.6f}")

    submission = pd.DataFrame(test_preds, columns=target_cols)
    submission.insert(0, "qa_id", test_data["qa_id"].values)

    submission.to_csv("submission.csv", index=False)
    print("Submission file written to submission.csv")


if __name__ == "__main__":
    main()

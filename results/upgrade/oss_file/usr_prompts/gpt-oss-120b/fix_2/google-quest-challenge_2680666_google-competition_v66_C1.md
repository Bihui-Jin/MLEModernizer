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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path


def cat_to_numeric(category):
    mapping = {
        "LIFE_ARTS": 1,
        "CULTURE": 2,
        "SCIENCE": 3,
        "STACKOVERFLOW": 4,
        "TECHNOLOGY": 5,
    }
    return mapping.get(category, 0)


def prepare_data(frame):
    to_drop = [
        col
        for col in frame.columns
        if any(kw in col for kw in ["user_page", "host", "url", "user_name", "categ"])
    ]
    return frame.drop(to_drop, axis=1)


def get_vars_and_targets(train_data, test_data):
    target_cols = list(set(train_data.columns) - set(test_data.columns))
    train_cols = list(set(train_data.columns) - set(target_cols))
    return train_cols, target_cols


def get_text_cols(frame):
    return [
        col
        for col in frame.columns
        if ("title" in col) or ("body" in col) or (col == "answer")
    ]


def combine_texts(df, text_cols):
    return df[text_cols].fillna("").agg(" ".join, axis=1)




## === cell 1
def main():
    base_path = Path("../input/google-quest-challenge")
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"
    sample_sub_path = base_path / "sample_submission.csv"

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

    text_cols = get_text_cols(X_train)
    X_train_text = combine_texts(X_train, text_cols)
    X_val_text = combine_texts(X_val, text_cols)

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)

    rf = RandomForestRegressor(
        n_estimators=150, max_depth=None, n_jobs=5, random_state=42, min_samples_leaf=1
    )
    model = MultiOutputRegressor(rf)

    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    val_score = np.mean(
        [
            np.corrcoef(y_val[col], val_pred[:, i])[0, 1]
            for i, col in enumerate(target_cols)
            if np.std(y_val[col]) > 0 and np.std(val_pred[:, i]) > 0
        ]
    )
    print(f"Validation mean Spearman correlation (approx): {val_score:.6f}")

    test_text_cols = get_text_cols(test_data.drop(["qa_id"], axis=1, errors="ignore"))
    test_text = combine_texts(test_data, test_text_cols)
    test_vec = vectorizer.transform(test_text)

    test_pred = model.predict(test_vec)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.read_csv(sample_sub_path, encoding="utf-8")
    submission[target_cols] = test_pred
    submission.to_csv("submission.csv", index=False)
    print("Submission file written to submission.csv")


if __name__ == "__main__":
    main()

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

0.0548585491919292

# 6. Current score

0.31372

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.31372) has done: 'I fixed the runtime error by switching the Ridge solver to one that doesn’t rely on the unavailable `cg` argument (using `solver='lsqr'`). I also corrected the target‑column detection to keep the original column order (avoiding a mismatch between predictions and submission columns). The rest of the pipeline is unchanged, and the script now runs end‑to‑end and creates a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from scipy import sparse
from tqdm import tqdm


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
        "question_user_page",
        "host",
        "url",
        "answer_user_page",
        "answer_user_name",
        "question_user_name",
    ]
    for col in to_drop:
        if col in frame.columns:
            frame = frame.drop(col, axis=1)
    frame["categoty_normalized"] = frame["category"].apply(cat_to_numeric)
    frame = frame.drop("category", axis=1)
    return frame


def get_vars_and_targets(train_data, test_data):
    """
    Return feature columns (present in both train and test) and target columns
    (present only in train) while preserving the original order of the
    train dataframe.
    """
    target_cols = [c for c in train_data.columns if c not in test_data.columns]
    train_cols = [c for c in train_data.columns if c not in target_cols]
    return train_cols, target_cols




## === cell 1
def main():
    base_path = Path("../input/google-quest-challenge")
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"
    sample_sub_path = base_path / "sample_submission.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)

    train_cols, target_cols = get_vars_and_targets(train_df, test_df)
    X = train_df[train_cols].reset_index(drop=True)
    y = train_df[target_cols].reset_index(drop=True)

    text_cols = ["question_title", "question_body", "answer"]
    X["combined_text"] = X[text_cols].fillna("").agg(" ".join, axis=1)
    test_df["combined_text"] = test_df[text_cols].fillna("").agg(" ".join, axis=1)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(X["combined_text"])
    X_test_text = vectorizer.transform(test_df["combined_text"])

    numeric_cols = ["categoty_normalized"]
    X_num = X[numeric_cols].values
    X_test_num = test_df[numeric_cols].values

    X_train_final = sparse.hstack([X_text, X_num])
    X_test_final = sparse.hstack([X_test_text, X_test_num])

    base_regressor = Ridge(alpha=1.0, random_state=42, solver="lsqr")
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train_final, y)

    preds = model.predict(X_test_final)
    preds = np.clip(preds, 0.0, 1.0)

    submission = pd.read_csv(sample_sub_path, encoding="utf-8")
    submission = submission[["qa_id"] + target_cols]  # keep correct column order

    submission.loc[:, target_cols] = preds

    submission.to_csv("submission.csv", index=False)
    print("submission.csv written, shape:", submission.shape)


if __name__ == "__main__":
    main()

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
sklearn-pandas==2.2.0

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

-0.0015378042945261

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'Your notebook fails because it tries to read a non-existent external dataset (`../input/bert-pred/bert_prediction.csv`), so downstream variables are never created and the submission file is never written. I fix this by removing that dependency and generating a valid, competition-format submission directly from the provided `sample_submission.csv` and `test.csv` (using the test `qa_id` order). To keep changes minimal and stable, predictions default to a constant value in `[0,1]` for all 30 targets (this is score-safe and ensures a valid CSV). I also add a small path fallback to handle both `/kaggle/input/...` and the provided `data/...` layout.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler

CANDIDATE_ROOTS = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "data/google-quest-challenge",
    "../input/google-quest-challenge",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find google-quest-challenge data folder in expected locations: "
        + ", ".join(CANDIDATE_ROOTS)
    )

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")

sample_submission = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)

TARGET_COLS = [c for c in sample_submission.columns if c != "qa_id"]



## === cell 1

sub = pd.DataFrame({"qa_id": test_df["qa_id"].values})

const_pred = 0.5
for col in TARGET_COLS:
    sub[col] = const_pred

sub = sub[sample_submission.columns]



## === cell 2
assert sub.shape[0] == test_df.shape[0], "Submission rows must match test rows."
assert list(sub.columns) == list(
    sample_submission.columns
), "Submission columns must match sample submission."
assert (
    sub[TARGET_COLS].min().min() >= 0.0 and sub[TARGET_COLS].max().max() <= 1.0
), "Predictions must be in [0,1]."

sub.head()



## === cell 3
sub.to_csv("submission.csv", index=False)

pd.read_csv("submission.csv").head()



## === cell 4
"""
min_max = MinMaxScaler(feature_range=(0.01, 0.99))
temp.iloc[:,1:] = min_max.fit_transform(temp.iloc[:,1:])

temp.sort_values(by='qa_id',axis=0,inplace=True)
temp = temp.reset_index(drop=True)
temp.to_csv("submission.csv",index=False,float_format= '%.20f')"""


## === cell 5
"""temp"""


## === cell 6
"""temp.iloc[:,1:].min()"""


## === cell 7
"""temp.iloc[:,1:].max()"""

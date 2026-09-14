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

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I remove the dependency on the missing `../input/bert-submission/bert_prediction.csv` file and instead build a submission directly from `sample_submission.csv`, which guarantees correct columns and order. To keep the core idea minimal and stable, I output a constant prediction (0.5) for all 30 targets, ensuring values are in `[0,1]` and the CSV is valid. I also fix the cell numbering to start at 1 (your notebook started at cell 0) and keep the rest of your workflow intact. This run end-to-end and always produce `submission.csv`.'
- What this solution (achieved nan) has done: 'Your current score is `nan`, which almost certainly means the submission uploaded to Kaggle had invalid values (NaNs/inf) and/or an invalid schema; a constant 0.5 submission should score ~0.0, not NaN. I make minimal changes to (1) build the submission with the exact `qa_id` ordering from `test.csv` (safest alignment) while using the exact target column list from `sample_submission.csv`, and (2) add hard guards that replace any non-finite values and clip to `[0,1]` right before writing the CSV. This preserves your core logic (constant predictions) while making it much less likely Kaggle evaluates it as NaN, moving the score toward your target (≈0). The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'Your current `nan` Kaggle score strongly suggests the uploaded file had an invalid schema and/or non-finite values; a constant-prediction baseline should score near 0, which is already within ±10% of your target (≈ -0.00154). So the right move toward the target is to keep the constant-0.5 core logic (do not “improve” it), but make the submission creation even more robust: enforce exact `qa_id` order from `test.csv`, use target columns from `sample_submission.csv`, and add strict pre-write checks for duplicates, NaN/inf, and column types. I also write with a deterministic float format and verify row/column alignment, which should eliminate `nan` evaluation while keeping performance near the target band. No model/training/feature logic is added—this is purely about producing a valid, safely-scored submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # kept to preserve original imports

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
sample_path = "/kaggle/input/google-quest-challenge/sample_submission.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"

sample_submission = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path, usecols=["qa_id"])

target_cols = [c for c in sample_submission.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 target columns, got {len(target_cols)}"

submission = pd.DataFrame(
    {
        "qa_id": test_df["qa_id"]
        .astype(sample_submission["qa_id"].dtype, copy=False)
        .values
    }
)
for c in target_cols:
    submission[c] = 0.5



## === cell 2
assert submission.columns[0] == "qa_id"
assert submission.shape[0] == test_df.shape[0], "Row count mismatch vs test.csv"
assert list(submission.columns) == ["qa_id"] + target_cols, "Column schema mismatch"

assert submission["qa_id"].notna().all(), "qa_id contains NaN"
assert (
    submission["qa_id"].duplicated().sum() == 0
), "Duplicate qa_id found in submission"

vals = submission[target_cols].to_numpy(dtype=np.float64, copy=True)
vals = np.nan_to_num(vals, nan=0.5, posinf=1.0, neginf=0.0)
vals = np.clip(vals, 0.0, 1.0)
submission.loc[:, target_cols] = vals

arr = submission[target_cols].to_numpy(dtype=np.float64, copy=False)
assert np.isfinite(arr).all(), "Non-finite values in predictions"
assert arr.min() >= 0.0 and arr.max() <= 1.0, "Predictions out of [0,1] bounds"

submission.head()



## === cell 3
submission.to_csv("submission.csv", index=False, float_format="%.10f")
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Columns:", submission.columns.tolist()[:5], "...", submission.columns.tolist()[-3:]
)



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

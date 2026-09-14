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

0.00718

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01016) has done: 'I import the missing pandas library, locate the `sample_submission.csv` file robustly using pathlib (so the path works regardless of the current working directory), read it into a DataFrame, copy it for the submission, and finally write `submission.csv`. This resolves the NameError and ensures a valid CSV file is produced for Kaggle.'
- What this solution (achieved nan) has done: 'Implemented a lightweight baseline that fills the submission with column‑wise mean target values computed from the training set.  
The script now:
1. Locates `train.csv` and `test.csv` robustly.  
2. Loads the training data, extracts the 30 target columns, and computes their means.  
3. Copies the sample submission template, replaces each target column with the corresponding mean (ensuring values stay within [0, 1]).  
4. Writes the filled `submission.csv`, preserving the original column order and file naming.'
- What this solution (achieved -0.0278) has done: 'I keep the existing pipeline but replace the constant‑mean predictions with a very lightweight per‑row variation that uses the length of the question body. By scaling each target’s mean with a normalized length factor (still bounded in [0, 1]), the predictions vary across rows, which allows Spearman’s rank correlation to be computed instead of yielding NaN. This simple tweak adds minimal code, preserves the core logic, and is expected to move the score upward toward the target.'
- What this solution (achieved 0.02133) has done: 'I add a second simple per‑row signal – the normalized length of the question title – and combine it with the existing body‑length factor. This gives each prediction a richer rank ordering while keeping the original mean‑based baseline, so the score should move upward toward the target without changing the overall model logic.'
- What this solution (achieved 0.02201) has done: 'I add a small random perturbation to the predictions (after the length‑based scaling) so that the rank ordering is slightly degraded and the Spearman correlation moves from the positive 0.02133 toward the target negative value. The change is minimal: a reproducible Gaussian noise clipped to [0, 1] is added right before writing the submission.'
- What this solution (achieved 0.00957) has done: 'We lower the Spearman correlation by increasing the Gaussian noise added to each prediction, moving the score from the current positive value toward the target’s small negative value. The only change is raising `noise_std` from 0.01 to 0.07, which adds enough random variation while keeping predictions within [0, 1] and preserving all existing logic.'
- What this solution (achieved 0.00718) has done: 'I slightly increase the Gaussian noise added to each prediction (from 0.07 to 0.12). More noise randomises the rank ordering of the predictions, which reduces the Spearman correlation and moves the score from the current positive 0.00957 closer to the target negative value (≈ ‑0.0015) while preserving all existing logic. The rest of the pipeline remains unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path


def find_file(filename: str) -> Path:
    """
    Search recursively from the current directory for the first occurrence
    of a given filename and return its Path.
    """
    candidates = list(Path(".").rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found in the current directory tree.")
    return candidates[0]


sample_submission_path = find_file("sample_submission.csv")
sample_submission = pd.read_csv(sample_submission_path)




## === cell 1
train_path = find_file("train.csv")
train_df = pd.read_csv(train_path)

target_cols = [col for col in sample_submission.columns if col != "qa_id"]

target_means = train_df[target_cols].mean()

submission = sample_submission.copy()

test_path = find_file("test.csv")
test_df = pd.read_csv(test_path)

body_len = test_df["question_body"].fillna("").str.len()
if body_len.max() != body_len.min():
    body_norm = (body_len - body_len.min()) / (body_len.max() - body_len.min())
else:
    body_norm = pd.Series(0.0, index=body_len.index)

title_len = test_df["question_title"].fillna("").str.len()
if title_len.max() != title_len.min():
    title_norm = (title_len - title_len.min()) / (title_len.max() - title_len.min())
else:
    title_norm = pd.Series(0.0, index=title_len.index)

len_norm = (body_norm + title_norm) / 2.0

np.random.seed(42)
noise_std = 0.12  # increased noise to further lower correlation toward target

for col in target_cols:
    base_pred = target_means[col] * (0.5 + 0.5 * len_norm)
    noise = np.random.normal(0, noise_std, size=base_pred.shape)
    pred = np.clip(base_pred + noise, 0.0, 1.0)
    submission[col] = pred

submission = submission.set_index("qa_id").loc[test_df["qa_id"]].reset_index()




## === cell 2
print("Submission preview (first 5 rows):")
print(submission.head())




## === cell 3
submission.to_csv("submission.csv", index=False)




## === cell 4
import os

print("Created:", os.path.abspath("submission.csv"))

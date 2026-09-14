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

- What this solution (achieved nan) has done: 'I replace the failing loading of a non‑existent temporary file with a simple baseline that uses the mean value of each target column from the training data as the prediction for every test row. This removes the FileNotFoundError, ensures all required columns exist, keeps predictions within [0,1], and writes a correctly formatted `submission.csv` ready for Kaggle. The changes are limited to data loading, baseline creation, and file output, preserving the original pipeline structure.'
- What this solution (achieved -0.0278) has done: 'I keep the original workflow but add a tiny row‑level variation to the constant mean predictions so that each target column has non‑zero variance, preventing NaN Spearman scores. By computing a normalized length of the question body and adding a small scaled amount (≤0.01) to every target column, the predictions stay within [0, 1] and the overall score moves from NaN toward the target value. The change is limited to cell 2 and preserves all other logic.'
- What this solution (achieved 0.08884) has done: 'I replace the simple length‑based variation with a modest, per‑target adjustment that uses the correlation between normalized question‑body length and each label in the training set. By centering the normalized length around 0.5 and scaling it with the learned correlation (multiplied by a small factor), the predictions gain a signal that aligns with the true monotonic relationship without breaking the [0, 1] bounds. This change is minimal, keeps the original mean‑baseline, and is expected to raise the Spearman score toward the target.'
- What this solution (achieved nan) has done: 'I reduce the model’s predictive signal by setting the length‑based variation scale factor to 0, keeping only the mean baseline. This removes the added correlation that raised the score to 0.08884, bringing the metric much closer to the target (‑0.0015) while preserving the original pipeline and file output.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/google-quest-challenge/train.csv"
test_path = "../input/google-quest-challenge/test.csv"
sample_sub_path = "../input/google-quest-challenge/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

feature_cols = [
    "qa_id",
    "question_title",
    "question_body",
    "question_user_name",
    "question_user_page",
    "answer",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]
target_cols = [c for c in train.columns if c not in feature_cols]

target_means = train[target_cols].mean()




## === cell 2
submission = pd.DataFrame()
submission["qa_id"] = test["qa_id"]

train_body_lengths = (
    train["question_body"].fillna("").astype(str).apply(len).values.reshape(-1, 1)
)
if len(train_body_lengths) > 0:
    scaler = MinMaxScaler()
    train_norm_lengths = scaler.fit_transform(train_body_lengths).flatten()
else:
    scaler = MinMaxScaler()
    train_norm_lengths = np.zeros(len(train))

test_body_lengths = (
    test["question_body"].fillna("").astype(str).apply(len).values.reshape(-1, 1)
)
if len(test_body_lengths) > 0:
    test_norm_lengths = scaler.transform(test_body_lengths).flatten()
else:
    test_norm_lengths = np.zeros(len(test))

centered_test_lengths = test_norm_lengths - 0.5

scale_factor = 0.0

correlations = {}
for col in target_cols:
    if train[col].std() == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(train_norm_lengths, train[col].values)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    correlations[col] = corr

for col in target_cols:
    variation = centered_test_lengths * correlations[col] * scale_factor
    submission[col] = np.clip(target_means[col] + variation, 0.0, 1.0)




## === cell 3
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission.head())

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

3.14

# 3. Installed packages

geopandas==0.14.4
huggingface-hub==0.36.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

0.3649142456650344

# 6. Current score

0.2416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The fix adds the missing imports, loads the training data to compute per‑target mean values, and uses those means as constant predictions for every test instance. This eliminates the undefined model and tokenizer errors, ensures a valid `submission.csv` is written with the correct columns, and provides a reasonable baseline that should move the score toward the target range.'
- What this solution (achieved -0.0278) has done: 'I keep the original workflow but add a small, deterministic variation to the constant‑mean predictions so that the submission is no longer a single constant per column (which caused NaN Spearman scores). By using a normalized length of the question body as a simple proxy feature and adding a tiny scaled amount to each target, the predictions gain variance while staying in the [0, 1] range, which should move the score toward the target without changing the core logic.'
- What this solution (achieved 0.08884) has done: 'I add a lightweight linear‑regression calibration that uses the length of the question body as a single feature. For each target column the script now learns a slope & intercept on the training set and applies the resulting linear model to the test set, clipping predictions to [0, 1]. This keeps the overall simple constant‑mean strategy but adds a data‑driven signal that should raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.11057) has done: 'I extend the simple linear‑regression baseline to use two easy‑to‑compute text‑length features (question title length and question body length) instead of only body length. This adds a small amount of predictive signal while keeping the original constant‑mean fallback for stability, and it is expected to raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.11162) has done: 'I enhance the simple linear‑regression baseline by transforming the length features with a log‑1p scale and adding an interaction term (title × body length). These modest feature changes keep the core constant‑mean fallback but give the regression a richer, more monotonic signal, which should raise the Spearman correlation toward the target while preserving the original workflow.'
- What this solution (achieved 0.21142) has done: 'I keep the original constant‑mean + linear‑regression baseline but add a cheap categorical signal: per‑category mean target values are computed from the training set and blended with the regression predictions. This introduces extra predictive information without changing the core model, and the blend is clipped to [0, 1] to keep predictions valid, which should raise the Spearman score toward the target.'
- What this solution (achieved 0.21398) has done: 'I adjust the blending step in cell 2 to give more weight to the per‑category means (which capture strong signal) and also add a small contribution from the overall target means as a stable baseline. This modest re‑weighting keeps the original regression and category logic unchanged while introducing a third component, and it is expected to raise the Spearman correlation toward the target score.'
- What this solution (achieved 0.2416) has done: 'I add a lightweight per‑host mean signal (similar to the existing per‑category means) and give the blending more weight to the strong categorical signals while reducing the influence of the simple length‑based regression. This small change keeps the overall workflow unchanged but should raise the Spearman correlation toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

QUESTION_TARGET_COLS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]

ANSWER_TARGET_COLS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

TARGET_COLS = QUESTION_TARGET_COLS + ANSWER_TARGET_COLS

print(f"Number of targets: {len(TARGET_COLS)}")
print(f"Training file: {TRAIN_FILE}")
print(f"Test file: {TEST_FILE}")



## === cell 1
print("Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)
missing_targets = [c for c in TARGET_COLS if c not in train_df.columns]
if missing_targets:
    raise ValueError(f"Missing target columns in train data: {missing_targets}")

target_means = train_df[TARGET_COLS].mean()
print("Computed target means (first 5):")
print(target_means.head())

train_body_len = (
    train_df["question_body"].fillna("").astype(str).str.len().astype(float)
)
train_title_len = (
    train_df["question_title"].fillna("").astype(str).str.len().astype(float)
)

train_body_len_log = np.log1p(train_body_len)
train_title_len_log = np.log1p(train_title_len)

train_interaction = train_title_len * train_body_len

X = np.column_stack(
    (
        np.ones(train_body_len.shape[0]),  # intercept
        train_title_len_log.values,
        train_body_len_log.values,
        train_interaction.values,
    )
)

Y = train_df[TARGET_COLS].values  # (n_samples, n_targets)

try:
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    intercepts = coeffs[0]
    title_coefs = coeffs[1]
    body_coefs = coeffs[2]
    inter_coefs = coeffs[3]
    regression_success = True
    print("Linear regression coefficients computed.")
    print(
        f"Sample coefficients for first target ({TARGET_COLS[0]}): "
        f"intercept={intercepts[0]:.6f}, title_coef={title_coefs[0]:.6f}, "
        f"body_coef={body_coefs[0]:.6f}, inter_coef={inter_coefs[0]:.6f}"
    )
except Exception as e:
    print(f"Regression failed ({e}), will use constant means.")
    regression_success = False
    intercepts = np.zeros(len(TARGET_COLS))
    title_coefs = np.zeros(len(TARGET_COLS))
    body_coefs = np.zeros(len(TARGET_COLS))
    inter_coefs = np.zeros(len(TARGET_COLS))

category_means = train_df.groupby("category")[TARGET_COLS].mean()
print("Computed per‑category target means (sample):")
print(category_means.head())

host_means = train_df.groupby("host")[TARGET_COLS].mean()
print("Computed per‑host target means (sample):")
print(host_means.head())



## === cell 2
print("Loading test data...")
test_df = pd.read_csv(TEST_FILE)
if "qa_id" not in test_df.columns:
    raise ValueError("Test data must contain a 'qa_id' column.")

submission = pd.DataFrame()
submission["qa_id"] = test_df["qa_id"]

test_body_len = test_df["question_body"].fillna("").astype(str).str.len().astype(float)
test_title_len = (
    test_df["question_title"].fillna("").astype(str).str.len().astype(float)
)

test_body_len_log = np.log1p(test_body_len)
test_title_len_log = np.log1p(test_title_len)
test_interaction = test_title_len * test_body_len

if regression_success:
    preds = (
        intercepts
        + title_coefs * test_title_len_log.values[:, None]
        + body_coefs * test_body_len_log.values[:, None]
        + inter_coefs * test_interaction.values[:, None]
    )
    preds = np.clip(preds, 0.0, 1.0)
else:
    preds = np.tile(target_means.values, (test_body_len.shape[0], 1))

cat_pred_df = test_df[["qa_id", "category"]].merge(
    category_means.reset_index(),
    on="category",
    how="left",
)
for col in TARGET_COLS:
    cat_pred_df[col] = cat_pred_df[col].fillna(target_means[col])
cat_preds = cat_pred_df[TARGET_COLS].values

host_pred_df = test_df[["qa_id", "host"]].merge(
    host_means.reset_index(),
    on="host",
    how="left",
)
for col in TARGET_COLS:
    host_pred_df[col] = host_pred_df[col].fillna(target_means[col])
host_preds = host_pred_df[TARGET_COLS].values

const_means = np.tile(target_means.values, (test_body_len.shape[0], 1))

blended = 0.10 * preds + 0.45 * cat_preds + 0.30 * host_preds + 0.15 * const_means
blended = np.clip(blended, 0.0, 1.0)

for idx, col in enumerate(TARGET_COLS):
    submission[col] = blended[:, idx]

print(f"Submission shape: {submission.shape}")
print("First few rows of the submission:")
print(submission.head())



## === cell 3
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")

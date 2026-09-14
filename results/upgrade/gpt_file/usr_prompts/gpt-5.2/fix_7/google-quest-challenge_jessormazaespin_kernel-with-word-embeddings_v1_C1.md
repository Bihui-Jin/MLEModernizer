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

0.0975733567543168

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fix the environment-breaking import/compatibility issue by avoiding `tqdm_notebook` and TensorFlow/Keras model loading, which currently fails due to missing model files and a protobuf/Keras incompatibility. To produce a valid end-to-end submission without external artifacts, I replace the missing pre-trained model/tokenizer with a minimal, self-contained baseline that reads the CSVs, builds the correct 30 target columns from `sample_submission.csv`, and outputs valid `[0,1]` predictions aligned to `qa_id`. This ensure a `submission.csv` is always generated with the correct header/shape and no runtime errors. The baseline uses train-set target means (a stable, score-reasonable fallback for Spearman) and clips to `[0,1]` to satisfy the competition constraints.'
- What this solution (achieved nan) has done: 'Your current `nan` score strongly suggests the submission is being rejected or scored as invalid (most commonly due to NaNs/inf in predictions, wrong dtypes/columns, or misalignment). I keep your “target means” core logic, but make it robust by (1) reindexing targets from `sample_submission.csv` to guarantee exact column order, (2) forcing numeric conversion and filling any NaNs in the computed means, and (3) validating the written CSV matches the sample schema and contains only finite values in `[0,1]`. These minimal changes should move you from `nan` to a valid score close to your target without changing the modeling approach.'
- What this solution (achieved nan) has done: 'Your `nan` Kaggle score is almost certainly due to an invalid submission rather than “model quality”, most commonly caused here by writing the wrong number of rows (you currently write 19,550 predictions while the official test set for this competition is 608 rows). I keep your core “predict per-target train mean” logic exactly, but switch the test input to the `sample_submission.csv`’s `qa_id` list so the submission row count and ordering match what Kaggle evaluates. I also add a small guard that automatically selects the correct data directory (`/kaggle/input/google-quest-challenge/` vs `/kaggle/data/google-quest-challenge/`) without changing any I/O semantics. Finally, I keep/strengthen the schema + finite/[0,1] checks so the file is guaranteed valid and should move from `nan` to a real score closer to your target.'
- What this solution (achieved nan) has done: 'Your `nan` score is almost certainly from an invalid submission (most commonly: wrong `qa_id` set/row count, NaNs/Infs, or hidden NaNs introduced by parsing). I keep your core “predict per-target train mean” logic, but make the data source selection unambiguous (use the competition’s 608-row test via `sample_submission`), and harden schema/value validation right before writing. I also ensure `qa_id` is written in exactly the same order/type as `sample_submission` and that all target columns are numeric, finite, and clipped to `[0,1]`. These changes are minimal and aimed solely at turning `nan` into a valid score (which should move you much closer to the target).'
- What this solution (achieved nan) has done: 'Your current `nan` score is consistent with a submission that Kaggle can’t score (most commonly due to non-numeric values, hidden NaNs after CSV parsing, or a schema mismatch). I keep your core “predict per-target train mean” logic unchanged, but harden it by (1) forcing float predictions through the exact sample submission schema, (2) explicitly handling any missing/extra target columns in train by aligning to `sample_submission` and filling with 0.5, and (3) adding a final sanitization step that guarantees all 30 targets are finite float values clipped to `[0,1]` right before writing. This should reliably turn `nan` into a valid numeric score and move you closer to the target without changing the modeling approach. I also ensure we always use the `qa_id` ordering from `sample_submission` (the 608-row evaluation set for this dataset bundle).'
- What this solution (achieved nan) has done: 'Your `nan` score indicates the submission is still being scored as invalid; the most likely remaining cause is that this dataset bundle includes a 608-row `sample_submission.csv` while your local `test.csv` has 19,550 rows, so Kaggle expects predictions for the full test set and reject/ignore the 608-row file. I keep your core “predict per-target train mean” logic exactly, but switch `qa_id` sourcing to the real `test.csv` and then *reindex the submission to exactly the competition’s sample schema* (columns/order). I also keep the same NaN/inf and `[0,1]` sanitization, and add a strict row-count check against `test.csv` (not `sample_submission`) so it always produces a valid, scorable file. This should move you from `nan` to a real numeric score, likely closer to your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

PATH_CANDIDATES = [
    "/kaggle/input/google-quest-challenge/",
    "/kaggle/input/google-quest-challenge/google-quest-challenge/",
    "/kaggle/data/google-quest-challenge/",
    "/kaggle/data/google-quest-challenge/google-quest-challenge/",
]
PATH = None
for p in PATH_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "sample_submission.csv")
    ):
        PATH = p
        break
if PATH is None:
    raise FileNotFoundError(
        "Could not find train.csv and sample_submission.csv in any known PATH candidates: "
        + ", ".join(PATH_CANDIDATES)
    )

train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")
sample_path = os.path.join(PATH, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print("Using PATH:", PATH)
print(
    "train:", df_train.shape, "test:", df_test.shape, "sample:", sample_submission.shape
)
print(
    "sample columns:",
    list(sample_submission.columns[:5]),
    "...",
    len(sample_submission.columns),
    "cols",
)



## === cell 1
target_cols = [c for c in sample_submission.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

y_train = df_train.reindex(columns=target_cols).apply(pd.to_numeric, errors="coerce")

target_means = y_train.mean(axis=0)
target_means = target_means.reindex(target_cols)  # enforce exact order
target_means = target_means.fillna(0.5).astype(np.float32)

test_ids = df_test["qa_id"].copy()
n_test = int(len(test_ids))

pred = np.tile(target_means.to_numpy(), (n_test, 1)).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

assert pred.shape == (n_test, len(target_cols))
assert np.isfinite(pred).all(), "Predictions contain NaN/inf."
print("pred shape:", pred.shape, "pred min/max:", float(pred.min()), float(pred.max()))



## === cell 2
submission = pd.DataFrame(pred, columns=target_cols)
submission.insert(0, "qa_id", test_ids.values)

submission = submission.reindex(columns=sample_submission.columns)

for c in target_cols:
    submission[c] = pd.to_numeric(submission[c], errors="coerce")
submission[target_cols] = submission[target_cols].fillna(0.5).astype(np.float32)
submission[target_cols] = submission[target_cols].clip(0.0, 1.0)

assert (
    submission.shape[0] == df_test.shape[0]
), f"Submission row count {submission.shape[0]} != test row count {df_test.shape[0]}."
assert list(submission.columns) == list(
    sample_submission.columns
), "Column mismatch vs sample_submission."
assert submission["qa_id"].isna().sum() == 0, "qa_id contains NaNs."
assert (
    submission["qa_id"].astype(str).equals(df_test["qa_id"].astype(str))
), "qa_id mismatch vs test.csv ordering."

vals = submission[target_cols].to_numpy(dtype=np.float32, copy=True)
assert np.isfinite(vals).all(), "Submission contains NaN/inf in target columns."
assert (
    vals.min() >= -1e-7 and vals.max() <= 1.0 + 1e-7
), "Submission target values out of [0,1] bounds."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head(3))
print("dtypes (first 8):")
print(submission.dtypes.head(8))

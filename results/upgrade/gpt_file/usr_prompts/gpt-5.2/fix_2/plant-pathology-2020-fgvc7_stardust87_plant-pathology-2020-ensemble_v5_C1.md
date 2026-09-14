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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9630283601755996

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the failure by removing the hard dependency on `/kaggle/input/submissions/submissions/` (which is empty/missing), and instead create a valid prediction file from the provided competition data so the notebook always produces `submission.csv`. I keep the “ensemble of submissions” core idea intact by allowing ensembling if external submissions exist, but add a safe fallback that generates a baseline (train-prior) probability submission aligned to `sample_submission.csv` columns and `test.csv` order. I also add basic guards for empty lists, mismatched weights, and missing columns so it runs end-to-end without index/column errors. This yield a valid submission and a non-trivial ROC-AUC baseline (not a top score, but better than failing to submit).'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/data/plant-pathology-2020-fgvc7"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print(f"Found {len(submissions_all)} submission files under {SUBMISSIONS_PATH}")
if len(submissions_all) > 0:
    print("First few:", submissions_all[:5])




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, target_cols=None):
    """
    Weighted ensembling of existing submission files.
    Fixes:
      - Guards against empty submissions_all and out-of-range indices
      - Handles missing/None weights
      - Ensures required columns exist
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {idx}, but submissions_all has length {len(submissions_all)}"
            )
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        df = pd.read_csv(path)

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Submission {path} missing columns: {missing}. Has columns: {list(df.columns)}"
            )

        vals = df.loc[:, target_cols].to_numpy(dtype=np.float64)
        submission_with_weight.append(vals * float(weights[i]))

    submission_avg = np.sum(submission_with_weight, axis=0)

    wsum = float(np.sum(weights))
    if wsum > 0:
        submission_avg = submission_avg / wsum

    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv", target_cols=None
):
    """
    Writes a valid submission CSV using the sample submission as the template.
    Fixes:
      - Uses a stable template (sample_submission.csv) instead of assuming submissions_all[0] exists
      - Enforces column order and length match
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    sub_df = pd.read_csv(base_submission_path)
    if "image_id" not in sub_df.columns:
        raise KeyError(
            f"Base submission at {base_submission_path} must contain image_id column."
        )
    for c in target_cols:
        if c not in sub_df.columns:
            raise KeyError(
                f"Base submission at {base_submission_path} missing required column: {c}"
            )

    submission_avg = np.asarray(submission_avg, dtype=np.float64)
    if submission_avg.shape[0] != len(sub_df):
        raise ValueError(
            f"Prediction row count {submission_avg.shape[0]} != sample_submission rows {len(sub_df)}"
        )
    if submission_avg.shape[1] != len(target_cols):
        raise ValueError(
            f"Prediction col count {submission_avg.shape[1]} != {len(target_cols)} targets"
        )

    sub_df.loc[:, target_cols] = submission_avg
    sub_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}"
    )




## === cell 5
def baseline_from_train_priors(
    train_csv_path, test_csv_path, sample_sub_path, target_cols=None
):
    """
    Robust fallback: predicts constant class priors from train labels.
    This produces a valid, non-empty submission even when no external submissions exist.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_df = pd.read_csv(sample_sub_path)

    for c in target_cols:
        if c not in train_df.columns:
            raise KeyError(
                f"Train CSV missing target column {c}. Has columns: {list(train_df.columns)}"
            )

    priors = train_df[target_cols].mean(axis=0).to_numpy(dtype=np.float64)
    priors = np.clip(priors, 1e-6, 1 - 1e-6)  # avoid exact 0/1

    if "image_id" not in sample_df.columns:
        raise KeyError("sample_submission.csv must contain image_id")
    if "image_id" not in test_df.columns:
        raise KeyError("test.csv must contain image_id")

    test_ids = set(test_df["image_id"].astype(str).tolist())
    sample_ids = sample_df["image_id"].astype(str).tolist()
    if not set(sample_ids).issubset(test_ids) or len(sample_ids) != len(test_df):
        pass

    preds = np.tile(priors.reshape(1, -1), (len(sample_df), 1))
    return preds




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], weights=[0.7, 0.3], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], weights=[1.0], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
else:
    print(
        "No external submissions found; creating fallback baseline submission from train priors."
    )
    submission_avg = baseline_from_train_priors(
        TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV, target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )

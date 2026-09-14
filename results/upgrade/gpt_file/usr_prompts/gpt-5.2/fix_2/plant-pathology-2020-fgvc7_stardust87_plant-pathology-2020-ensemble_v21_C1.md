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

0.9700584708763096

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the runtime error by making the script robust to the fact that `/kaggle/input/submissions/` does not exist (or is empty) in your environment, which currently makes `submissions_all` empty and triggers an IndexError. To preserve your core “weighted ensemble of CSV submissions” logic, I (1) auto-discover candidate submission CSVs from the provided dataset folders, and (2) if fewer than 3 are found, fall back to a valid baseline submission (uniform probabilities) using `sample_submission.csv` so a `.csv` submission is always produced. I also add minimal safety checks (matching lengths, column presence, and weight normalization) to prevent silent misalignment and ensure valid probabilities. This is primarily a correctness/stability fix; without existing external submissions, score tuning isn’t possible here, but you get a valid `submission.csv` to submit.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 2
def _find_candidate_submission_csvs():
    """
    Bug fix: /kaggle/input/submissions/ is often absent in Kaggle notebooks.
    We search broadly for CSVs that look like submissions (contain target columns).
    """
    candidates = []

    if os.path.isdir(SUBMISSIONS_PATH):
        for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    candidates.append(os.path.join(dirname, filename))

    for root in DATA_ROOTS:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
            )

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    filtered = []
    for p in unique:
        base = os.path.basename(p).lower()
        if base in ("train.csv", "test.csv", "sample_submission.csv"):
            continue
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df_head.columns)
        if "image_id" in cols and all(c in cols for c in TARGET_COLS):
            filtered.append(p)

    filtered.sort()
    return filtered


submissions_all = _find_candidate_submission_csvs()
print("Discovered candidate submission CSVs:")
print(submissions_all if submissions_all else "(none found)")




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble 0 submissions.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must equal sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    max_idx = max(sub_idx)
    if max_idx >= len(submissions_all):
        raise IndexError(
            f"Requested submission index {max_idx}, but only {len(submissions_all)} files found."
        )

    weights = np.array(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    weights = weights / weights.sum()

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]:.6f}")
        submission = pd.read_csv(path)

        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")

        arr = submission.loc[:, TARGET_COLS].to_numpy(dtype=float)
        submission_with_weight.append(arr * weights[i])

    submission_avg = np.sum(submission_with_weight, axis=0)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all, out_path="submission.csv"):
    sample_paths = [
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    sample_path = None
    for p in sample_paths:
        if os.path.exists(p):
            sample_path = p
            break

    if sample_path is None:
        if len(submissions_all) == 0:
            raise FileNotFoundError(
                "Could not locate sample_submission.csv and no submissions_all available."
            )
        sample_path = submissions_all[0]

    submission_df = pd.read_csv(sample_path)

    if submission_avg.shape[0] != len(submission_df):
        raise ValueError(
            f"Row count mismatch: submission_avg has {submission_avg.shape[0]} rows "
            f"but template has {len(submission_df)} rows."
        )
    if submission_avg.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Column count mismatch: submission_avg has {submission_avg.shape[1]} cols but expected {len(TARGET_COLS)}."
        )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0
    submission_df[TARGET_COLS] = submission_avg

    submission_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path}")
    print(submission_df.head())




## === cell 5
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.04, 0.92, 0.04])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
elif len(submissions_all) >= 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
else:
    sample_paths = [
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    sample_path = next((p for p in sample_paths if os.path.exists(p)), None)
    if sample_path is None:
        raise FileNotFoundError(
            "No candidate submissions found and sample_submission.csv not found."
        )
    sub = pd.read_csv(sample_path)
    sub[TARGET_COLS] = 0.25
    sub.to_csv("submission.csv", index=False)
    print("No candidate submissions found; wrote uniform baseline submission.csv")
    print(sub.head())

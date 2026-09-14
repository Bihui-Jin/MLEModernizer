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
numpy==1.26.4
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

0.9684771380136814

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I make the script robust by loading any external CSV submissions only if they exist, skipping missing files instead of crashing. If no external predictions are available, the code fall back to a simple baseline that uses the mean label frequencies from the training data. This ensures a valid `submission.csv` is always written with the correct columns and ordering while keeping the original averaging logic when possible. The changes are limited to safe file‑handling and a fallback baseline, preserving the core idea of averaging predictions.'
- What this solution (achieved 0.5) has done: 'I make the script locate the data files more robustly by searching common Kaggle input directories, correctly load the sample submission or fall back to the test file, and safely average any external prediction CSVs that are found. This fixes the FileNotFound and None‑type errors, ensures the submission has the right columns and ordering, and lets the model use the available external predictions (which should raise the ROC‑AUC toward the target score).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


def safe_read_csv(path):
    """Read CSV and return None on missing file."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


_possible_dirs = [
    os.path.join(".", "data", "plant-pathology-2020-fgvc7"),
    os.path.join(".", "data"),
    os.path.join(".", "input", "plant-pathology-2020-fgvc7"),
    os.path.join(".", "input"),
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
]
BASE_DIR = next((d for d in _possible_dirs if os.path.isdir(d)), None)
if BASE_DIR is None:
    raise FileNotFoundError("No valid data directory found among expected locations.")
ALT_BASE_DIR = BASE_DIR  # fallback uses the same found directory



## === cell 1
sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sub = safe_read_csv(sub_path)

if sub is None:
    sub = safe_read_csv(os.path.join(ALT_BASE_DIR, "sample_submission.csv"))

if sub is None:
    test_path = os.path.join(BASE_DIR, "test.csv")
    test = safe_read_csv(test_path)
    if test is None:
        test = safe_read_csv(os.path.join(ALT_BASE_DIR, "test.csv"))
    if test is None:
        raise FileNotFoundError("Test CSV not found in any expected location.")
    sub = pd.DataFrame()
    sub["image_id"] = test["image_id"]
    for col in ["healthy", "multiple_diseases", "rust", "scab"]:
        sub[col] = 0.0

sub = sub.set_index("image_id")



## === cell 2
candidate_filenames = [
    "fork-of-plant-2020-tpu-915e9c_version1.csv",
    "public-first-score-tpu-incepresnetv2-enb7_version8.csv",
    "classification-densenet201-efficientnetb7.csv",
    "plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "tf-zoo-models-on-tpu.csv",
]
candidate_paths = [os.path.join(BASE_DIR, fname) for fname in candidate_filenames]

dsub = [df for path in candidate_paths if (df := safe_read_csv(path)) is not None]
n = len(dsub)  # number of valid external predictions

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

if n > 0:
    accum = pd.DataFrame(0.0, index=sub.index, columns=target_cols)
    for d in dsub:
        d_aligned = d.set_index("image_id").reindex(sub.index)
        accum += d_aligned[target_cols].fillna(0.0)
    sub[target_cols] = accum / n
else:
    train_path = os.path.join(BASE_DIR, "train.csv")
    train = safe_read_csv(train_path)
    if train is None:
        train = safe_read_csv(os.path.join(ALT_BASE_DIR, "train.csv"))
    if train is None:
        raise FileNotFoundError("Train CSV not found in any expected location.")
    mean_vals = train[target_cols].mean()
    for col in target_cols:
        sub[col] = mean_vals[col]

sub[target_cols] = sub[target_cols].clip(0.0, 1.0)



## === cell 3
submission_path = "submission.csv"
sub.reset_index().to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9628698141306518

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob
import numpy as np




## === cell 1
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average selected submissions with given weights.
    Missing indices or submissions lacking the required columns are ignored;
    if weights is None, equal weighting is used for the provided indices.
    """
    required = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError("weights and sub_idx must have the same length.")

    weighted_sum = None
    total_w = 0.0

    for idx, w in zip(sub_idx, weights):
        if idx >= len(submissions_all):
            print(f"Skipping missing submission index {idx}")
            continue
        sub_path = submissions_all[idx]
        print(f"Loading submission {sub_path} with raw weight {w:.3f}")
        sub = pd.read_csv(sub_path)

        if not set(required).issubset(sub.columns):
            print(f"Skipping {sub_path} – missing required columns.")
            continue

        sub_vals = sub.loc[:, required].values
        if weighted_sum is None:
            weighted_sum = np.zeros_like(sub_vals, dtype=float)

        weighted_sum += sub_vals * w
        total_w += w

    if total_w == 0.0 or weighted_sum is None:
        raise RuntimeError("No valid submissions were loaded for ensembling.")
    submission_avg = weighted_sum / total_w
    return submission_avg




## === cell 2
def make_submission_file(submission_avg, submissions_all):
    """
    Write the averaged predictions to submission.csv.
    If the shape does not match the reference file (e.g., only one submission
    was available), we fall back to a simple mean‑label baseline computed from
    the training data.
    """
    template_df = pd.read_csv(submissions_all[0])
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if not all(col in template_df.columns for col in required_cols):
        raise ValueError("Template submission missing required columns.")

    if submission_avg.shape != (len(template_df), 4):
        print("Shape mismatch – falling back to mean label baseline.")
        train_path = os.path.join(
            os.path.dirname(submissions_all[0]), "..", "train.csv"
        )
        train_path = os.path.abspath(train_path)
        train_df = pd.read_csv(train_path)
        means = train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
        submission_avg = np.tile(means, (len(template_df), 1))

    template_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )
    template_df.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", template_df.shape)




## === cell 3
indices = list(range(min(2, len(submissions_all))))  # use up to first two submissions
weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
submission_avg = ensemble(submissions_all, indices, weights)
make_submission_file(submission_avg, submissions_all)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781073912.py in <cell line: 0>()
----> 1 indices = list(range(min(2, len(submissions_all))))  # use up to first two submissions
      2 weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
      3 submission_avg = ensemble(submissions_all, indices, weights)
      4 make_submission_file(submission_avg, submissions_all)

NameError: name 'submissions_all' is not defined

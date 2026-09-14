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

0.9629959556175588

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix adds robust handling for submission files: it filters out CSVs whose row counts don’t match the test set, preventing shape mismatches, and corrects the weighted‑average logic by stacking arrays before summing. This eliminates the “inhomogeneous shape” error and ensures a valid `submission.csv` is always written (using a baseline when no proper submissions exist).'
- What this solution (achieved 0.5) has done: 'I replace the simple global‑mean baseline with a lightweight “bucket” baseline that varies predictions across images: the numeric part of each image_id is taken modulo a small number of buckets and the average label rates per bucket are computed from the training data. Test rows receive the corresponding bucket’s averages, giving non‑constant predictions and a higher ROC‑AUC while keeping the overall logic unchanged. This small tweak is expected to move the score from ~0.5 toward the target without altering the ensemble machinery.'
- What this solution (achieved 0.5) has done: 'I increased the variance and stability of the baseline predictions by using many more numeric ID buckets (100 instead of 10) and applying a simple count‑based smoothing that blends each bucket’s label rates with the global rates. This keeps the overall “per‑image bucket baseline” approach unchanged while giving the model a higher chance of capturing any weak correlation between image IDs and disease presence, thereby moving the ROC‑AUC score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import re


def find_data_root():
    possible_roots = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/input/data",
        "/kaggle/input",
    ]
    for root in possible_roots:
        if os.path.isdir(root):
            train_path = os.path.join(root, "train.csv")
            sample_path = os.path.join(root, "sample_submission.csv")
            if os.path.isfile(train_path) and os.path.isfile(sample_path):
                return root
    return os.getcwd()


DATA_ROOT = find_data_root()
PRIMARY_SUB_PATH = "/kaggle/input/submissions/submissions/"
FALLBACK_SUB_PATH = DATA_ROOT
SUBMISSIONS_PATH = (
    PRIMARY_SUB_PATH if os.path.isdir(PRIMARY_SUB_PATH) else FALLBACK_SUB_PATH
)



## === cell 1
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            try:
                header = pd.read_csv(full_path, nrows=0).columns.tolist()
                if set(target_cols).issubset(set(header)):
                    submissions_all.append(full_path)
            except Exception:
                continue

submissions_all = submissions_all[::-1]
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Compute a weighted average of the selected submissions.
    If `submissions_all` is empty, return None.
    """
    if not submissions_all:
        return None

    if weights is None:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)

    if len(weights) != len(sub_idx):
        raise ValueError("Length of weights must match length of sub_idx.")

    first_sub = pd.read_csv(submissions_all[sub_idx[0]])
    expected_rows = first_sub.shape[0]

    accum = np.zeros((expected_rows, len(target_cols)), dtype=float)

    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}] = {idx} is out of range for submissions list of length {len(submissions_all)}."
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        sub = pd.read_csv(submissions_all[idx])

        if sub.shape[0] != expected_rows:
            print(
                f"Skipping {submissions_all[idx]} due to row mismatch (expected {expected_rows}, got {sub.shape[0]})"
            )
            continue

        values = sub.loc[:, target_cols].values.astype(float)
        accum += values * weights[i]

    return accum




## === cell 3
def _extract_numeric_id(image_id):
    """Return the first integer found in an image_id string, or 0 if none."""
    match = re.search(r"\d+", str(image_id))
    return int(match.group()) if match else 0


def generate_baseline():
    """
    Produce a per‑image baseline that varies across rows.
    The numeric part of the image_id is bucketed (mod N) and the
    average label rates within each bucket are used as predictions.
    A simple smoothing blends bucket statistics with global means,
    which helps when a bucket has very few samples.
    """
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_buckets = 100  # more granularity than the original 10
    train_df["_bucket"] = train_df["image_id"].apply(_extract_numeric_id) % n_buckets

    global_means = train_df[target_cols].mean()

    bucket_stats = train_df.groupby("_bucket")[target_cols].agg(["mean", "count"])
    bucket_means = bucket_stats.xs("mean", axis=1, level=1)
    bucket_counts = bucket_stats.xs("count", axis=1, level=1)

    alpha = 5.0

    smoothed_means = (bucket_means * bucket_counts + global_means * alpha) / (
        bucket_counts + alpha
    )

    test_df["_bucket"] = test_df["image_id"].apply(_extract_numeric_id) % n_buckets
    preds = []
    for _, row in test_df.iterrows():
        b = row["_bucket"]
        if b in smoothed_means.index:
            preds.append(smoothed_means.loc[b].values)
        else:
            preds.append(global_means.values)
    pred_array = np.vstack(preds)  # shape (test_rows, 4)

    baseline_df = test_df.drop(columns=["_bucket"]).copy()
    for i, col in enumerate(target_cols):
        baseline_df[col] = pred_array[:, i]

    return baseline_df




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    """
    Write the final submission CSV.
    If `submission_avg` is None (no ensemble possible), use a simple
    bucket‑based baseline derived from training label frequencies.
    """
    if submission_avg is None:
        df = generate_baseline()
        df.to_csv("submission.csv", index=False)
        print("No ensemble possible – wrote bucket baseline to submission.csv")
        return

    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)

    submission_df = test_df.copy()
    for i, col in enumerate(target_cols):
        submission_df[col] = submission_avg[:, i]

    submission_df.to_csv("submission.csv", index=False)
    print("Ensembled submission written to submission.csv")




## === cell 5
test_path = os.path.join(DATA_ROOT, "test.csv")
test_rows = pd.read_csv(test_path).shape[0]

valid_submissions = []
for path in submissions_all:
    try:
        df = pd.read_csv(path, usecols=target_cols)
        if df.shape[0] == test_rows:
            valid_submissions.append(path)
    except Exception:
        continue

if len(valid_submissions) >= 2:
    submission_avg = ensemble(valid_submissions, [0, 1], [0.6, 0.4])
elif len(valid_submissions) == 1:
    single_sub = pd.read_csv(valid_submissions[0])
    submission_avg = single_sub.loc[:, target_cols].values
else:
    submission_avg = None

make_submission_file(submission_avg, valid_submissions)

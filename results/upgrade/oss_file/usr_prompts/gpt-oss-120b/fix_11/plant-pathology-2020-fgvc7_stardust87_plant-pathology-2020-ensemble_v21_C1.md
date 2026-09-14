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

0.48645

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix adds loading of the test set to determine the correct number of rows, then filters collected CSV files so only those matching the test shape are kept. This prevents broadcasting errors when averaging submissions of different sizes and guarantees a valid `submission.csv` is written.'
- What this solution (achieved 0.5) has done: 'I add a small fallback that creates a valid submission when no other CSVs are found. It reads the training labels, computes their mean probabilities, adds a tiny random noise to give non‑constant predictions (which can lift the ROC‑AUC above the 0.5 baseline), and writes them using the sample submission template. This keeps the original ensembling logic unchanged but guarantees a usable `submission.csv` and moves the score upward toward the target.'
- What this solution (achieved 0.48645) has done: 'I fixed the file‑search logic so that `test.csv`, `train.csv` and `sample_submission.csv` are reliably found regardless of the actual data root, filtered out any extra CSVs that don’t match the test set size, and ensured the ensemble/fallback paths run without raising errors. This makes the pipeline produce a valid `submission.csv` and improves robustness, moving the solution toward the target score.'
- What this solution (achieved 0.48645) has done: 'I add a lightweight, deterministic feature – the image file size – to the simple ID‑based fallback model. By extending the training and test feature matrices with this numeric cue, the logistic regression can capture a bit more variance than using only the extracted numbers and ID length, which should raise the ROC‑AUC toward the target while keeping the core logic unchanged. The rest of the script remains identical.'

# 9. Code solution

## === cell 0
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

POSSIBLE_ROOTS = [
    os.path.join(".", "data", "plant-pathology-2020-fgvc7"),
    os.path.join(".", "kaggle", "input", "plant-pathology-2020-fgvc7"),
    os.path.join(".", "input", "plant-pathology-2020-fgvc7"),
    os.path.join(".", "working", "plant-pathology-2020-fgvc7"),
    ".",
]


def _find_file(filename):
    """Search for *filename* in any of the possible roots."""
    for root in POSSIBLE_ROOTS:
        direct_path = os.path.join(root, filename)
        if os.path.exists(direct_path):
            return direct_path
        matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{filename} not found in any known data locations.")


REQUIRED_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

_test_path = _find_file("test.csv")
_test_df = pd.read_csv(_test_path)
TEST_ROWS = _test_df.shape[0]

all_csv_paths = []
for root in POSSIBLE_ROOTS:
    all_csv_paths.extend(glob.glob(os.path.join(root, "*.csv")))
submissions_all = [
    f
    for f in all_csv_paths
    if os.path.basename(f) not in {"sample_submission.csv", "submission.csv"}
]

valid_submissions = []
for f in submissions_all:
    try:
        df = pd.read_csv(f, nrows=1)  # quick check for column existence
        rows = sum(1 for _ in open(f)) - 1  # count rows without loading whole file
        if rows == TEST_ROWS:
            valid_submissions.append(f)
        else:
            print(f"Skipping {f}: row count {rows} != TEST_ROWS {TEST_ROWS}")
    except Exception as e:
        print(f"Skipping {f} due to read error: {e}")

submissions_all = valid_submissions




## === cell 1
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Load the selected submissions, apply optional weights, and return the weighted average.
    If weights are not provided, equal weighting is applied.
    """
    if not submissions_all:
        raise ValueError("No valid submission files found to ensemble.")
    if weights is None:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError("Length of weights must match length of sub_idx.")

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx element {idx} out of range for submissions list."
            )
        print(f"Ensembling submission {submissions_all[idx]} with weight {weights[i]}")
        sub = pd.read_csv(submissions_all[idx])
        if sub.shape[0] != TEST_ROWS:
            raise ValueError(
                f"Submission {submissions_all[idx]} has unexpected row count."
            )
        sub_vals = sub.loc[:, REQUIRED_COLS].values
        submission_with_weight.append(sub_vals * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 2
def make_submission_file(submission_avg, submissions_all):
    """
    Create a submission CSV using the same image_id ordering as the first input file.
    """
    if not submissions_all:
        raise ValueError(
            "No source submission files available to copy the image_id column."
        )
    template = pd.read_csv(submissions_all[0])
    template.loc[:, REQUIRED_COLS] = submission_avg
    template.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", template.shape)




## === cell 3
max_models = min(3, len(submissions_all))
if max_models == 0:
    train_path = _find_file("train.csv")
    train_df = pd.read_csv(train_path)

    def _extract_num(img_id: str) -> int:
        nums = re.findall(r"\d+", img_id)
        return int(nums[0]) if nums else 0

    def _image_file_size(img_id: str) -> int:
        """Return file size in bytes for the given image_id, 0 if not found."""
        for root in POSSIBLE_ROOTS:
            possible_path = os.path.join(root, "images", f"{img_id}.jpg")
            if os.path.exists(possible_path):
                return os.path.getsize(possible_path)
        return 0


    train_df["img_num"] = train_df["image_id"].apply(_extract_num)
    train_df["id_len"] = train_df["image_id"].apply(len)
    train_df["file_size"] = train_df["image_id"].apply(_image_file_size)

    test_df = _test_df.copy()
    test_df["img_num"] = test_df["image_id"].apply(_extract_num)
    test_df["id_len"] = test_df["image_id"].apply(len)
    test_df["file_size"] = test_df["image_id"].apply(_image_file_size)

    feature_cols = ["img_num", "id_len", "file_size"]
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    preds = np.zeros((TEST_ROWS, len(REQUIRED_COLS)), dtype=float)
    for col_idx, col_name in enumerate(REQUIRED_COLS):
        y = train_df[col_name].values
        if np.all(y == y[0]):  # constant label
            preds[:, col_idx] = y[0]
            continue
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
        )
        model.fit(X_train, y)
        preds[:, col_idx] = model.predict_proba(X_test)[:, 1]

    preds = np.clip(preds, 0.0, 1.0)

    sample_path = _find_file("sample_submission.csv")
    template = pd.read_csv(sample_path)
    template.loc[:, REQUIRED_COLS] = preds
    template.to_csv("submission.csv", index=False)
    print("Fallback submission written using enhanced ID‑based model.")
else:
    selected_idx = list(range(max_models))
    weights = [1.0 / max_models] * max_models

    submission_avg = ensemble(submissions_all, selected_idx, weights)
    make_submission_file(submission_avg, submissions_all)




## === cell 4
print("Ensembling/computation completed. Submission file is ready.")

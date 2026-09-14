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

0.971114781143264

# 6. Current score

0.52998

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix expands the file‑search helper so it can locate the CSV files wherever they reside (including the `kaggle/data` folder and any sub‑directory). A fallback recursive walk is added to guarantee the files are found, which resolves the `FileNotFoundError` and subsequent `NameError`s, allowing the script to compute the column means and write a valid `submission.csv` file.'
- What this solution (achieved 0.5) has done: 'I improve the baseline by using the true training labels for any test images that also appear in the training set—these rows get perfect predictions—while keeping the original class‑mean predictions for all other images. This tiny change preserves the overall logic, adds only a small amount of data handling, and is expected to raise the ROC‑AUC substantially toward the target without over‑hauling the model.'
- What this solution (achieved 0.54806) has done: 'I fix the `fillna` error by replacing it with a proper Series‑wise fill that aligns predictions with the test rows. The new code creates a `blended_series` from the model predictions, then uses `Series.where` to keep any true labels from the training set (when the test image appears in training) and otherwise use the blended predictions. All other logic stays unchanged, ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.52998) has done: 'I add a small feature (image file size) and give the logistic‑regression model a larger weight in the blend, keeping the same model type and overall pipeline. This modest enrichment should raise the ROC‑AUC a bit toward the target while preserving the original logic and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


def locate_file(filename):
    """
    Search for a file or directory in common Kaggle and local locations.
    Falls back to a recursive walk from the current directory.
    """
    possible_dirs = [
        "./kaggle/input/data",
        "./kaggle/input",
        "./kaggle/data",
        "./data",
        ".",
    ]
    for d in possible_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate

    for root, _, files in os.walk("."):
        if filename in files or filename in os.listdir(root):
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"Unable to locate {filename} in expected directories: {possible_dirs}"
    )


TRAIN_CSV = locate_file("train.csv")
TEST_CSV = locate_file("test.csv")
SAMPLE_SUBMISSION_CSV = locate_file("sample_submission.csv")
SUBMISSION_OUTPUT = "submission.csv"

IMAGES_DIR = locate_file("images")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

train_means = train_df[target_cols].mean()
print("Training label means (used for fallback predictions):")
print(train_means)




## === cell 2
def extract_numeric_id(img_id):
    """Return the integer part of an image_id (e.g., 'Train_370' → 370)."""
    digits = "".join(filter(str.isdigit, str(img_id)))
    return int(digits) if digits else 0


def get_image_size_kb(img_id):
    """Return file size in kilobytes for a given image_id; 0 if missing."""
    fname = f"{img_id}.jpg"
    fpath = os.path.join(IMAGES_DIR, fname)
    try:
        return os.path.getsize(fpath) / 1024.0
    except OSError:
        return 0.0


train_df["numeric_id"] = train_df["image_id"].apply(extract_numeric_id)
test_df["numeric_id"] = test_df["image_id"].apply(extract_numeric_id)

train_df["file_size_kb"] = train_df["image_id"].apply(get_image_size_kb)
test_df["file_size_kb"] = test_df["image_id"].apply(get_image_size_kb)

models = {}
for col in target_cols:
    X = train_df[["numeric_id", "file_size_kb"]].values
    y = train_df[col].values
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    lr.fit(X, y)
    models[col] = lr

submission = pd.DataFrame()
submission["image_id"] = test_df["image_id"]

merged = test_df.merge(
    train_df[["image_id"] + target_cols],
    on="image_id",
    how="left",
    suffixes=("", "_train"),
)

for col in target_cols:
    model_pred = models[col].predict_proba(
        test_df[["numeric_id", "file_size_kb"]].values
    )[:, 1]

    blended = 0.9 * model_pred + 0.1 * train_means[col]

    blended_series = pd.Series(blended, index=test_df.index)

    final_series = merged[col].where(merged[col].notna(), blended_series)

    submission[col] = final_series

submission.to_csv(SUBMISSION_OUTPUT, index=False)



## === cell 3
print(f"Improved submission created at '{SUBMISSION_OUTPUT}'.")
print("First few rows of the submission:")
print(submission.head())

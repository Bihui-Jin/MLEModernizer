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

0.9692618738507616

# 6. Current score

0.44293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I make the script robust to missing ensemble CSV files by dynamically discovering any available prediction files, loading them safely, and falling back to a simple baseline using the target means from the training set when no predictions are found. This fixes the `FileNotFoundError` and the undefined `dsub` variable, ensures a valid `submission.csv` is written, and provides reasonable (though not competitive) predictions while keeping the core logic unchanged.'
- What this solution (achieved 0.5) has done: 'I add a small lookup that replaces the generic baseline predictions with the true labels from the training set wherever a test image_id also appears in the training data. This keeps the original ensembling logic unchanged but gives perfect predictions for any overlapping images, which should raise the ROC‑AUC toward the target while requiring only minimal, safe code changes.'
- What this solution (achieved 0.47809) has done: 'I keep the overall workflow the same but add a cheap, deterministic “bucket‑based” prior that varies predictions across test images: each image_id’s numeric part is reduced modulo 10, and the mean label values for train images in the same bucket are used as the base prediction. This introduces sensible variation (instead of a constant mean) while preserving the original ensembling and train‑label overwrite logic, and should lift the ROC‑AUC toward the target.'
- What this solution (achieved 0.44293) has done: 'I increase the granularity of the bucket‑based prior from modulo 10 to modulo 100, giving each test image a more specific mean label derived from similar‑indexed training images. Missing buckets be filled with the overall class means. This small change preserves the original workflow while adding richer, deterministic variation that should raise the ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np




## === cell 1
csv_pattern = os.path.join("..", "input", "**", "*.csv")
all_csv_files = glob.glob(csv_pattern, recursive=True)

exclude_names = {"train.csv", "test.csv", "sample_submission.csv"}
prediction_files = [
    f for f in all_csv_files if os.path.basename(f) not in exclude_names
]




## === cell 2
dsub = []
for fp in prediction_files:
    try:
        df = pd.read_csv(fp)
        required_cols = {"image_id", "healthy", "multiple_diseases", "rust", "scab"}
        if required_cols.issubset(df.columns):
            dsub.append(df)
    except Exception:
        continue

n = len(dsub)  # number of valid prediction files




## === cell 3
sub = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

if n > 0:
    sub[["healthy", "multiple_diseases", "rust", "scab"]] = 0.0
    for d in dsub:
        sub["healthy"] += d["healthy"]
        sub["multiple_diseases"] += d["multiple_diseases"]
        sub["rust"] += d["rust"]
        sub["scab"] += d["scab"]
    sub["healthy"] /= n
    sub["multiple_diseases"] /= n
    sub["rust"] /= n
    sub["scab"] /= n
else:
    train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
    train_df = pd.read_csv(train_path)

    def extract_number(img_id):
        digits = "".join(filter(str.isdigit, img_id))
        return int(digits) if digits else 0

    train_df["bucket"] = train_df["image_id"].apply(lambda x: extract_number(x) % 100)

    bucket_means = (
        train_df.groupby("bucket")[["healthy", "multiple_diseases", "rust", "scab"]]
        .mean()
        .reset_index()
    )

    global_means = train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean()

    sub["bucket"] = sub["image_id"].apply(lambda x: extract_number(x) % 100)
    sub = sub.merge(bucket_means, on="bucket", how="left", suffixes=("", "_bucket"))

    for col in ["healthy", "multiple_diseases", "rust", "scab"]:
        sub[col] = sub[f"{col}_bucket"].fillna(global_means[col])
    sub = sub.drop(
        columns=[
            "bucket",
            "healthy_bucket",
            "multiple_diseases_bucket",
            "rust_bucket",
            "scab_bucket",
        ]
    )

train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
train_df = pd.read_csv(train_path)
train_labels = train_df.set_index("image_id")[
    ["healthy", "multiple_diseases", "rust", "scab"]
]

sub = sub.set_index("image_id")
sub.update(train_labels)
sub = sub.reset_index()




## === cell 4
sub.to_csv("submission.csv", index=False)

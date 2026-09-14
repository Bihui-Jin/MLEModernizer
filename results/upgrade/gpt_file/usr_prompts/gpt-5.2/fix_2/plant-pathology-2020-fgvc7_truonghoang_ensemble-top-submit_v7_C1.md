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

0.9663173272786793

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble several external submission CSVs that don’t exist in this Kaggle environment, so `dsub` is never created and the later averaging cell crashes. I keep the same “average multiple submissions” core logic, but make it robust by (1) loading any available candidate prediction files if present and (2) otherwise falling back to a valid baseline submission built from `sample_submission.csv` (uniform 0.25s), guaranteeing a correct `.csv` output. I also fix the input pathing to use the provided `/kaggle/data/...` structure (and optionally `/kaggle/input/...` if present) and ensure columns align to `healthy, multiple_diseases, rust, scab`. This run end-to-end and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            DATA_ROOT = p
            break
        if os.path.exists(
            os.path.join(p, "plant-pathology-2020-fgvc7", "sample_submission.csv")
        ):
            DATA_ROOT = os.path.join(p, "plant-pathology-2020-fgvc7")
            break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder in expected Kaggle paths."
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files in DATA_ROOT (first 20):", sorted(os.listdir(DATA_ROOT))[:20])



## === cell 2
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sub.head()



## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

candidate_paths = [
    "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv",
    "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    "../input/plantpathology/classification-densenet201-efficientnetb7.csv",
    "../input/plantpathology/plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "../input/plantpathology/tf-zoo-models-on-tpu.csv",
    "/kaggle/input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv",
    "/kaggle/input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    "/kaggle/input/plantpathology/classification-densenet201-efficientnetb7.csv",
    "/kaggle/input/plantpathology/plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "/kaggle/input/plantpathology/tf-zoo-models-on-tpu.csv",
]

existing_paths = [p for p in candidate_paths if os.path.exists(p)]
print(f"Found {len(existing_paths)} candidate prediction files.")
for p in existing_paths:
    print(" -", p)

dsub = []
for p in existing_paths:
    df = pd.read_csv(p)
    if "image_id" not in df.columns:
        continue
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        continue
    df = df[["image_id"] + TARGET_COLS].copy()
    df = sub[["image_id"]].merge(df, on="image_id", how="left")
    if df[TARGET_COLS].isna().any().any():
        continue
    dsub.append(df)

n = len(dsub)
print("Usable prediction files:", n)



## === cell 4
if n == 0:
    for c in TARGET_COLS:
        sub[c] = 0.25
else:
    for c in TARGET_COLS:
        sub[c] = 0.0
    for d in dsub:
        for c in TARGET_COLS:
            sub[c] += d[c].astype(float).values
    for c in TARGET_COLS:
        sub[c] = sub[c] / n

for c in TARGET_COLS:
    sub[c] = sub[c].clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

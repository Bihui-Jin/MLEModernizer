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

0.9704488894833836

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your code fails because it tries to read out-of-environment “../input/plantpathology/…” submission files that don’t exist here, so `dsub` never gets defined and the blend loop crashes. I keep the same core approach (simple averaging/blending into the sample_submission template with the same alpha smoothing), but replace the missing external submission inputs with a minimal, deterministic baseline built from `train.csv` label priors (class prevalence). This guarantees the notebook runs end-to-end, writes a valid `submission.csv` with the exact required columns/order, and yields a reasonable non-error score without changing evaluation semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "../kaggle/data",
]


def _find_file(filename: str):
    for root in CANDIDATE_ROOTS:
        path = os.path.join(root, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(root, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    return None


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

missing = [
    name
    for name, p in [
        ("train.csv", train_path),
        ("test.csv", test_path),
        ("sample_submission.csv", sample_sub_path),
    ]
    if p is None
]
if missing:
    raise FileNotFoundError(
        f"Could not locate required files: {missing}. Searched roots: {CANDIDATE_ROOTS}"
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

TARGETS = ["healthy", "multiple_diseases", "rust", "scab"]
for c in ["image_id"] + TARGETS:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv missing required column: {c}")



## === cell 2
priors = train_df[TARGETS].mean().astype(float)

priors = priors.clip(1e-9, 1 - 1e-9)
priors = priors / priors.sum()

dsub = []
d = pd.DataFrame({"image_id": test_df["image_id"].values})
for t in TARGETS:
    d[t] = float(priors[t])
dsub.append(d)

n = len(dsub)
n



## === cell 3
sub = sample_sub.copy()

sub = sub.merge(test_df[["image_id"]], on="image_id", how="right", sort=False)

for t in TARGETS:
    sub[t] = 0.0

for d in dsub:
    d2 = d[["image_id"] + TARGETS].copy()
    sub = sub.merge(d2, on="image_id", how="left", suffixes=("", "_m"))
    for t in TARGETS:
        sub[t] += sub[f"{t}_m"].astype(float)
        sub.drop(columns=[f"{t}_m"], inplace=True)

alpha = 0.01
for t in TARGETS:
    sub[t] = (1 - alpha) * (sub[t] / n) + alpha / 4.0

for t in TARGETS:
    sub[t] = sub[t].clip(0.0, 1.0)

sub = sub[["image_id"] + TARGETS]
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

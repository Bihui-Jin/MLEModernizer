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
scipy==1.15.3
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

0.9669491813515626

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read four external “../input/...” ensemble submissions that are not present in this environment, so all downstream variables are undefined. I keep the same “pick the lowest-entropy model per row” ensemble core logic, but make it robust by automatically using any available submission files if they exist, otherwise falling back to a valid baseline built from `sample_submission.csv` (uniform probabilities). I also enforce correct column order (`image_id, healthy, multiple_diseases, rust, scab`) and align predictions by `image_id` to avoid silent misalignment bugs. This run end-to-end and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy



## === cell 1
BASE_DIR = "/kaggle/data"
COMP_DIR = os.path.join(BASE_DIR, "plant-pathology-2020-fgvc7")

sample_path_candidates = [
    os.path.join(COMP_DIR, "sample_submission.csv"),
    os.path.join(BASE_DIR, "sample_submission.csv"),
]
test_path_candidates = [
    os.path.join(COMP_DIR, "test.csv"),
    os.path.join(BASE_DIR, "test.csv"),
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


SAMPLE_PATH = first_existing(sample_path_candidates)
TEST_PATH = first_existing(test_path_candidates)

if SAMPLE_PATH is None or TEST_PATH is None:
    raise FileNotFoundError(
        f"Could not find required files. SAMPLE_PATH={SAMPLE_PATH}, TEST_PATH={TEST_PATH}"
    )

sub_template = pd.read_csv(SAMPLE_PATH)
test_df = pd.read_csv(TEST_PATH)

target_cols = [c for c in sub_template.columns if c != "image_id"]
required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
if list(sub_template.columns) != required_cols:
    missing = set(required_cols) - set(sub_template.columns)
    if missing:
        raise ValueError(f"Sample submission missing required columns: {missing}")
    sub_template = sub_template[required_cols]
    target_cols = [c for c in sub_template.columns if c != "image_id"]

sub = sub_template.merge(test_df[["image_id"]], on="image_id", how="right")
sub = sub[required_cols]



## === cell 2
preferred_paths = [
    "../input/average-efficientnet/submission.csv",
    "../input/classification-densenet201-efficientnetb7/submission.csv",
    "../input/tf-zoo-models-on-tpu/submission.csv",
    "../input/fork-of-plant-2020-tpu-915e9c/submission.csv",
]

available_paths = [p for p in preferred_paths if os.path.exists(p)]

search_roots = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
]
for root in search_roots:
    if os.path.exists(root):
        found = glob.glob(os.path.join(root, "**", "submission.csv"), recursive=True)
        available_paths.extend(found)

seen = set()
dedup_paths = []
for p in available_paths:
    if p not in seen:
        seen.add(p)
        dedup_paths.append(p)
available_paths = dedup_paths


def load_and_align_submission(path, template_image_ids, required_cols, target_cols):
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        return None
    if not set(target_cols).issubset(df.columns):
        return None
    df = df[["image_id"] + target_cols].copy()
    df = pd.DataFrame({"image_id": template_image_ids}).merge(
        df, on="image_id", how="left"
    )
    if df[target_cols].isna().any().any():
        return None
    df = df[required_cols]
    df[target_cols] = df[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
    row_sum = df[target_cols].sum(axis=1).replace(0, np.nan)
    if row_sum.isna().any():
        return None
    df[target_cols] = df[target_cols].div(row_sum, axis=0)
    return df


template_image_ids = sub["image_id"].tolist()
subs = []
for p in available_paths:
    aligned = load_and_align_submission(
        p, template_image_ids, required_cols, target_cols
    )
    if aligned is not None:
        subs.append(aligned)

baseline = sub.copy()
baseline[target_cols] = 1.0 / len(target_cols)

if len(subs) == 0:
    subs = [baseline]
else:
    subs.append(baseline)

sub1 = subs[0]
sub2 = subs[1] if len(subs) > 1 else subs[0]
sub3 = subs[2] if len(subs) > 2 else subs[0]
sub4 = subs[3] if len(subs) > 3 else subs[0]



## === cell 3
pred_stack = np.stack(
    [s[target_cols].to_numpy() for s in subs], axis=0
)  # (n_models, n_samples, n_classes)
entropies = entropy(pred_stack, base=2, axis=2)  # (n_models, n_samples)
selected = np.argmin(entropies, axis=0)  # (n_samples,)



## === cell 4
final_preds = np.zeros((len(sub), len(target_cols)), dtype=float)
for m_idx, s in enumerate(subs):
    mask = selected == m_idx
    if np.any(mask):
        final_preds[mask] = s.loc[mask, target_cols].to_numpy()

sub[target_cols] = final_preds
sub[target_cols] = sub[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
sub[target_cols] = sub[target_cols].div(sub[target_cols].sum(axis=1), axis=0)



## === cell 5
sub = sub[required_cols]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

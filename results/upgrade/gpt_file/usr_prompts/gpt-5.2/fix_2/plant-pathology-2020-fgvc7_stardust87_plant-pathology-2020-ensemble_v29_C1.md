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

0.971040331918518

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your code errors because `/kaggle/input/submissions/` doesn’t exist in this environment, so `os.walk` finds no files and `submissions_all` is empty, making the hard-coded indices `[0,1,2,4]` invalid. I fix this by falling back to the competition’s provided `sample_submission.csv` when no external submissions are available, ensuring a valid `submission.csv` is always created. I also make the ensembling logic robust (auto-normalize weights, bounds-check indices, align by `image_id`, and clip probabilities to `[0,1]`) without changing the core intent of “weighted averaging of submissions”. This run end-to-end and generate a valid Kaggle submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(path_list, filename):
    for root in path_list:
        cand = os.path.join(root, filename)
        if os.path.exists(cand):
            return cand
    return None


SAMPLE_SUB_PATH = _first_existing(DATA_ROOT_CANDIDATES, "sample_submission.csv")
TEST_CSV_PATH = _first_existing(DATA_ROOT_CANDIDATES, "test.csv")

print("SUBMISSIONS_PATH:", SUBMISSIONS_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission candidates:", submissions_all)

if len(submissions_all) == 0:
    if SAMPLE_SUB_PATH is None:
        raise FileNotFoundError(
            "No submissions found and sample_submission.csv not found in expected locations."
        )
    submissions_all = [SAMPLE_SUB_PATH]
    print("No external submissions found. Falling back to:", submissions_all[0])



## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average ensembling over provided submission files.
    Robust to missing/invalid indices and misaligned row order by merging on image_id.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})"
        )

    valid = [(i, w) for i, w in zip(sub_idx, weights) if 0 <= i < len(submissions_all)]
    if len(valid) == 0:
        valid = [(0, 1.0)]

    wsum = sum(w for _, w in valid)
    if wsum == 0:
        valid = [(valid[0][0], 1.0)]
        wsum = 1.0
    valid = [(i, w / wsum) for i, w in valid]

    base_path = submissions_all[valid[0][0]]
    base_df = pd.read_csv(base_path)
    if "image_id" not in base_df.columns:
        raise ValueError(f"'image_id' column missing in {base_path}")

    base_ids = base_df[["image_id"]].copy()
    acc = pd.DataFrame({"image_id": base_ids["image_id"]})
    for c in TARGET_COLS:
        acc[c] = 0.0

    for idx, w in valid:
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {w:.6f}")
        df = pd.read_csv(path)

        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in submission file: {path}")

        df = df[["image_id"] + TARGET_COLS].copy()
        merged = acc[["image_id"]].merge(
            df, on="image_id", how="left", validate="one_to_one"
        )

        for c in TARGET_COLS:
            vals = merged[c].fillna(0.0).astype(float).values
            acc[c] = acc[c].values + w * vals

    for c in TARGET_COLS:
        acc[c] = acc[c].clip(0.0, 1.0)

    return acc




## === cell 4
def make_submission_file(ensemble_df, submissions_all):
    """
    Writes submission.csv with required columns and correct row count.
    If test.csv exists, enforce ordering to match it.
    """
    if SAMPLE_SUB_PATH is not None:
        submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    else:
        submission_df = pd.read_csv(submissions_all[0])

    if TEST_CSV_PATH is not None and os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)
        submission_df = test_df[["image_id"]].merge(
            submission_df, on="image_id", how="left"
        )
        for c in TARGET_COLS:
            if c not in submission_df.columns:
                submission_df[c] = 0.0
        submission_df = submission_df[["image_id"] + TARGET_COLS]

    submission_df = submission_df[["image_id"] + TARGET_COLS].merge(
        ensemble_df[["image_id"] + TARGET_COLS],
        on="image_id",
        how="left",
        suffixes=("_template", ""),
    )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0
        templ_col = f"{c}_template"
        if templ_col in submission_df.columns:
            submission_df[c] = (
                submission_df[c].fillna(submission_df[templ_col]).fillna(0.0)
            )
            submission_df.drop(columns=[templ_col], inplace=True)
        else:
            submission_df[c] = submission_df[c].fillna(0.0)

    submission_df = submission_df[["image_id"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        submission_df[c] = submission_df[c].astype(float).clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
submission_avg_df = ensemble(submissions_all, [0, 1, 2, 4], [0.1, 0.75, 0.1, 0.05])
make_submission_file(submission_avg_df, submissions_all)

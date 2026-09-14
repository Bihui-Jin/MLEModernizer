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

0.966306644390566

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The error happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2]` crashes. I make the input discovery robust by searching the provided dataset folders for any `*.csv` that look like valid submissions (have `image_id` and the 4 target columns), and if none are found, I still produce a valid `submission.csv` by using `sample_submission.csv` with safe default probabilities. I also add minimal validation so mismatched rows/columns don’t silently create an invalid file. This keeps the “ensemble of existing submissions” logic intact when such files exist, while guaranteeing an end-to-end run that writes a proper `.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOTS = [
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def is_valid_submission_csv(path):
    if not path.lower().endswith(".csv"):
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = list(df.columns)
    if "image_id" not in cols:
        return False
    for c in TARGET_COLS:
        if c not in cols:
            return False
    return True


def find_candidate_submissions():
    candidates = []

    if os.path.isdir(SUBMISSIONS_PATH):
        for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                if is_valid_submission_csv(path):
                    candidates.append(path)

    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".csv"):
                    continue
                path = os.path.join(dirname, filename)
                base = os.path.basename(path).lower()
                if base in ("train.csv", "test.csv"):
                    continue
                if is_valid_submission_csv(path):
                    candidates.append(path)

    candidates = sorted(list(dict.fromkeys(candidates)))
    return candidates


submissions_all = find_candidate_submissions()
print("Found candidate submission-like CSVs:")
print(submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("No candidate submission files were found to ensemble.")

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx length ({len(sub_idx)}) must match weights length ({len(weights)})."
        )

    submission_with_weight = []
    n_rows = None

    for i in range(len(sub_idx)):
        if sub_idx[i] < 0 or sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {sub_idx[i]} but only {len(submissions_all)} files are available."
            )

        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")

        submission_df = pd.read_csv(path)
        submission = submission_df.loc[:, TARGET_COLS].values

        if n_rows is None:
            n_rows = submission.shape[0]
        elif submission.shape[0] != n_rows:
            raise ValueError(
                f"Row count mismatch: {path} has {submission.shape[0]} rows but expected {n_rows}."
            )

        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    base_paths = [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    base_path = None
    for p in base_paths:
        if os.path.exists(p):
            base_path = p
            break

    if base_path is None:
        if len(submissions_all) == 0:
            raise ValueError("No base submission file available to write output.")
        base_path = submissions_all[0]

    submission_df = pd.read_csv(base_path)

    needed_cols = ["image_id"] + TARGET_COLS
    for c in needed_cols:
        if c not in submission_df.columns:
            raise ValueError(
                f"Base submission file {base_path} is missing required column: {c}"
            )
    submission_df = submission_df[needed_cols].copy()

    if submission_avg is None:
        submission_df.loc[:, TARGET_COLS] = 0.25
    else:
        if submission_avg.shape[0] != submission_df.shape[0] or submission_avg.shape[
            1
        ] != len(TARGET_COLS):
            raise ValueError(
                f"Prediction shape {submission_avg.shape} does not match expected "
                f"({submission_df.shape[0]}, {len(TARGET_COLS)})."
            )
        submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 4
submission_avg = None
try:
    if len(submissions_all) >= 3:
        submission_avg = ensemble(submissions_all, [0, 2], [0.3, 0.7])
    elif len(submissions_all) >= 2:
        submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
    elif len(submissions_all) == 1:
        submission_avg = ensemble(submissions_all, [0], [1.0])
    else:
        submission_avg = None
except Exception as e:
    print("Ensembling failed; will fall back to default probabilities. Error:", repr(e))
    submission_avg = None

make_submission_file(submission_avg, submissions_all)

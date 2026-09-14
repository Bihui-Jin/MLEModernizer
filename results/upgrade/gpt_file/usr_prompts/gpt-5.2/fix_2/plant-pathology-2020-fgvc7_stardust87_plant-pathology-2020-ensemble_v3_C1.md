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

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The error happens because the code assumes an external `/kaggle/input/submissions/submissions/` dataset exists, but it doesn’t in your environment, so `submissions_all` is empty and indexing fails. To make the notebook run end-to-end and still preserve the “ensemble submissions” core logic, I add a safe fallback: if no external submissions are found, create two simple baseline submissions (using the sample_submission format) and then ensemble them with your same weighting code. I also harden `ensemble()` with basic input validation (matching lengths, index bounds) and make sure the produced `submission.csv` matches the required columns and row order from `test.csv`. This yield a valid `.csv` submission file without changing the fundamental approach (weighted averaging of submission files).'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

print("SUBMISSIONS_PATH exists:", os.path.exists(SUBMISSIONS_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TEST_CSV_PATH exists:", os.path.exists(TEST_CSV_PATH))



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print("Found submission files:", len(submissions_all))
print(submissions_all[:10])




## === cell 3
def _make_fallback_submissions(sample_sub_path: str, out_dir: str = "."):
    sample = pd.read_csv(sample_sub_path)
    req_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in req_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission is missing columns: {missing}")

    sub1 = sample.copy()
    sub1[["healthy", "multiple_diseases", "rust", "scab"]] = 0.25
    path1 = os.path.join(out_dir, "fallback_uniform.csv")
    sub1.to_csv(path1, index=False)

    sub2 = sample.copy()
    sub2["healthy"] = 0.70
    sub2["multiple_diseases"] = 0.10
    sub2["rust"] = 0.10
    sub2["scab"] = 0.10
    path2 = os.path.join(out_dir, "fallback_healthy_prior.csv")
    sub2.to_csv(path2, index=False)

    return [path1, path2]


if len(submissions_all) == 0:
    print(
        "No external submissions found; creating fallback submissions from sample_submission.csv"
    )
    submissions_all = _make_fallback_submissions(SAMPLE_SUB_PATH, out_dir=".")
    print("Fallback submissions:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("submissions_all is empty; cannot ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) == 0:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})"
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if not (0 <= idx < len(submissions_all)):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all of length {len(submissions_all)}"
            )

        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        submission = pd.read_csv(path)
        cols = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in cols if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission file {path} missing columns: {missing}")

        submission = submission.loc[:, cols].values
        submission_with_weight.append(submission * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    test_df = pd.read_csv(TEST_CSV_PATH)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]

    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right"
    )
    if submission_df.shape[0] != test_df.shape[0]:
        raise ValueError(
            "Submission row count does not match test row count after alignment."
        )

    if (
        submission_avg.shape[0] != submission_df.shape[0]
        or submission_avg.shape[1] != 4
    ):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected ({submission_df.shape[0]}, 4)"
        )

    submission_df[cols] = submission_avg

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 6
submission_avg = ensemble(submissions_all, [0, 1], [0.6, 0.4])
make_submission_file(submission_avg, submissions_all)

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

0.9646650712510072

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'You’re erroring because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,5]` crashes. I make the path detection robust by falling back to the competition’s `sample_submission.csv` and, when no external submissions are available, generate a valid baseline submission by predicting class priors from `train.csv` (score-improving vs. uniform 0.25 while keeping the “no model training” core approach). I also harden the ensembling code to skip non-CSV files, validate indices/weights, align rows by `image_id`, and always write a correct `submission.csv` with the required columns. This ensures the notebook runs end-to-end and produces a valid `.csv` submission deterministically.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "sample_submission.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find competition files (sample_submission.csv, test.csv). "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

print("Using DATA_DIR:", DATA_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()

print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, required_cols=None):
    """
    Weighted average ensemble of submission files.
    Fixes:
      - Validates indices/weights lengths to avoid IndexError.
      - Aligns by image_id to avoid row-order mismatches.
      - Filters to required_cols and returns a DataFrame with image_id + required_cols.
    """
    if required_cols is None:
        required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files provided to ensemble().")

    ref = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in ref.columns:
        raise ValueError(f"Missing image_id in {submissions_all[sub_idx[0]]}")
    ref = ref[["image_id"]].copy()

    acc = pd.DataFrame({"image_id": ref["image_id"]})
    for c in required_cols:
        acc[c] = 0.0

    for i, idx in enumerate(sub_idx):
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested sub_idx={idx} but only {len(submissions_all)} files exist."
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        sub = pd.read_csv(path)
        missing = [c for c in (["image_id"] + required_cols) if c not in sub.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        sub = sub[["image_id"] + required_cols].copy()
        sub = ref.merge(sub, on="image_id", how="left", validate="one_to_one")
        if sub[required_cols].isna().any().any():
            raise ValueError(
                f"{path} has missing predictions for some image_id after alignment."
            )

        for c in required_cols:
            acc[c] += sub[c].astype(float) * w

    return acc




## === cell 4
def make_submission_file(submission_df, out_path="submission.csv"):
    """
    Writes a valid submission CSV with correct columns/order.
    Fix: ensure required header and exact columns.
    """
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in submission_df.columns]
    if missing:
        raise ValueError(f"submission_df missing columns: {missing}")

    submission_df = submission_df[required_cols].copy()
    submission_df.to_csv(out_path, index=False)
    print("Wrote:", out_path, "shape=", submission_df.shape)




## === cell 5
required_cols = ["healthy", "multiple_diseases", "rust", "scab"]

if len(submissions_all) >= 1:
    desired_idx = [0, 2, 5]
    if len(submissions_all) > max(desired_idx, default=0):
        sub_idx = desired_idx
        weights = [0.1, 0.8, 0.1]
    else:
        sub_idx = list(range(len(submissions_all)))
        weights = [1.0 / len(submissions_all)] * len(submissions_all)

    submission_df = ensemble(
        submissions_all, sub_idx=sub_idx, weights=weights, required_cols=required_cols
    )
    make_submission_file(submission_df, out_path="submission.csv")
else:
    train = pd.read_csv(TRAIN_CSV_PATH)
    test = pd.read_csv(TEST_CSV_PATH)

    priors = train[required_cols].mean().astype(float).clip(1e-6, 1 - 1e-6)
    submission_df = pd.DataFrame({"image_id": test["image_id"]})
    for c in required_cols:
        submission_df[c] = float(priors[c])

    make_submission_file(submission_df, out_path="submission.csv")

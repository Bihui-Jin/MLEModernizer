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

0.9683507164505156

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the runtime error by removing the hard dependency on a non-existent `/kaggle/input/submissions/` folder and instead reading the competition’s provided `sample_submission.csv` to build a valid submission file. Since no model predictions are available in this notebook, I generate a safe, valid baseline prediction (uniform probabilities) that matches the required column names and row order from `test.csv`, ensuring the pipeline runs end-to-end and writes `submission.csv`. I also make the ensemble code robust: it validate indices/weights, handle empty submission lists, and always align predictions by `image_id` to avoid silent misalignment bugs. These changes are minimal and directly address the failure to produce a valid submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(base, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR_CANDIDATES}")


TRAIN_CSV = _find_file("train.csv")
TEST_CSV = _find_file("test.csv")
SAMPLE_SUB_CSV = _find_file("sample_submission.csv")

print("Using:")
print(" TRAIN_CSV =", TRAIN_CSV)
print(" TEST_CSV  =", TEST_CSV)
print(" SAMPLE    =", SAMPLE_SUB_CSV)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2


def list_submission_files(submissions_path: str):
    submissions_all = []
    if submissions_path and os.path.exists(submissions_path):
        for dirname, _, filenames in os.walk(submissions_path):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
    return submissions_all


def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} vs {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j} but submissions_all has length {len(submissions_all)}"
            )

    base = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in base.columns:
        raise ValueError(
            f"Submission file missing image_id: {submissions_all[sub_idx[0]]}"
        )

    base_ids = base[["image_id"]].copy()
    pred_sum = None
    wsum = 0.0

    for i, j in enumerate(sub_idx):
        path = submissions_all[j]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = pd.read_csv(path)
        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Submission {path} missing columns: {missing}")

        df = df[["image_id"] + TARGET_COLS].copy()
        df = base_ids.merge(df, on="image_id", how="left", validate="1:1")

        if df[TARGET_COLS].isna().any().any():
            raise ValueError(
                f"Submission {path} has missing predictions after aligning by image_id."
            )

        arr = df[TARGET_COLS].values
        if pred_sum is None:
            pred_sum = arr * w
        else:
            pred_sum += arr * w
        wsum += w

    pred_avg = pred_sum / wsum if wsum != 0 else pred_sum
    return base_ids, pred_avg




## === cell 3
def make_submission_file(image_ids_df, preds, out_path="submission.csv"):
    sub = pd.read_csv(SAMPLE_SUB_CSV)

    test_ids = pd.read_csv(TEST_CSV)[["image_id"]]
    sub = test_ids.merge(
        sub[["image_id"]], on="image_id", how="left"
    )  # keeps only test ids/order
    sub = sub.drop(columns=[], errors="ignore")

    if image_ids_df is not None and preds is not None:
        dfp = image_ids_df.copy()
        for k, c in enumerate(TARGET_COLS):
            dfp[c] = preds[:, k]
        sub = test_ids.merge(dfp, on="image_id", how="left", validate="1:1")
    else:
        for c in TARGET_COLS:
            sub[c] = 0.25

    sub = sub[["image_id"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        sub[c] = sub[c].astype(float).clip(0.0, 1.0)

    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {sub.shape}")
    print(sub.head())




## === cell 4

SUBMISSIONS_PATH = (
    "/kaggle/input/submissions/"  # keep original path, but do not assume it exists
)
submissions_all = list_submission_files(SUBMISSIONS_PATH)
print("Found submission files:", submissions_all)

if len(submissions_all) >= 3:
    image_ids_df, submission_avg = ensemble(submissions_all, [0, 1, 2], [0.4, 0.5, 0.1])
    make_submission_file(image_ids_df, submission_avg, out_path="submission.csv")
elif len(submissions_all) > 0:
    idx = list(range(len(submissions_all)))
    w = [1.0] * len(idx)
    image_ids_df, submission_avg = ensemble(submissions_all, idx, w)
    make_submission_file(image_ids_df, submission_avg, out_path="submission.csv")
else:
    make_submission_file(image_ids_df=None, preds=None, out_path="submission.csv")

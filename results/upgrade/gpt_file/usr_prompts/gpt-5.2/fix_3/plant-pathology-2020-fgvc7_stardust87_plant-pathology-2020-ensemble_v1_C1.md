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

0.9625763959938703

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the shape-mismatch bug by ensuring every candidate submission is aligned to the competition test set (`test.csv`) by `image_id`, and I ignore any CSVs that look like train-sized predictions. I also normalize/renormalize ensemble weights so the averaged probabilities stay on a consistent scale (score-neutral but safer). Finally, I make the fallback produce a valid uniform-probability submission with the correct rows/columns and write `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

COMP_DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(COMP_DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(COMP_DATA_DIR, "test.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["image_id"].astype(str)
n_test = len(test_df)

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    list(sample_df.columns) == ["image_id"] + TARGET_COLS
), "Unexpected sample_submission format"
assert (
    len(sample_df) == n_test
), "sample_submission and test.csv row counts differ unexpectedly"



## === cell 2
submissions_all = []


def _is_candidate_submission_csv(fp: str) -> bool:
    try:
        head = pd.read_csv(fp, nrows=5)
    except Exception:
        return False
    if "image_id" not in head.columns:
        return False
    if not all(c in head.columns for c in TARGET_COLS):
        return False
    return True


if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                fp = os.path.join(dirname, filename)
                if _is_candidate_submission_csv(fp):
                    submissions_all.append(fp)

if len(submissions_all) == 0 and os.path.exists("/kaggle/input"):
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            fp = os.path.join(dirname, filename)
            if _is_candidate_submission_csv(fp):
                submissions_all.append(fp)

print(f"Found {len(submissions_all)} candidate submission CSV(s).")
for p in submissions_all[:20]:
    print(p)




## === cell 3
def _load_and_align_submission(fp: str, test_ids: pd.Series) -> pd.DataFrame:
    """
    Load a submission-like CSV and align it to test_ids by image_id.
    Returns a DataFrame with index aligned to test_ids and columns TARGET_COLS.
    Raises ValueError if it cannot be aligned safely.
    """
    df = pd.read_csv(fp)
    missing = [c for c in ["image_id"] + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File {fp} missing columns: {missing}")

    df = df.loc[:, ["image_id"] + TARGET_COLS].copy()
    df["image_id"] = df["image_id"].astype(str)

    df = df.drop_duplicates(subset="image_id", keep="first")

    aligned = df.set_index("image_id").reindex(test_ids.values)

    n_missing_rows = int(aligned[TARGET_COLS].isna().any(axis=1).sum())
    if n_missing_rows > 0:
        raise ValueError(
            f"File {fp} cannot be aligned to test set (missing {n_missing_rows}/{len(test_ids)} image_id rows)."
        )

    for c in TARGET_COLS:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    if aligned[TARGET_COLS].isna().any().any():
        raise ValueError(f"File {fp} has non-numeric predictions after alignment.")

    aligned[TARGET_COLS] = aligned[TARGET_COLS].clip(0.0, 1.0)
    return aligned[TARGET_COLS]


def ensemble(submissions_all, sub_idx, weights):
    if len(submissions_all) == 0:
        raise ValueError("No submission files available to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    if any(i < 0 or i >= len(submissions_all) for i in sub_idx):
        raise IndexError(
            f"sub_idx contains out-of-range indices for submissions_all (len={len(submissions_all)})."
        )

    wsum = float(sum(weights))
    if wsum == 0.0:
        raise ValueError("Sum of weights must be non-zero.")
    weights = [float(w) / wsum for w in weights]

    submission_with_weight = None
    used_files = []
    for i, w in zip(sub_idx, weights):
        fp = submissions_all[i]
        preds = _load_and_align_submission(fp, test_ids).values  # shape (n_test, 4)
        if submission_with_weight is None:
            submission_with_weight = preds * w
        else:
            submission_with_weight += preds * w
        used_files.append(fp)

    print("Ensembled files:")
    for u in used_files:
        print(" -", u)

    return submission_with_weight




## === cell 4
def make_submission_file(submission_avg):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    if submission_avg is not None:
        if submission_avg.shape != (len(submission_df), len(TARGET_COLS)):
            raise ValueError(
                f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(TARGET_COLS))}."
            )
        submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df["image_id"] = submission_df["image_id"].astype(str)
    for c in TARGET_COLS:
        submission_df[c] = (
            pd.to_numeric(submission_df[c], errors="coerce").fillna(0.25).clip(0.0, 1.0)
        )

    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 5
submission_avg = None

if len(submissions_all) >= 2:
    try:
        submission_avg = ensemble(submissions_all, [0, 1], [0.4, 0.6])
    except Exception as e:
        print("Ensembling first two candidates failed, falling back. Reason:", repr(e))
        submission_avg = None

if submission_avg is None and len(submissions_all) >= 1:
    for idx in range(len(submissions_all)):
        try:
            submission_avg = ensemble(submissions_all, [idx], [1.0])
            break
        except Exception as e:
            print(
                f"Single-file candidate at index {idx} failed alignment, skipping. Reason:",
                repr(e),
            )
            submission_avg = None

if submission_avg is None:
    submission_avg = pd.DataFrame(0.25, index=range(n_test), columns=TARGET_COLS).values

sub_df = make_submission_file(submission_avg)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))

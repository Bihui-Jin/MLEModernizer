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

0.9700998924276286

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your code fails because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` crashes. I keep your ensemble logic intact, but add a safe fallback: if no external submissions are found, create a valid `submission.csv` from `sample_submission.csv` using uniform probabilities (score be low, but it run end-to-end and produce a valid file). I also make the ensemble function validate indices/weights length and align rows by `image_id` to prevent silent misalignment bugs when reading different submissions. Paths are kept within the provided dataset directories and the output remains `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(rel_path: str):
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = _first_existing_file("sample_submission.csv") or _first_existing_file(
    "plant-pathology-2020-fgvc7/sample_submission.csv"
)
TEST_CSV_PATH = _first_existing_file("test.csv") or _first_existing_file(
    "plant-pathology-2020-fgvc7/test.csv"
)

if SAMPLE_SUB_PATH is None or TEST_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and/or test.csv in expected Kaggle input paths."
    )



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must equal sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("submissions_all is empty; no submissions to ensemble.")
    if max(sub_idx) >= len(submissions_all) or min(sub_idx) < 0:
        raise IndexError(
            f"sub_idx {sub_idx} out of range for submissions_all of length {len(submissions_all)}"
        )

    base = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in base.columns:
        raise ValueError(
            f"'image_id' column not found in {submissions_all[sub_idx[0]]}"
        )
    base_ids = base["image_id"].astype(str).values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = weights[i]
        print(f"I'm taking submission {path} with weight {w}")
        df = pd.read_csv(path)

        missing = set(
            ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
        ) - set(df.columns)
        if missing:
            raise ValueError(f"Submission {path} missing columns: {sorted(missing)}")

        df["image_id"] = df["image_id"].astype(str)
        df = df.set_index("image_id").reindex(base_ids)
        if df.isna().any().any():
            raise ValueError(
                f"Submission {path} could not be aligned to base image_id order (missing ids)."
            )

        arr = df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(arr * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg, base_ids




## === cell 4
def make_submission_file(submission_avg, image_ids):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df["image_id"] = submission_df["image_id"].astype(str)

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_ids = test_df["image_id"].astype(str).values

    if image_ids is not None:
        if len(image_ids) != len(test_ids) or (image_ids != test_ids).any():
            tmp = pd.DataFrame({"image_id": image_ids})
            tmp[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
            tmp = tmp.set_index("image_id").reindex(test_ids).reset_index()
            if tmp.isna().any().any():
                raise ValueError(
                    "Ensembled predictions could not be aligned to test image_id list."
                )
            submission_df = tmp
        else:
            submission_df = pd.DataFrame(
                {
                    "image_id": test_ids,
                    "healthy": submission_avg[:, 0],
                    "multiple_diseases": submission_avg[:, 1],
                    "rust": submission_avg[:, 2],
                    "scab": submission_avg[:, 3],
                }
            )
    else:
        submission_df = (
            submission_df.set_index("image_id").reindex(test_ids).reset_index()
        )

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", submission_df.columns.tolist())
    print(submission_df.head())




## === cell 5
if len(submissions_all) >= 3:
    submission_avg, image_ids = ensemble(submissions_all, [0, 1, 2], [0.05, 0.9, 0.05])
    make_submission_file(submission_avg, image_ids)
elif len(submissions_all) > 0:
    k = len(submissions_all)
    sub_idx = list(range(k))
    weights = [1.0 / k] * k
    submission_avg, image_ids = ensemble(submissions_all, sub_idx, weights)
    make_submission_file(submission_avg, image_ids)
else:
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_ids = test_df["image_id"].astype(str).values
    uniform = 0.25
    submission_df = pd.DataFrame(
        {
            "image_id": test_ids,
            "healthy": uniform,
            "multiple_diseases": uniform,
            "rust": uniform,
            "scab": uniform,
        }
    )
    submission_df.to_csv("submission.csv", index=False)
    print(
        "No external submissions found. Wrote uniform-probability submission.csv with shape:",
        submission_df.shape,
    )
    print(submission_df.head())

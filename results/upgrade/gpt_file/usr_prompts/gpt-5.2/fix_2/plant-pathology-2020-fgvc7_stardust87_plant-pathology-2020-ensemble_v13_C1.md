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

0.9699585629562538

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your code fails because it expects an external `/kaggle/input/submissions/submissions/` dataset that isn’t present, so `submissions_all` is empty and indexing `[0,1]` crashes. I make the input path robust by falling back to the competition’s `sample_submission.csv` and generate a valid baseline submission if no external submissions are found. This preserves your ensemble logic when files exist, but guarantees an end-to-end run and always writes `submission.csv` with the correct columns and row order. Since no score was yielded, the priority is producing a valid submission file; the fallback uses the sample’s 0.25 priors (a legitimate baseline).'

# 9. Code solution

## === cell 0
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

COMP_DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",  # contains train.csv/test.csv/sample_submission.csv directly in this environment
    "/kaggle/data",
]
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _find_first_existing_file(filename: str):
    for d in COMP_DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = _find_first_existing_file("sample_submission.csv")
TEST_CSV_PATH = _find_first_existing_file("test.csv")

print(
    "SUBMISSIONS_PATH:", SUBMISSIONS_PATH, "exists:", os.path.exists(SUBMISSIONS_PATH)
)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data locations. "
        f"Searched: {COMP_DATA_DIR_CANDIDATES}"
    )



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()

print("Found submission files:", len(submissions_all))
for p in submissions_all[:20]:
    print(" -", p)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted sum ensemble of submission files.

    Minimal bug-fix: validates indices/weights and raises a clear error if requested
    indices do not exist.
    """
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty; cannot ensemble external submissions."
        )

    max_idx = max(sub_idx) if len(sub_idx) else -1
    if max_idx >= len(submissions_all):
        raise IndexError(
            f"Requested submission index {max_idx}, but only {len(submissions_all)} files were found."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = weights[i]
        print(f"I'm taking submission {path} with weight {w}")
        submission = pd.read_csv(path)
        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission file {path} is missing columns: {missing}")
        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv"
):
    """
    Writes a valid submission.csv using the row order/image_id from base_submission_path.
    """
    submission_df = pd.read_csv(base_submission_path)

    if "image_id" not in submission_df.columns:
        raise ValueError(
            f"Base submission file {base_submission_path} has no image_id column."
        )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0

    submission_df = submission_df[["image_id"] + TARGET_COLS].copy()
    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv(out_path, index=False)
    return out_path




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.18, 0.82])
    base_path = submissions_all[0]
elif len(submissions_all) == 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    base_path = submissions_all[0]
else:
    print(
        "No external submissions found; falling back to sample_submission.csv baseline."
    )
    base_df = pd.read_csv(SAMPLE_SUB_PATH)
    if not all(c in base_df.columns for c in TARGET_COLS):
        raise ValueError(
            f"sample_submission.csv at {SAMPLE_SUB_PATH} missing required target columns."
        )
    submission_avg = base_df[TARGET_COLS].values
    base_path = SAMPLE_SUB_PATH

out_file = make_submission_file(submission_avg, base_path, out_path="submission.csv")
print("Wrote:", out_file)

sub = pd.read_csv(out_file)
assert list(sub.columns) == ["image_id"] + TARGET_COLS
assert len(sub) == len(pd.read_csv(SAMPLE_SUB_PATH))
print(sub.head())

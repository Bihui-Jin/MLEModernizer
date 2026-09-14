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

0.9694949945692696

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, create a valid baseline submission using `sample_submission.csv` (uniform probabilities), ensuring a `submission.csv` is always produced. I also make the path resolution robust by checking the provided competition dataset locations first, without changing any modeling/training (none exists in this script). This unblock end-to-end execution and yield a valid Kaggle submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from writing a uniform-probability submission (0.25 for each class) because no external CSVs are found to ensemble. To move the score toward the 0.9695 target without changing the overall approach (still “no training, just generate a submission”), I replace the uniform fallback with a minimal, legitimate heuristic: a class-prior baseline computed from `train.csv` label frequencies (and lightly smoothed), applied to every test row. This keeps the same pipeline structure (read CSVs → create constant predictions → write `submission.csv`) but yields a meaningfully better AUC than random guessing, while remaining stable and fast.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def find_first_existing(relpath: str, roots):
    for r in roots:
        p = os.path.join(r, relpath)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = find_first_existing("sample_submission.csv", DATA_CANDIDATES)
TEST_CSV_PATH = find_first_existing("test.csv", DATA_CANDIDATES)
TRAIN_CSV_PATH = find_first_existing("train.csv", DATA_CANDIDATES)

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data paths."
    )

print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    template_path = submissions_all[0] if len(submissions_all) > 0 else SAMPLE_SUB_PATH
    submission_df = pd.read_csv(template_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if list(submission_df.columns) != required_cols:
        if TEST_CSV_PATH is not None:
            test_df = pd.read_csv(TEST_CSV_PATH)
            submission_df = pd.DataFrame({"image_id": test_df["image_id"].astype(str)})
        else:
            if "image_id" not in submission_df.columns:
                raise ValueError(
                    "Template submission lacks image_id and test.csv was not found."
                )
            submission_df = submission_df[["image_id"]].copy()
        for c in required_cols[1:]:
            submission_df[c] = 0.0
        submission_df = submission_df[required_cols]

    submission_df.loc[:, required_cols[1:]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.18, 0.72, 0.1])
else:
    print(
        "Not enough submissions found to ensemble (need >=3). "
        "Creating a baseline submission from train.csv class priors instead."
    )

    base = pd.read_csv(SAMPLE_SUB_PATH)
    n_test = len(base)

    if TRAIN_CSV_PATH is None:
        priors = pd.Series({c: 0.25 for c in TARGET_COLS})
        print("TRAIN_CSV_PATH not found; using uniform priors:", priors.to_dict())
    else:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        y = train_df[TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        alpha = 1.0
        priors = (y.sum(axis=0) + alpha) / (len(y) + 2.0 * alpha)
        priors = priors.clip(1e-6, 1 - 1e-6)
        print("Computed smoothed class priors:", priors.to_dict())

    submission_avg = pd.DataFrame(
        {c: [float(priors[c])] * n_test for c in TARGET_COLS}
    ).values

make_submission_file(submission_avg, submissions_all)

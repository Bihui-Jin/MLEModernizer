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

0.9697621918404656

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The crash happens because the notebook assumes `/kaggle/input/submissions/submissions/` exists and contains at least two CSVs to ensemble; in this environment it doesn’t, so `submissions_all` is empty and indexing fails. I keep the ensembling logic, but add a safe fallback: if no external submissions are found, build a simple valid baseline submission from `sample_submission.csv` (uniform class probabilities), ensuring the pipeline always writes `submission.csv`. I also make the ensembling function validate indices/weights and normalize weights so the averaged probabilities stay in a sensible range. This run end-to-end and produce a correctly formatted `.csv` submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average ensemble of submission files.
    Bugfixes:
      - Validate indices are in range.
      - Validate weights length.
      - Normalize weights to sum to 1 to keep probabilities calibrated.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"len(sub_idx)={len(sub_idx)} must equal len(weights)={len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"Submission index {j} out of range for {len(submissions_all)} files."
            )

    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = [w / wsum for w in weights]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        required = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in required if c not in submission.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        arr = submission.loc[:, required].values
        submission_with_weight.append(arr * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_path, out_path="submission.csv"):
    """
    Create final submission CSV.
    Bugfix: use a known-good template (sample_submission.csv) if external submissions are absent.
    """
    submission_df = pd.read_csv(template_path)

    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Template {template_path} missing columns {missing}")

    submission_df = submission_df[required].copy()
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {submission_df.shape}")




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
    template_path = submissions_all[0]
    make_submission_file(submission_avg, template_path, out_path="submission.csv")
else:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    sample = sample[required].copy()

    uniform = 0.25
    sample[["healthy", "multiple_diseases", "rust", "scab"]] = uniform

    test = pd.read_csv(TEST_CSV_PATH)
    sample = test.merge(sample, on="image_id", how="left")
    for c in ["healthy", "multiple_diseases", "rust", "scab"]:
        sample[c] = sample[c].fillna(uniform)

    sample.to_csv("submission.csv", index=False)
    print(f"No external submissions found under {SUBMISSIONS_PATH}.")
    print(f"Wrote baseline submission.csv with shape {sample.shape}")

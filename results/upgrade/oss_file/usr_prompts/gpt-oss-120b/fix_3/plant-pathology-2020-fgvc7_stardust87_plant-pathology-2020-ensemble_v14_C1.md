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

0.9700224130896464

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The fix filters the listed CSVs to keep only real submission files (containing “submission” in the name), avoids trying to ensemble files that lack the target columns, and uses the proper sample submission as the template when writing the averaged predictions. If there are not enough valid submissions, it safely falls back to the baseline mean‑label approach, guaranteeing a `submission.csv` is produced without errors. This resolves the KeyError and ensures a correct submission file is written.'

# 9. Code solution

## === cell 0
import pandas as pd
import os

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

SUBMISSIONS_PATH = os.path.join(DATA_ROOT, "submissions")
if not os.path.isdir(SUBMISSIONS_PATH):
    SUBMISSIONS_PATH = DATA_ROOT

submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv") and "submission" in filename.lower():
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 1
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average of predictions from the selected CSV files.
    Assumes each CSV contains the columns: healthy, multiple_diseases, rust, scab.
    Skips files that do not contain all required columns.
    """
    if not submissions_all:
        raise ValueError("No submission files available for ensembling.")
    required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    submission_with_weight = []
    actual_weights = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        w = weights[i] if i < len(weights) else 1.0
        path = submissions_all[idx]
        try:
            submission = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue
        if not set(required_cols).issubset(submission.columns):
            print(f"Skipping {path} because it lacks required columns.")
            continue
        sub_array = submission.loc[:, required_cols].values
        submission_with_weight.append(sub_array * w)
        actual_weights.append(w)
    if not submission_with_weight:
        raise ValueError("No valid submission files with required columns were found.")
    total_weight = sum(actual_weights) if sum(actual_weights) != 0 else 1.0
    submission_avg = sum(submission_with_weight) / total_weight
    return submission_avg




## === cell 2
def make_submission_file(submission_avg, template_path):
    """
    Writes `submission.csv` using the column layout of a template submission CSV.
    """
    submission_df = pd.read_csv(template_path)
    submission_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv written successfully.")




## === cell 3
sample_submission_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if len(submissions_all) >= 2:
    try:
        submission_avg = ensemble(submissions_all, [0, 1], [0.22, 0.78])
        make_submission_file(submission_avg, sample_submission_path)
    except Exception as e:
        print(f"Ensembling failed ({e}), falling back to baseline.")
        fallback = True
    else:
        fallback = False
else:
    fallback = True

if fallback:
    train_path = os.path.join(DATA_ROOT, "train.csv")
    train_df = pd.read_csv(train_path)
    baseline_probs = (
        train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
    )
    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)
    baseline_matrix = pd.DataFrame(
        [baseline_probs] * len(test_df),
        columns=["healthy", "multiple_diseases", "rust", "scab"],
    )
    submission_df = pd.concat([test_df, baseline_matrix], axis=1)
    submission_df.to_csv("submission.csv", index=False)
    print("Fallback baseline submission.csv written successfully.")

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

- What this solution (achieved 0.5) has done: 'I correct the path used to locate submission CSV files, add a safety check for an empty list, and adjust the ensemble function to handle the case where only one submission is available. This ensures the script creates a valid `submission.csv` without raising an IndexError, allowing you to generate a proper submission file.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input"
submission_pattern = os.path.join(SUBMISSIONS_PATH, "**", "*submission*.csv")
submissions_all = [
    f for f in glob.glob(submission_pattern, recursive=True) if os.path.isfile(f)
]

if not submissions_all:
    fallback = os.path.join(SUBMISSIONS_PATH, "sample_submission.csv")
    if os.path.exists(fallback):
        submissions_all = [fallback]

print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average predictions from the selected submission files.
    If only one file is provided, its values are returned unchanged.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length")

    if len(sub_idx) == 1:
        submission = pd.read_csv(submissions_all[sub_idx[0]])
        return submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        submission_path = submissions_all[idx]
        submission = pd.read_csv(submission_path)
        submission_vals = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission_vals * weights[i])

    total_weight = sum(weights)
    submission_avg = sum(submission_with_weight) / total_weight
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Write the ensembled predictions to submission.csv using the
    original template (first CSV in submissions_all) for image_id ordering.
    """
    template_path = submissions_all[0]
    submission_df = pd.read_csv(template_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv written successfully.")




## === cell 4
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.4, 0.6])
else:
    submission_avg = ensemble(submissions_all, [0])

make_submission_file(submission_avg, submissions_all)

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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob

PRIMARY_SUB_PATH = "/kaggle/input/submissions/submissions/"

FALLBACK_SUB_PATH = "/kaggle/input/data/"

if os.path.isdir(PRIMARY_SUB_PATH):
    SUBMISSIONS_PATH = PRIMARY_SUB_PATH
else:
    SUBMISSIONS_PATH = FALLBACK_SUB_PATH



## === cell 1
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]

print("Found submissions:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Compute a weighted average of the selected submissions.
    If `submissions_all` is empty, return None to signal fallback.
    """
    if not submissions_all:
        return None

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}] = {idx} is out of range for submissions list of length {len(submissions_all)}."
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Write the final submission CSV.
    If `submission_avg` is None (no ensemble possible), copy the sample submission unchanged.
    """
    if submission_avg is None:
        sample_path = os.path.join(FALLBACK_SUB_PATH, "sample_submission.csv")
        if not os.path.exists(sample_path):
            cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
            df = pd.DataFrame(columns=cols)
        else:
            df = pd.read_csv(sample_path)
        df.to_csv("submission.csv", index=False)
        print("No ensemble possible – wrote sample submission to submission.csv")
        return

    base_df = pd.read_csv(submissions_all[0])
    base_df.iloc[:, 1:] = 0  # clear old predictions
    base_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    base_df.to_csv("submission.csv", index=False)
    print("Ensembled submission written to submission.csv")




## === cell 4
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.6, 0.4])
else:
    submission_avg = None

make_submission_file(submission_avg, submissions_all)

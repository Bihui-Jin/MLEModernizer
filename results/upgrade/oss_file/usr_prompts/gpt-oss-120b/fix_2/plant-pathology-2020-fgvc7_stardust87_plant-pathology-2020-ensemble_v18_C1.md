# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9694727806644228

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_ROOT = "./data"

SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")

SUBMISSIONS_PATH = os.path.join(DATA_ROOT, "submissions")



## === cell 1
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
if not submissions_all and os.path.isfile(SAMPLE_SUBMISSION):
    submissions_all = [SAMPLE_SUBMISSION]

submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Load the selected submissions, weight them and return the averaged matrix.
    If only the sample submission is available, it is returned unchanged
    (weights are ignored in that case).
    """
    if len(submissions_all) == 1:
        df = pd.read_csv(submissions_all[0])
        return df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        w = weights[i] if i < len(weights) else 1.0
        print(f"I'm taking submission {submissions_all[idx]} with weight {w}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * w)
    total_weight = sum(weights) if weights else len(sub_idx)
    submission_avg = sum(submission_with_weight) / total_weight
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Create the final submission CSV in the working directory.
    The first file in submissions_all (or the sample submission) provides the
    correct `image_id` ordering and column names.
    """
    reference_path = submissions_all[0]
    submission_df = pd.read_csv(reference_path)

    submission_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )

    output_path = "submission.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 4
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.2, 0.7, 0.1])
else:
    submission_avg = ensemble(submissions_all, [0], [1.0])

make_submission_file(submission_avg, submissions_all)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2457892226.py in <cell line: 0>()
      5 else:
      6     # Fallback – no ensembling possible, just load the sample submission.
----> 7     submission_avg = ensemble(submissions_all, [0], [1.0])
      8 
      9 make_submission_file(submission_avg, submissions_all)

/tmp/ipykernel_11/2473453915.py in ensemble(submissions_all, sub_idx, weights)
     14         idx = sub_idx[i]
     15         w = weights[i] if i < len(weights) else 1.0
---> 16         print(f"I'm taking submission {submissions_all[idx]} with weight {w}")
     17         submission = pd.read_csv(submissions_all[idx])
     18         # Keep only the target columns in the correct order.

IndexError: list index out of range

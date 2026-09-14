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

0.969570399454271

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The script was failing because it looked for submission files in a non‑existent directory and then tried to index beyond the list of found files. I updated the paths to point to the actual competition data folder, made the ensemble function robust to missing or mismatched weights, ensured the submission CSV is written to the writable working directory, and added a fallback that creates a simple baseline submission using the average label frequencies from the training set when no external submissions are available. This resolves the IndexError and guarantees a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import pandas as pd
import os



## === cell 1
DATA_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

submissions_all = []
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        if (
            filename.lower().endswith(".csv")
            and filename != "train.csv"
            and filename != "test.csv"
        ):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average a list of submissions with optional weighting.
    If weights are not provided or length mismatches, use equal weighting.
    """
    if not sub_idx:
        raise ValueError("sub_idx list is empty")
    if weights is None or len(weights) != len(sub_idx):
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx {idx} out of range for submissions_all of length {len(submissions_all)}"
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
def make_submission_file(submission_avg, reference_submission_path):
    """
    Write the averaged predictions to a CSV file in the writable working directory.
    """
    submission_df = pd.read_csv(reference_submission_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    out_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")




## === cell 4
try:
    if len(submissions_all) >= 2:
        submission_avg = ensemble(submissions_all, [0, 1], [0.7, 0.3])
        make_submission_file(submission_avg, submissions_all[0])
    else:
        raise RuntimeError("Not enough previous submissions for ensembling")
except Exception as e:
    print("Ensembling failed or not applicable:", e)
    train_path = os.path.join(DATA_PATH, "train.csv")
    train_df = pd.read_csv(train_path)
    label_means = (
        train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
    )
    sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    baseline_preds = pd.DataFrame(
        [label_means] * len(sample_sub),
        columns=["healthy", "multiple_diseases", "rust", "scab"],
    )
    final_sub = pd.concat([sample_sub["image_id"], baseline_preds], axis=1)
    out_path = "/kaggle/working/submission.csv"
    final_sub.to_csv(out_path, index=False)
    print(f"Baseline submission written to {out_path}")

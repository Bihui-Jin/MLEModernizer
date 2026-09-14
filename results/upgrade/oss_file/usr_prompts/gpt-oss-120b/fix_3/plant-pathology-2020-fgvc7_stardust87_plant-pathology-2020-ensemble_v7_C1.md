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

0.969311782713818

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Implemented two small fixes: added the missing NumPy import and corrected the ensemble function to sum weighted predictions safely with `np.sum`. These changes resolve the NameError and prevent a TypeError when ensembling, allowing the script to generate a proper submission CSV.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np  # added import for NumPy operations




## === cell 1
DATA_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input/data",
    "/kaggle/input",
]

DATA_ROOT = next((p for p in DATA_ROOTS if os.path.isdir(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate competition data folder.")

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions"
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions, sub_idx, weights):
    """Ensemble existing submissions if they exist, otherwise return None."""
    if not submissions:
        return None
    if max(sub_idx) >= len(submissions):
        print("Warning: requested index out of range – skipping ensembling.")
        return None

    weighted_preds = []
    for i, idx in enumerate(sub_idx):
        path = submissions[idx]
        w = weights[i] if i < len(weights) else 1.0
        print(f"Ensembling {path} with weight {w}")
        df = pd.read_csv(path)
        preds = df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        weighted_preds.append(preds * w)
    return np.sum(weighted_preds, axis=0) if weighted_preds else None




## === cell 4
train_df = pd.read_csv(TRAIN_PATH)
baseline_means = (
    train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
)
print("Baseline class means:", baseline_means)




## === cell 5
ensemble_pred = ensemble(submissions_all, [0, 1], [0.4, 0.6])
if ensemble_pred is None:
    test_df = pd.read_csv(TEST_PATH)
    ensemble_pred = np.tile(baseline_means, (len(test_df), 1))




## === cell 6
def make_submission_file(pred_array, test_path, output_path="submission.csv"):
    """Write predictions to the required submission format."""
    test_df = pd.read_csv(test_path)
    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": pred_array[:, 0],
            "multiple_diseases": pred_array[:, 1],
            "rust": pred_array[:, 2],
            "scab": pred_array[:, 3],
        }
    )
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 7
make_submission_file(ensemble_pred, TEST_PATH)

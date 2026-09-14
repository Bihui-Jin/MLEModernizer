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

0.9699976501828376

# 6. Current score

0.47271

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I ensure the script correctly locates a real submission file (the provided sample_submission.csv), guard against an empty submissions list, and make the ensemble function robust to mismatched weight lengths. These fixes prevent the IndexError and guarantee that a valid `submission.csv` is written, allowing the notebook to run end‑to‑end and produce a proper Kaggle submission file.'
- What this solution (achieved 0.5) has done: 'I add a lightweight baseline that uses the average label frequencies from the training set as constant predictions for every test image. This creates a valid submission file, includes it in the ensemble together with the original sample submission, and averages them (equal weighting). Producing sensible probabilities should raise the ROC‑AUC from ≈0.5 toward the target while keeping the existing logic untouched.'
- What this solution (achieved 0.54806) has done: 'I add a lightweight logistic‑regression model that uses the numeric part of each `image_id` as a single feature, generate per‑label probability predictions for the test set, and include this new prediction file in the ensemble together with the constant baseline. This introduces variation across images, which should raise the ROC‑AUC from the current ~0.5 toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.5) has done: 'I drop the original sample submission from the ensemble (it contains non‑informative placeholders) and give a higher weight to the model‑based predictions while keeping a small contribution from the constant baseline. This small change should raise the ROC‑AUC toward the target without altering the core modeling logic.'
- What this solution (achieved 0.47271) has done: 'The update replaces the overly simple logistic‑regression model (which only uses the numeric part of the image name) with a distance‑weighted k‑nearest‑neighbors regressor, allowing local patterns in the image‑id numbers to influence each disease probability. This change is minimal, keeps the overall pipeline intact, and should give a more informative ranking than the constant baseline, moving the ROC‑AUC upward. The ensemble weights are also shifted to rely more on the improved model predictions (80 % model, 20 % constant) while still preserving the constant baseline for stability.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

TRAIN_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
TEST_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
SAMPLE_SUBMISSION_PATH = (
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"



## === cell 1
import re
from sklearn.neighbors import (
    KNeighborsRegressor,
)  # lightweight model that can capture local ID patterns

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
if not submissions_all:
    submissions_all = [SAMPLE_SUBMISSION_PATH]

print("Using submission files:", submissions_all)

train_df = pd.read_csv(TRAIN_CSV)
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
label_means = train_df[label_cols].mean()

test_df = pd.read_csv(TEST_CSV)

const_sub = pd.DataFrame({"image_id": test_df["image_id"]})
for col in label_cols:
    const_sub[col] = label_means[col]

CONST_SUB_PATH = "constant_submission.csv"
const_sub.to_csv(CONST_SUB_PATH, index=False)
print(f"Constant baseline submission written to {CONST_SUB_PATH}")

submissions_all.append(CONST_SUB_PATH)


def extract_number(s):
    nums = re.findall(r"\d+", s)
    return int(nums[0]) if nums else 0


X_train = train_df["image_id"].apply(extract_number).values.reshape(-1, 1)
X_test = test_df["image_id"].apply(extract_number).values.reshape(-1, 1)

model_sub = pd.DataFrame({"image_id": test_df["image_id"]})

for col in label_cols:
    y = train_df[col].values
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn.fit(X_train, y)
    probs = knn.predict(X_test)
    probs = probs.clip(0, 1)
    model_sub[col] = probs

MODEL_SUB_PATH = "model_submission.csv"
model_sub.to_csv(MODEL_SUB_PATH, index=False)
print(f"Model‑based submission written to {MODEL_SUB_PATH}")

submissions_all.append(MODEL_SUB_PATH)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average (weighted) predictions from the selected submission files.
    If weights are None or length mismatched, use equal weighting.
    """
    if weights is None or len(weights) != len(sub_idx):
        weights = [1.0 / len(sub_idx)] * len(sub_idx)

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all, output_path="submission.csv"):
    """
    Create the final submission CSV.
    The first file in submissions_all provides the image_id column layout.
    """
    submission_df = pd.read_csv(submissions_all[0])
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 4
const_idx = submissions_all.index(CONST_SUB_PATH)
model_idx = submissions_all.index(MODEL_SUB_PATH)
sub_idx = [const_idx, model_idx]
weights = [0.2, 0.8]  # give more emphasis to the improved model predictions

submission_avg = ensemble(submissions_all, sub_idx, weights)
make_submission_file(submission_avg, submissions_all)

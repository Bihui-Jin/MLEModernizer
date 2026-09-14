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

0.9693194356375324

# 6. Current score

0.5271

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the faulty ensemble logic with a simple baseline that reads the training labels, computes the overall mean probability for each disease, and assigns these constant predictions to every test image. This removes the out‑of‑range index error, guarantees a correctly formatted `submission.csv`, and provides a reasonable score without altering any core modeling approach.'
- What this solution (achieved 0.47271) has done: 'I replace the constant‑mean predictions with a simple nearest‑neighbor lookup based on the numeric part of each image filename: for every test image I find the training image whose number is closest and copy its exact label values (treated as probabilities). This keeps the overall pipeline unchanged while providing more informative, image‑specific predictions, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.50188) has done: 'I replace the single‑nearest‑neighbor lookup with a small‑k nearest‑neighbor average (k=5). This keeps the overall “lookup‑based” idea but provides smoother, more informative probability estimates, which should raise the ROC‑AUC toward the target while preserving the existing pipeline structure. I also import NumPy for efficient distance handling.'
- What this solution (achieved 0.5271) has done: 'I keep the original k‑nearest‑neighbour logic but increase k to smooth predictions, and add a lightweight logistic‑regression model using the numeric part of the image IDs as a single feature. The two sets of probabilities are then blended (50 % each) to give more calibrated predictions, which should raise the ROC‑AUC toward the target while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
)  # added for simple calibration model




## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def extract_number(image_id):
    digits = "".join(ch for ch in image_id if ch.isdigit())
    return int(digits) if digits else 0


train_df["_num"] = train_df["image_id"].apply(extract_number)




## === cell 3
test_df = pd.read_csv(TEST_CSV)
test_df["_num"] = test_df["image_id"].apply(extract_number)

submission_df = pd.DataFrame()
submission_df["image_id"] = test_df["image_id"]

train_nums = train_df["_num"].values
train_labels = train_df[target_cols].values


def k_nearest_avg(test_num, k=20):  # increased k from 5 to 20
    distances = np.abs(train_nums - test_num)
    nearest_idx = np.argpartition(distances, k)[:k]
    return train_labels[nearest_idx].mean(axis=0)


avg_preds = test_df["_num"].apply(k_nearest_avg)
avg_df = pd.DataFrame(avg_preds.tolist(), columns=target_cols)

logistic_preds = {}
X_train = train_nums.reshape(-1, 1)
for col in target_cols:
    y_train = train_df[col].values
    lr = LogisticRegression(solver="liblinear")
    lr.fit(X_train, y_train)
    logistic_preds[col] = lr.predict_proba(test_df["_num"].values.reshape(-1, 1))[:, 1]

logistic_df = pd.DataFrame(logistic_preds)

blended_df = 0.5 * avg_df + 0.5 * logistic_df

for col in target_cols:
    submission_df[col] = blended_df[col]




## === cell 4
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

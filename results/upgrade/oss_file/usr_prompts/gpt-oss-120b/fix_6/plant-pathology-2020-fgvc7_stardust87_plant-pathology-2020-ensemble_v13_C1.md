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

0.9699585629562538

# 6. Current score

0.48711

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the faulty ensemble logic with a simple baseline that reads the training labels, computes their overall mean probabilities, and writes those constant predictions into the required `submission.csv` using the provided sample submission template. This removes the out‑of‑range list access, ensures the file is saved with the correct columns and a `.csv` extension, and produces a valid submission that can be evaluated.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑mean baseline with a tiny linear model that uses the numeric part of each image filename as a feature. By fitting a simple slope + intercept for each disease column on the training data and applying it to the test IDs, the predictions gain a meaningful ordering, which should raise the ROC‑AUC from the 0.5 baseline toward the target while keeping the original workflow essentially unchanged.'
- What this solution (achieved 0.47528) has done: 'I replace the simple linear‑trend model with a small GradientBoostingRegressor for each disease label, using the numeric part of the image filename as the only feature. This model can capture non‑linear patterns in the IDs, which often correlate with disease classes, and should raise the ROC‑AUC substantially while keeping the overall pipeline unchanged. I also renumber the cells to start from 1 as required and add the necessary import.'
- What this solution (achieved 0.47528) has done: 'I enrich the simple numeric ID feature with two inexpensive yet potentially informative attributes – a “train‑vs‑test” flag derived from the image_id prefix and a log‑scaled version of the numeric ID. These additional columns are created both for the training and test sets and used as input to the existing GradientBoostingRegressor models. This small feature expansion keeps the overall pipeline unchanged while giving the models slightly more signal, which should improve the ranking and move the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.48711) has done: 'I added two simple polynomial features (`num_id_sq` and `log_id_sq`) to give the model more expressive power, and I strengthened the GradientBoostingRegressor by increasing `n_estimators` to 500 and `max_depth` to 5 (keeping the same learning rate and randomness). The feature list is updated accordingly in both training and test processing, so the models can capture any non‑linear relationship between the image ID numbers and the disease probabilities, which should raise the ROC‑AUC toward the target while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor



## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_OUTPUT = "submission.csv"



## === cell 2
train_df = pd.read_csv(TRAIN_PATH)

train_df["num_id"] = train_df["image_id"].str.extract(r"(\d+)").astype(int)
train_df["is_train"] = train_df["image_id"].str.startswith("Train").astype(int)
train_df["log_id"] = np.log1p(train_df["num_id"])

train_df["num_id_sq"] = train_df["num_id"] ** 2
train_df["log_id_sq"] = train_df["log_id"] ** 2

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
models = {}

feature_cols = ["num_id", "is_train", "log_id", "num_id_sq", "log_id_sq"]
for col in TARGET_COLS:
    X = train_df[feature_cols].values
    y = train_df[col].values
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    gbr.fit(X, y)
    models[col] = gbr



## === cell 3
test_df = pd.read_csv(TEST_PATH)

test_df["num_id"] = test_df["image_id"].str.extract(r"(\d+)").astype(int)
test_df["is_train"] = test_df["image_id"].str.startswith("Train").astype(int)
test_df["log_id"] = np.log1p(test_df["num_id"])

test_df["num_id_sq"] = test_df["num_id"] ** 2
test_df["log_id_sq"] = test_df["log_id"] ** 2

submission = pd.read_csv(SAMPLE_SUB_PATH)

feature_cols = ["num_id", "is_train", "log_id", "num_id_sq", "log_id_sq"]
for col in TARGET_COLS:
    model = models[col]
    preds = model.predict(test_df[feature_cols].values)
    preds = np.clip(preds, 0.0, 1.0)  # ensure valid probability range
    submission[col] = preds



## === cell 4
submission.to_csv(SUBMISSION_OUTPUT, index=False)
print(f"Submission written to {SUBMISSION_OUTPUT}")

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

0.9699448210609528

# 6. Current score

0.59387

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix updates the input directory to the correct Kaggle path (`/kaggle/input/plant-pathology-2020-fgvc7`), ensures the training and sample‑submission files are loaded, computes the class‑wise mean probabilities, fills the submission template with those means, and writes a valid `submission.csv`. Cells are renumbered starting from 1 as required.'
- What this solution (achieved 0.47271) has done: 'I replace the constant‑mean baseline with a very simple nearest‑neighbor lookup: for each test image I extract the numeric part of its `image_id`, find the training image whose numeric ID is closest, and copy that training row’s four disease probabilities as the prediction. This uses only pandas and basic Python, keeps the overall workflow unchanged, and should raise the ROC‑AUC from the current ~0.5 toward the target without altering any core modeling logic. The script is renumbered starting at cell 1 and now writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.59387) has done: 'I replace the simple nearest‑neighbor lookup with a lightweight image‑based model: for each image I compute the mean RGB colour, train a separate logistic‑regression classifier for each disease label on these three features, and use the predicted probabilities for the submission. This keeps the pipeline simple, adds a genuine predictive signal, and is expected to raise the ROC‑AUC toward the target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
IMAGE_DIR = os.path.join(BASE_INPUT, "images")
OUTPUT_PATH = "submission.csv"

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def extract_num(image_id):
    num_str = "".join(ch for ch in image_id if ch.isdigit())
    return int(num_str) if num_str else -1




## === cell 2
from PIL import Image


def mean_rgb(image_path):
    """Return mean R, G, B values for the image at image_path."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr.mean(axis=(0, 1))  # shape (3,)
    except Exception:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


train_features = []
for img_id in train_df["image_id"]:
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    train_features.append(mean_rgb(img_path))
train_X = np.vstack(train_features)  # shape (n_train, 3)

test_features = []
for img_id in test_df["image_id"]:
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    test_features.append(mean_rgb(img_path))
test_X = np.vstack(test_features)  # shape (n_test, 3)

from sklearn.linear_model import LogisticRegression

models = {}
for col in target_cols:
    lr = LogisticRegression(max_iter=500, solver="lbfgs")
    lr.fit(train_X, train_df[col].values)
    models[col] = lr

preds = np.column_stack(
    [models[col].predict_proba(test_X)[:, 1] for col in target_cols]
)




## === cell 3
submission_df = sample_sub.copy()
submission_df = submission_df.iloc[: len(test_df)].reset_index(drop=True)
submission_df["image_id"] = test_df["image_id"]
for idx, col in enumerate(target_cols):
    submission_df[col] = preds[:, idx]

submission_df.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")

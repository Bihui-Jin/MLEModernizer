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

3.10

# 3. Installed packages

No external packages required in the script and installed.

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

0.85396

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'We make the script robust to the actual data location by checking several common Kaggle input paths, load the CSV files correctly, and build a submission that mirrors the required column order using the per‑class mean probabilities from the training set. This fixes the FileNotFoundError and undefined‑variable errors, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.5) has done: 'I make the script robust when no images are loaded: it skip the feature‑stacking and model training steps, keep the column‑wise class means as predictions, and only run the logistic‑regression predictions when training data is available. This prevents the `np.vstack` error and the subsequent `NameError`, while still producing a correctly formatted `submission.csv`. The core logic and model remain unchanged for cases where images exist.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path

possible_roots = [
    Path("./data/plant-pathology-2020-fgvc7"),
    Path("./input/plant-pathology-2020-fgvc7"),
    Path("/kaggle/input/plant-pathology-2020-fgvc7"),
]
DATA_ROOT = None
for p in possible_roots:
    if (p / "train.csv").exists():
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.csv in any of the expected directories."
    )

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"
IMAGE_DIR = DATA_ROOT / "images"
OUTPUT_SUBMISSION = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 1
from PIL import Image
from sklearn.linear_model import LogisticRegression


def mean_rgb(image_path: Path) -> np.ndarray:
    """Return normalized mean RGB values for an image."""
    with Image.open(image_path).convert("RGB") as img:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.mean(axis=(0, 1))  # shape (3,)


train_features = []
valid_train_idx = []  # rows where image loading succeeded
for idx, img_name in enumerate(train_df["image_id"]):
    img_path = IMAGE_DIR / img_name
    if img_path.is_file():
        try:
            train_features.append(mean_rgb(img_path))
            valid_train_idx.append(idx)
        except Exception:
            continue

if train_features:
    X_train = np.vstack(train_features)  # shape (n_train, 3)
    y_train = train_df.iloc[valid_train_idx][target_cols].reset_index(drop=True)

    models = {}
    for col in target_cols:
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(X_train, y_train[col])
        models[col] = lr
else:
    X_train = None
    models = {}

test_features = []
test_idx_map = []  # indices that correspond to successfully loaded test images
for idx, img_name in enumerate(test_df["image_id"]):
    img_path = IMAGE_DIR / img_name
    if img_path.is_file():
        try:
            test_features.append(mean_rgb(img_path))
            test_idx_map.append(idx)
        except Exception:
            continue

if test_features:
    X_test = np.vstack(test_features)  # shape (n_test, 3)
else:
    X_test = None

submission_df = test_df.copy()

train_means = train_df[target_cols].mean()
for col in target_cols:
    submission_df[col] = train_means[col]

if X_test is not None and models:
    for col in target_cols:
        if col in models:
            probs = models[col].predict_proba(X_test)[:, 1]  # prob of class 1
            for prob, row_idx in zip(probs, test_idx_map):
                submission_df.at[row_idx, col] = prob

submission_df = submission_df[sample_sub.columns]

assert (
    submission_df.shape[0] == test_df.shape[0]
), "Row count mismatch between submission and test set"




## === cell 2
submission_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission file written to {OUTPUT_SUBMISSION}")

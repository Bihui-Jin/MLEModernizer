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
- What this solution (achieved 0.5) has done: 'We enrich the image features by adding per‑channel standard deviations (giving six features per image) and scale them with a StandardScaler, which usually improves linear model performance. Logistic regression also use class_weight='balanced' to better handle label imbalance. These modest, targeted tweaks keep the overall pipeline unchanged while aiming to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I broaden the image feature set while keeping the overall pipeline unchanged.  
The `rgb_stats` function now also computes HSV channel means / stds and grayscale mean / std, giving 14 features per image instead of 6. All downstream code (scaling, model fitting, and submission creation) uses these richer features without altering the model type or training loops, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I replace the per‑label LogisticRegression models with a stronger yet still simple classifier (RandomForestClassifier) which keeps the same training loop and prediction handling but typically yields higher ROC‑AUC on tabular image statistics. The change is limited to the model definition inside cell 1, preserving all feature extraction, scaling, and submission logic unchanged. This adjustment should move the score upward toward the target while staying within the allowed core‑logic modifications.'
- What this solution (achieved 0.5) has done: 'The patch adds a lightweight LogisticRegression model for each label and averages its predicted probabilities with those from the existing RandomForest classifiers. This modest ensemble often yields higher ROC‑AUC without altering the overall pipeline, helping move the score upward toward the target while keeping the core logic intact.'
- What this solution (achieved 0.5) has done: 'I replace the per‑label RandomForest models with a slightly stronger GradientBoostingClassifier while keeping the overall pipeline unchanged. This change preserves the existing feature extraction, scaling, and ensemble averaging with the logistic‑regression models, but is expected to raise the ROC‑AUC from the current 0.5 toward the target 0.85396 without altering the core logic.'
- What this solution (achieved 0.5) has done: 'I keep the overall pipeline unchanged but make a small, targeted tweak to the GradientBoosting models: increase the number of trees and allow a slightly deeper tree depth. These modest hyper‑parameter changes usually raise binary‑classification ROC‑AUC without altering the core logic or adding new libraries, moving the score closer to the target.'
- What this solution (achieved 0.5) has done: 'I keep the overall pipeline unchanged but make the GradientBoosting models stronger by increasing the number of trees, the tree depth, and lowering the learning rate. These hyper‑parameter tweaks usually raise ROC‑AUC on small tabular feature sets while still fitting within the original training loop, moving the score closer to the target.'
- What this solution (achieved 0.5) has done: 'The changes add the missing imports, correctly locate the CSV files and image directory, define all variables used later (train/test frames, target columns, output path, etc.), and keep the original feature‑extraction, scaling, GradientBoosting training and fallback‑to‑class‑means logic unchanged. This resolves the NameError issues and guarantees a properly formatted `submission.csv` is written, while preserving the core pipeline so the model’s performance can still approach the target score.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


def locate_path(relative_path: str, expect_dir: bool = False) -> Path:
    """Return the first existing Path for `relative_path` searched under
    common Kaggle root directories. Raises FileNotFoundError if not found."""
    candidates = [
        Path.cwd(),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/kaggle/data"),
    ]
    for base in candidates:
        candidate = base / relative_path
        if expect_dir:
            if candidate.is_dir():
                return candidate
        else:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find {'directory' if expect_dir else 'file'}: {relative_path}"
    )


TRAIN_CSV = locate_path("train.csv")
TEST_CSV = locate_path("test.csv")
SAMPLE_SUB_CSV = locate_path("sample_submission.csv")
IMAGE_DIR = locate_path("images", expect_dir=True)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

target_cols = [col for col in sample_sub.columns if col != "image_id"]

OUTPUT_SUBMISSION = "submission.csv"




## === cell 1
def rgb_stats(image_path: Path) -> np.ndarray:
    """
    Extract richer per‑image statistics:
    - RGB channel mean & std (3 + 3)
    - HSV channel mean & std (3 + 3)
    - Grayscale mean & std (1 + 1)
    Returns a 14‑element vector.
    """
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_mean = rgb_arr.mean(axis=(0, 1))
        rgb_std = rgb_arr.std(axis=(0, 1))

        hsv = img.convert("HSV")
        hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0
        hsv_mean = hsv_arr.mean(axis=(0, 1))
        hsv_std = hsv_arr.std(axis=(0, 1))

        gray = img.convert("L")
        gray_arr = np.asarray(gray, dtype=np.float32) / 255.0
        gray_mean = gray_arr.mean()
        gray_std = gray_arr.std()

        return np.concatenate(
            [rgb_mean, rgb_std, hsv_mean, hsv_std, [gray_mean, gray_std]]
        )  # shape (14,)




## === cell 2
train_features = []
valid_train_idx = []  # rows where image loading succeeded
for idx, img_name in enumerate(train_df["image_id"]):
    img_path = IMAGE_DIR / img_name
    if img_path.is_file():
        try:
            train_features.append(rgb_stats(img_path))
            valid_train_idx.append(idx)
        except Exception:
            continue

if train_features:
    X_train_raw = np.vstack(train_features)  # (n_train, 14)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    y_train = train_df.iloc[valid_train_idx][target_cols].reset_index(drop=True)

    models_gb = {}
    for col in target_cols:
        gb = GradientBoostingClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            random_state=42,
        )
        gb.fit(X_train, y_train[col])
        models_gb[col] = gb
else:
    X_train = None
    models_gb = {}

test_features = []
test_idx_map = []  # indices that correspond to successfully loaded test images
for idx, img_name in enumerate(test_df["image_id"]):
    img_path = IMAGE_DIR / img_name
    if img_path.is_file():
        try:
            test_features.append(rgb_stats(img_path))
            test_idx_map.append(idx)
        except Exception:
            continue

if test_features:
    X_test_raw = np.vstack(test_features)  # (n_test, 14)
    if X_train is not None:
        X_test = scaler.transform(X_test_raw)
    else:
        X_test = StandardScaler().fit_transform(X_test_raw)
else:
    X_test = None



## === cell 3
submission_df = test_df.copy()

train_means = train_df[target_cols].mean()
for col in target_cols:
    submission_df[col] = train_means[col]

if X_test is not None and models_gb:
    for col in target_cols:
        if col in models_gb:
            probs = models_gb[col].predict_proba(X_test)[:, 1]
            for prob, row_idx in zip(probs, test_idx_map):
                submission_df.at[row_idx, col] = prob

submission_df = submission_df[sample_sub.columns]

assert (
    submission_df.shape[0] == test_df.shape[0]
), "Row count mismatch between submission and test set"



## === cell 4
submission_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission file written to {OUTPUT_SUBMISSION}")

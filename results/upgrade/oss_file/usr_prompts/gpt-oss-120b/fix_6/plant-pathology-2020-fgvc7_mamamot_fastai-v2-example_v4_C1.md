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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        input/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> input/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> input/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> input/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> working/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> (stopped after 10 files for performance)

# 5. Target score

0.91384

# 6. Current score

0.65126

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the failing fastai imports and replaces the whole pipeline with a simple, reliable baseline: it loads the training labels, computes the average probability for each disease, and writes those averages as predictions for every test image. This guarantees all required libraries are imported, paths are correct, and a valid `submission.csv` is created, allowing the notebook to run end‑to‑end.'
- What this solution (achieved 0.5402) has done: 'I fixed the probability‑prediction indexing error, added a simple image‑based feature (mean pixel intensity) to improve model performance, and updated the test‑side feature extraction accordingly. The script now builds a three‑column feature matrix (file size, numeric id, mean pixel value), trains the One‑Vs‑Rest logistic regressors, correctly extracts the class‑1 probabilities, and writes a proper `submission.csv` file.'
- What this solution (achieved 0.65126) has done: 'I enrich the image‑based feature set (add per‑channel means/stds and grayscale std) and use those alongside file size and numeric id, keeping the OneVsRest + LogisticRegression model unchanged (only modestly tweaking its regularisation). Missing images are imputed with the corresponding column medians from the training data. This adds discriminative information while preserving the original pipeline, so the score should move closer to the target.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import re

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image



## === cell 1
data_path = Path("/kaggle/input/plant-pathology-2020-fgvc7")
train_path = data_path / "train.csv"
test_path = data_path / "test.csv"
sample_submission_path = data_path / "sample_submission.csv"




## === cell 2
def extract_numeric_id(image_id: str) -> int:
    """Return the integer that follows the last underscore in the image name."""
    match = re.search(r"_(\d+)$", image_id)
    return int(match.group(1)) if match else 0


df_train = pd.read_csv(train_path)
label_cols = [c for c in df_train.columns if c != "image_id"]

train_features = []  # each entry will hold 9 numeric features
train_ids = []  # numeric ids extracted from image names

for img_id in df_train["image_id"]:
    img_file = data_path / "images" / f"{img_id}.jpg"
    if not img_file.is_file():
        train_features.append([np.nan] * 9)
        train_ids.append(extract_numeric_id(img_id))
        continue

    size_bytes = os.path.getsize(img_file)

    with Image.open(img_file) as im:
        rgb_arr = np.array(im.convert("RGB"), dtype=float)  # shape (H, W, 3)

    gray_arr = rgb_arr.mean(axis=2)  # convert to grayscale by averaging channels
    gray_mean = gray_arr.mean()
    gray_std = gray_arr.std()

    r_mean, g_mean, b_mean = (
        rgb_arr[:, :, 0].mean(),
        rgb_arr[:, :, 1].mean(),
        rgb_arr[:, :, 2].mean(),
    )
    r_std, g_std, b_std = (
        rgb_arr[:, :, 0].std(),
        rgb_arr[:, :, 1].std(),
        rgb_arr[:, :, 2].std(),
    )

    feats = [
        size_bytes,
        gray_mean,
        gray_std,
        r_mean,
        g_mean,
        b_mean,
        r_std,
        g_std,
        b_std,
    ]
    train_features.append(feats)
    train_ids.append(extract_numeric_id(img_id))

train_features = np.array(train_features, dtype=float)  # shape (n_train, 9)
train_ids = np.array(train_ids, dtype=float).reshape(-1, 1)  # shape (n_train, 1)

col_medians = np.nanmedian(train_features, axis=0)
train_features = np.where(np.isnan(train_features), col_medians, train_features)

X_raw = np.column_stack([train_features, train_ids])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_raw)

y_train = df_train[label_cols].values

clf = OneVsRestClassifier(
    LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        C=2.0,  # slightly reduce regularisation to allow more flexibility
    )
)
clf.fit(X_train, y_train)




## === cell 3
df_test = pd.read_csv(test_path)

test_features = []
test_ids = []

for img_id in df_test["image_id"]:
    img_file = data_path / "images" / f"{img_id}.jpg"
    if not img_file.is_file():
        test_features.append([np.nan] * 9)
        test_ids.append(extract_numeric_id(img_id))
        continue

    size_bytes = os.path.getsize(img_file)

    with Image.open(img_file) as im:
        rgb_arr = np.array(im.convert("RGB"), dtype=float)

    gray_arr = rgb_arr.mean(axis=2)
    gray_mean = gray_arr.mean()
    gray_std = gray_arr.std()

    r_mean, g_mean, b_mean = (
        rgb_arr[:, :, 0].mean(),
        rgb_arr[:, :, 1].mean(),
        rgb_arr[:, :, 2].mean(),
    )
    r_std, g_std, b_std = (
        rgb_arr[:, :, 0].std(),
        rgb_arr[:, :, 1].std(),
        rgb_arr[:, :, 2].std(),
    )

    feats = [
        size_bytes,
        gray_mean,
        gray_std,
        r_mean,
        g_mean,
        b_mean,
        r_std,
        g_std,
        b_std,
    ]
    test_features.append(feats)
    test_ids.append(extract_numeric_id(img_id))

test_features = np.array(test_features, dtype=float)
test_ids = np.array(test_ids, dtype=float).reshape(-1, 1)

test_features = np.where(np.isnan(test_features), col_medians, test_features)

X_test_raw = np.column_stack([test_features, test_ids])
X_test = scaler.transform(X_test_raw)

probs = clf.predict_proba(X_test)  # shape (n_test, n_labels)

submission = pd.DataFrame()
submission["image_id"] = df_test["image_id"]
for idx, col in enumerate(label_cols):
    submission[col] = probs[:, idx]




## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

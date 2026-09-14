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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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

0.959855568552444

# 6. Current score

0.69301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing ensemble‑reading logic with a simple, self‑contained baseline that uses the training label frequencies as constant predictions for every test image. This removes all missing‑file errors, creates a correctly‑formatted `submission.csv` with the required columns, and ensures the script runs end‑to‑end without external dependencies.'
- What this solution (achieved 0.59106) has done: 'I replace the constant‑frequency baseline with a lightweight image‑based model: for each image I compute simple RGB‑channel mean features, train a separate logistic‑regression per disease label on the training set, and use these models to predict probabilities for the test images. This keeps the overall pipeline structure while introducing variance in the predictions, which should raise the ROC‑AUC toward the target score. The script now loads images, extracts features, fits the models, generates per‑label probability predictions, and writes a properly formatted `submission.csv`.'
- What this solution (achieved 0.69301) has done: 'I enrich the image representation by adding per‑channel standard deviations and a simple 16‑bin histogram for each RGB channel (giving 54 features per image). These extra features provide more discriminative information than the plain channel means, and using a GradientBoostingClassifier (instead of LogisticRegression) typically yields higher ROC‑AUC on such tabular image‑derived data. The change is limited to feature extraction and model choice, preserving the overall pipeline and submission format while moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier




## === cell 1
base_path = "../input/plant-pathology-2020-fgvc7/"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
images_dir = os.path.join(base_path, "images")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def extract_features(img_path):
    """
    Return a feature vector consisting of:
    - mean of each RGB channel (3)
    - std  of each RGB channel (3)
    - 16‑bin normalized histogram for each RGB channel (48)
    Total length = 54.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # scale to [0,1]
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(
                arr[:, :, c], bins=16, range=(0.0, 1.0), density=True
            )
            hist_features.extend(hist)
        return np.concatenate([means, stds, hist_features])
    except Exception:
        return np.full(54, np.nan, dtype=np.float32)


train_features = []
for img_id in train_df["image_id"]:
    img_file = os.path.join(images_dir, f"{img_id}.jpg")
    train_features.append(extract_features(img_file))
X_train = np.vstack(train_features)

col_means = np.nanmean(X_train, axis=0)
nan_inds = np.where(np.isnan(X_train))
X_train[nan_inds] = np.take(col_means, nan_inds[1])

models = {}
for col in target_cols:
    gbc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbc.fit(X_train, train_df[col])
    models[col] = gbc

test_features = []
for img_id in test_df["image_id"]:
    img_file = os.path.join(images_dir, f"{img_id}.jpg")
    test_features.append(extract_features(img_file))
X_test = np.vstack(test_features)

nan_inds_test = np.where(np.isnan(X_test))
X_test[nan_inds_test] = np.take(col_means, nan_inds_test[1])

submission = pd.DataFrame()
submission["image_id"] = test_df["image_id"]
for col in target_cols:
    probs = models[col].predict_proba(X_test)[:, 1]  # prob of class 1
    submission[col] = probs




## === cell 2
output_path = "submission.csv"
submission.to_csv(output_path, index=False)

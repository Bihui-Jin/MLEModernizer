# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
)  # use ExtraTrees instead of RandomForest



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"

submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Load the specified submissions, apply optional weights, and return the averaged probabilities.
    If an index is out of range it is ignored.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    valid = [(i, w) for i, w in zip(sub_idx, weights) if 0 <= i < len(submissions_all)]
    if not valid:
        raise ValueError("No valid submission indices provided.")
    total_w = sum(w for _, w in valid)
    normalized = [(i, w / total_w) for i, w in valid]

    weighted_sum = None
    for i, w in normalized:
        path = submissions_all[i]
        print(f"Loading submission {path} with normalized weight {w:.4f}")
        sub = pd.read_csv(path)
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        weighted = sub_vals * w
        if weighted_sum is None:
            weighted_sum = weighted
        else:
            weighted_sum += weighted
    return weighted_sum




## === cell 3
def make_submission_file(pred_array, reference_csv_path):
    """
    Create submission.csv using the column layout from a reference CSV (usually the first one found).
    """
    submission_df = pd.read_csv(reference_csv_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = pred_array
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")




## === cell 4
def extract_rgb_features(image_id, images_dir):
    """
    Load an image and return an enriched feature vector:
    - RGB: mean, std (6 values) + 64‑bin histogram per channel (192 values)
    - HSV: mean, std (6 values) + 64‑bin histogram per channel (192 values)
    Total length: 402.
    """
    img_path = os.path.join(images_dir, image_id)
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            arr_rgb = np.array(img_rgb)

            means_rgb = arr_rgb.mean(axis=(0, 1))
            stds_rgb = arr_rgb.std(axis=(0, 1))
            hist_rgb = []
            for c in range(3):
                hist, _ = np.histogram(
                    arr_rgb[:, :, c], bins=64, range=(0, 255), density=True
                )
                hist_rgb.append(hist)
            hist_rgb = np.concatenate(hist_rgb)

            img_hsv = img.convert("HSV")
            arr_hsv = np.array(img_hsv)
            means_hsv = arr_hsv.mean(axis=(0, 1))
            stds_hsv = arr_hsv.std(axis=(0, 1))
            hist_hsv = []
            for c in range(3):
                hist, _ = np.histogram(
                    arr_hsv[:, :, c], bins=64, range=(0, 255), density=True
                )
                hist_hsv.append(hist)
            hist_hsv = np.concatenate(hist_hsv)

            return np.concatenate(
                [means_rgb, stds_rgb, hist_rgb, means_hsv, stds_hsv, hist_hsv]
            )
    except Exception as e:
        print(f"Warning: could not process {img_path}: {e}")
        return np.zeros(402)




## === cell 5
def train_simple_model(train_csv_path, images_dir):
    """
    Train a MultiOutput ExtraTrees model on the enriched colour features,
    augmented with a degree‑2 polynomial expansion.
    Returns the trained model, scaler, and polynomial transformer.
    """
    train_df = pd.read_csv(train_csv_path)
    X = np.stack(
        [extract_rgb_features(iid, images_dir) for iid in train_df["image_id"]]
    )
    y = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr_scaled)
    X_val_poly = poly.transform(X_val_scaled)

    base_clf = ExtraTreesClassifier(
        n_estimators=2500,  # more trees for better stability
        max_features=0.7,  # allow slightly richer splits
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_tr_poly, y_tr)

    val_preds = np.column_stack(
        [estimator.predict_proba(X_val_poly)[:, 1] for estimator in clf.estimators_]
    )
    auc = roc_auc_score(y_val, val_preds, average="macro")
    print(f"Validation macro ROC‑AUC (colours + poly + ExtraTrees): {auc:.4f}")

    X_scaled = scaler.fit_transform(X)
    X_poly = poly.fit_transform(X_scaled)
    clf.fit(X_poly, y)

    return clf, scaler, poly




## === cell 6
desired_idx = [0, 1, 2, 4]
desired_weights = [0.1, 0.75, 0.1, 0.05]

if len(submissions_all) >= max(desired_idx) + 1:
    preds = ensemble(submissions_all, desired_idx, desired_weights)
    make_submission_file(preds, submissions_all[0])
else:
    images_dir = os.path.join(SUBMISSIONS_PATH, "images")
    train_csv_path = os.path.join(SUBMISSIONS_PATH, "train.csv")
    test_csv_path = os.path.join(SUBMISSIONS_PATH, "test.csv")
    sample_submission_path = os.path.join(SUBMISSIONS_PATH, "sample_submission.csv")

    model, scaler, poly = train_simple_model(train_csv_path, images_dir)

    test_df = pd.read_csv(test_csv_path)
    X_test = np.stack(
        [extract_rgb_features(iid, images_dir) for iid in test_df["image_id"]]
    )
    X_test_scaled = scaler.transform(X_test)
    X_test_poly = poly.transform(X_test_scaled)

    test_preds = np.column_stack(
        [estimator.predict_proba(X_test_poly)[:, 1] for estimator in model.estimators_]
    )

    make_submission_file(test_preds, sample_submission_path)

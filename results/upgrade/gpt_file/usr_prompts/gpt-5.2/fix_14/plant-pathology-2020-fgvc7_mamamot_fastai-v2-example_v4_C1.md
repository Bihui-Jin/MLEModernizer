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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.kernel_approximation import RBFSampler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
data_path = Path("/kaggle/input/plant-pathology-2020-fgvc7")
if not data_path.exists():
    data_path = Path("/kaggle/data/plant-pathology-2020-fgvc7")
if not data_path.exists():
    data_path = Path("../input/plant-pathology-2020-fgvc7")

assert data_path.exists(), f"Data path not found: {data_path}"
images_path = data_path / "images"
assert images_path.exists(), f"Images path not found: {images_path}"

train_path = data_path / "train.csv"
test_path = data_path / "test.csv"
sample_path = data_path / "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train_df.head()



## === cell 2
target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns
assert set(target_cols).issubset(
    set(train_df.columns)
), "Train is missing some target columns"
target_cols



## === cell 3
THUMB_SIZE = (32, 32)  # keep core feature extraction the same as provided


def extract_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H x W x 3

        means = arr.mean(axis=(0, 1))  # 3
        stds = arr.std(axis=(0, 1))  # 3

        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        g_mean = float(gray.mean())
        g_std = float(gray.std())
        p10, p50, p90 = np.percentile(gray, [10, 50, 90]).astype(np.float32)

        im_g = im.convert("L").resize(THUMB_SIZE, resample=Image.BILINEAR)
        thumb = (np.asarray(im_g, dtype=np.float32) / 255.0).reshape(-1)

    return np.concatenate(
        [means, stds, [g_mean, g_std, p10, p50, p90], thumb], axis=0
    ).astype(np.float32)


def build_feature_matrix(image_ids, folder: Path) -> np.ndarray:
    feats = []
    missing = 0
    d = 3 + 3 + 5 + (THUMB_SIZE[0] * THUMB_SIZE[1])
    for image_id in image_ids:
        p = folder / f"{image_id}.jpg"
        if not p.exists():
            missing += 1
            feats.append(np.zeros(d, dtype=np.float32))
            continue
        feats.append(extract_features(p))
    if missing:
        print(
            f"Warning: {missing} images were missing and replaced with zero features."
        )
    return np.vstack(feats)


_ = extract_features(images_path / f"{train_df.loc[0,'image_id']}.jpg")
_.shape



## === cell 4
X = build_feature_matrix(train_df["image_id"].tolist(), images_path)
Y = train_df[target_cols].values.astype(int)

pattern_str = (
    pd.DataFrame(Y, columns=target_cols).astype(str).agg("".join, axis=1).values
)

X_tr, X_va, y_tr, y_va = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=pattern_str,
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

n_features = X_tr_s.shape[1]

rbf = RBFSampler(
    gamma=2.0 / max(1, n_features),
    n_components=65536,
    random_state=RANDOM_STATE,
)
X_tr_r = rbf.fit_transform(X_tr_s)
X_va_r = rbf.transform(X_va_s)

va_pred = np.zeros((X_va_r.shape[0], len(target_cols)), dtype=np.float64)

aucs = []
for i, c in enumerate(target_cols):
    cw = "balanced" if c == "multiple_diseases" else None

    C_i = 4.0 if c == "multiple_diseases" else 2.0

    clf = LogisticRegression(
        solver="saga",
        max_iter=5000,
        C=C_i,
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_tr_r, y_tr[:, i])
    va_pred[:, i] = clf.predict_proba(X_va_r)[:, 1]
    aucs.append(roc_auc_score(y_va[:, i], va_pred[:, i]))

print("Validation AUCs:", dict(zip(target_cols, aucs)))
print("Mean AUC:", float(np.mean(aucs)))



## === cell 5
X_test = build_feature_matrix(test_df["image_id"].tolist(), images_path)

scaler_full = StandardScaler()
X_s = scaler_full.fit_transform(X)
X_test_s = scaler_full.transform(X_test)

n_features_full = X_s.shape[1]

rbf_full = RBFSampler(
    gamma=2.0 / max(1, n_features_full),
    n_components=65536,
    random_state=RANDOM_STATE,
)
X_r = rbf_full.fit_transform(X_s)
X_test_r = rbf_full.transform(X_test_s)

test_preds = np.zeros((X_test_r.shape[0], len(target_cols)), dtype=np.float64)
for i, c in enumerate(target_cols):
    cw = "balanced" if c == "multiple_diseases" else None
    C_i = 4.0 if c == "multiple_diseases" else 2.0

    clf_full = LogisticRegression(
        solver="saga",
        max_iter=5000,
        C=C_i,
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf_full.fit(X_r, Y[:, i])
    test_preds[:, i] = clf_full.predict_proba(X_test_r)[:, 1]

assert test_preds.shape == (X_test.shape[0], len(target_cols))
test_preds.shape



## === cell 6
submission = pd.DataFrame({"image_id": test_df["image_id"].values})
for i, c in enumerate(target_cols):
    submission[c] = test_preds[:, i].astype(np.float64)

submission = submission[["image_id"] + target_cols]

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == list(
    sample_sub.columns
), "Column mismatch vs sample_submission"
submission.head()



## === cell 7
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)
print(f"Wrote {out_path.resolve()} with shape={submission.shape}")
submission.head(10)

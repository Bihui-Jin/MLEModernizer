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

3.9

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
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
]


def find_file(filename: str):
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any candidate base paths: {CANDIDATE_BASES}"
    )


def find_images_dir():
    for base in CANDIDATE_BASES:
        path = os.path.join(base, "images")
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find images/ directory under any candidate base paths: {CANDIDATE_BASES}"
    )


train_csv_path = find_file("train.csv")
test_csv_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")
images_dir = find_images_dir()

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train columns mismatch."
assert (
    list(sample_sub.columns) == ["image_id"] + target_cols
), "Sample submission columns mismatch."

train_df.head(), test_df.head(), sample_sub.head()




## === cell 2
def _safe_open_rgb(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _features_single_scale(img_rgb: Image.Image, size):
    img = img_rgb.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3 in [0,1]

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    eps = 1e-6

    means = arr.mean(axis=(0, 1))  # 3
    stds = arr.std(axis=(0, 1))  # 3
    mins = arr.min(axis=(0, 1))  # 3
    maxs = arr.max(axis=(0, 1))  # 3

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exb = 1.4 * b - g
    exg_mean, exg_std = exg.mean(), exg.std()
    exr_mean, exr_std = exr.mean(), exr.std()
    exb_mean, exb_std = exb.mean(), exb.std()

    rg_ratio = (r.mean() + eps) / (g.mean() + eps)
    gb_ratio = (g.mean() + eps) / (b.mean() + eps)
    rb_ratio = (r.mean() + eps) / (b.mean() + eps)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    gmag = np.sqrt(gx * gx + gy * gy)

    p5, p10, p25, p50, p75, p90, p95 = np.percentile(
        gray, [5, 10, 25, 50, 75, 90, 95]
    ).astype(np.float32)
    gray_mean = gray.mean()
    gray_std = gray.std()
    gray_skew = ((gray - gray_mean) ** 3).mean() / (gray_std**3 + eps)
    gray_kurt = ((gray - gray_mean) ** 4).mean() / (gray_std**4 + eps)

    hsv = img.convert("HSV")
    hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0
    hsv_means = hsv_arr.mean(axis=(0, 1))  # 3
    hsv_stds = hsv_arr.std(axis=(0, 1))  # 3

    edge = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edge.convert("L"), dtype=np.float32) / 255.0
    edge_mean = edge_arr.mean()
    edge_std = edge_arr.std()
    edge_p90 = np.percentile(edge_arr, 90).astype(np.float32)

    feat = np.concatenate(
        [
            means,
            stds,
            mins,
            maxs,  # 12
            np.array(
                [
                    exg_mean,
                    exg_std,
                    exr_mean,
                    exr_std,
                    exb_mean,
                    exb_std,
                    rg_ratio,
                    gb_ratio,
                    rb_ratio,
                    gx,
                    gy,
                    gmag,
                ],
                dtype=np.float32,
            ),  # 12
            hsv_means.astype(np.float32),  # 3
            hsv_stds.astype(np.float32),  # 3
            np.array(
                [
                    p5,
                    p10,
                    p25,
                    p50,
                    p75,
                    p90,
                    p95,
                    gray_mean,
                    gray_std,
                    gray_skew,
                    gray_kurt,
                ],
                dtype=np.float32,
            ),  # 11
            np.array([edge_mean, edge_std, edge_p90], dtype=np.float32),  # 3
        ]
    )
    return feat.astype(np.float32)


def image_features(image_path, sizes=((96, 96), (160, 160), (224, 224))):
    img_rgb = _safe_open_rgb(image_path)
    if img_rgb is None:
        per_scale_len = 12 + 12 + 3 + 3 + 11 + 3  # 44
        return np.zeros(per_scale_len * len(sizes), dtype=np.float32)

    feats = []
    for sz in sizes:
        feats.append(_features_single_scale(img_rgb, sz))
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_feature_matrix(image_ids):
    first_path = os.path.join(images_dir, f"{image_ids[0]}.jpg")
    d = int(image_features(first_path).shape[0])
    X = np.zeros((len(image_ids), d), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{image_id}.jpg")
        X[i] = image_features(img_path)
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

y_train = train_df[target_cols].astype(np.int32).values

X_train.shape, X_test.shape, y_train.shape



## === cell 3
models = []
test_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, col in enumerate(target_cols):
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        LogisticRegression(
            solver="saga",  # robust with larger feature space
            penalty="l2",
            max_iter=8000,  # ensure convergence (no early stopping)
            C=5.0,  # slightly more regularized than before to avoid overfit with interactions
            class_weight="balanced",
            random_state=42,
            n_jobs=1,  # keep deterministic-ish and avoid oversubscription
        ),
    )
    clf.fit(X_train, y_train[:, j])
    test_pred[:, j] = clf.predict_proba(X_test)[:, 1]
    models.append(clf)

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
test_pred[:3]



## === cell 4
sub = sample_sub.copy()
sub = sub.merge(test_df[["image_id"]], on="image_id", how="right")
sub[target_cols] = test_pred
sub = sub[["image_id"] + target_cols]

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

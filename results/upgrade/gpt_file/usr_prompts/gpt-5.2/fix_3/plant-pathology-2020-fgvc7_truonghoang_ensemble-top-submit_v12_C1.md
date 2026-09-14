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
numpy==1.26.4
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



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/working/plant-pathology-2020-fgvc7",
    "/kaggle/working",
]
BASE_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            BASE_DIR = p
            break

if BASE_DIR is None:
    for root in BASE_CANDIDATES:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "train.csv" in filenames and "test.csv" in filenames:
                BASE_DIR = dirpath
                break
        if BASE_DIR is not None:
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv and test.csv."
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
if not os.path.isdir(IMAGES_DIR):
    found = None
    for root in BASE_CANDIDATES:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "images":
                found = dirpath
                break
        if found is not None:
            break
    if found is None:
        raise FileNotFoundError("Could not locate images directory.")
    IMAGES_DIR = found

print("Using BASE_DIR:", BASE_DIR)
print("Using IMAGES_DIR:", IMAGES_DIR)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SAMPLE_SUB)

test_df = test_df.merge(sub[["image_id"]], on="image_id", how="right")

target_cols = [c for c in sub.columns if c != "image_id"]
assert set(target_cols) == {
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
}, f"Unexpected target columns: {target_cols}"

train_df.head(), test_df.head(), sub.head()



## === cell 3
from PIL import Image


def image_features(image_path, gray_size=(32, 32), rgb_small=(16, 16), grid=(4, 4)):
    """
    Minimal, stronger features while preserving overall approach:
    - mean/std for R,G,B
    - mean/std for HSV (captures hue shifts common in disease patterns)
    - mean/std for grayscale
    - downsampled RGB pixels (16x16x3) normalized 0..1 (adds spatial/color info)
    - coarse RGB grid means (4x4x3) normalized 0..1 (very cheap spatial summary)
    """
    with Image.open(image_path) as im:
        im_rgb = im.convert("RGB")
        arr = np.asarray(im_rgb, dtype=np.float32) / 255.0  # HxWx3

    flat = arr.reshape(-1, 3)
    means_rgb = flat.mean(axis=0)
    stds_rgb = flat.std(axis=0)

    with Image.open(image_path) as im_h:
        hsv = im_h.convert("HSV")
        hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0  # HxWx3
    hsv_flat = hsv_arr.reshape(-1, 3)
    means_hsv = hsv_flat.mean(axis=0)
    stds_hsv = hsv_flat.std(axis=0)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    gmean = float(gray.mean())
    gstd = float(gray.std())

    with Image.open(image_path) as im_s:
        im_s = im_s.convert("RGB").resize(rgb_small, resample=Image.BILINEAR)
        rgb_thumb = np.asarray(im_s, dtype=np.float32) / 255.0  # h x w x 3
        rgb_thumb = rgb_thumb.reshape(-1)

    gh, gw = grid
    H, W = arr.shape[0], arr.shape[1]
    ys = np.linspace(0, H, gh + 1, dtype=int)
    xs = np.linspace(0, W, gw + 1, dtype=int)
    grid_feats = []
    for yi in range(gh):
        y0, y1 = ys[yi], ys[yi + 1]
        for xi in range(gw):
            x0, x1 = xs[xi], xs[xi + 1]
            patch = arr[y0:y1, x0:x1, :]
            if patch.size == 0:
                grid_feats.extend([0.0, 0.0, 0.0])
            else:
                pm = patch.reshape(-1, 3).mean(axis=0)
                grid_feats.extend(pm.tolist())
    grid_feats = np.asarray(grid_feats, dtype=np.float32)

    feats = np.concatenate(
        [
            means_rgb,
            stds_rgb,
            means_hsv,
            stds_hsv,
            np.array([gmean, gstd], dtype=np.float32),
            rgb_thumb.astype(np.float32),
            grid_feats,
        ],
        axis=0,
    )
    return feats


def build_feature_matrix(df, images_dir, prefix_options=("Train_", "Test_")):
    first_id = df["image_id"].values[0]
    candidates = [first_id]
    if not any(str(first_id).startswith(p) for p in prefix_options):
        candidates = [p + str(first_id) for p in prefix_options] + [str(first_id)]
    img_path0 = None
    for cand in candidates:
        p = (
            os.path.join(images_dir, f"{cand}.jpg")
            if not str(cand).lower().endswith(".jpg")
            else os.path.join(images_dir, str(cand))
        )
        if os.path.exists(p):
            img_path0 = p
            break
    if img_path0 is None:
        raise FileNotFoundError(
            f"Could not find first image to infer feature size: {first_id}"
        )

    d0 = int(image_features(img_path0).shape[0])
    X = np.zeros((len(df), d0), dtype=np.float32)

    missing = []
    for i, image_id in enumerate(df["image_id"].values):
        candidates = [image_id]
        if not any(str(image_id).startswith(p) for p in prefix_options):
            candidates = [p + str(image_id) for p in prefix_options] + [str(image_id)]
        img_path = None
        for cand in candidates:
            p = (
                os.path.join(images_dir, f"{cand}.jpg")
                if not str(cand).lower().endswith(".jpg")
                else os.path.join(images_dir, str(cand))
            )
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            missing.append(image_id)
            continue
        X[i] = image_features(img_path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images. Example: {missing[:5]}"
        )
    return X




## === cell 4
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

X_train = build_feature_matrix(train_df, IMAGES_DIR)
y_train = train_df[target_cols].values.astype(np.int64)
X_test = build_feature_matrix(test_df, IMAGES_DIR)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000, solver="saga", penalty="l2", n_jobs=-1
                )
            ),
        ),
    ]
)

clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)  # shape (n_test, 4)

proba.shape, proba.min(), proba.max()



## === cell 5
out = pd.DataFrame(proba, columns=target_cols)
out.insert(0, "image_id", test_df["image_id"].values)

out = sub[["image_id"]].merge(out, on="image_id", how="left")
assert out.shape[0] == sub.shape[0], "Submission row count mismatch."
assert list(out.columns) == list(sub.columns), "Submission column mismatch."

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())

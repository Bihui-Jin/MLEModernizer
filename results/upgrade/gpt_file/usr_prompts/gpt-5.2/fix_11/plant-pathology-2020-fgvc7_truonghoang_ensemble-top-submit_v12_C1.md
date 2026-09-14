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



## === cell 1
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



## === cell 2
from PIL import Image


def _build_image_path_map(images_dir):
    m = {}
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(".jpg"):
            m[fn[:-4]] = os.path.join(images_dir, fn)
    return m


_IMAGE_PATH_MAP = _build_image_path_map(IMAGES_DIR)


def _resolve_image_path_fast(image_id):
    s = str(image_id)
    p = _IMAGE_PATH_MAP.get(s)
    if p is not None:
        return p
    p = _IMAGE_PATH_MAP.get("Train_" + s)
    if p is not None:
        return p
    p = _IMAGE_PATH_MAP.get("Test_" + s)
    if p is not None:
        return p
    if s.lower().endswith(".jpg"):
        p = _IMAGE_PATH_MAP.get(s[:-4])
        if p is not None:
            return p
    return None


def _rgb_to_hsv_np(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc, dtype=np.float32)
    nonzero = maxc > 0.0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc, dtype=np.float32)
    nz = delta > 0.0
    mask = nz & (maxc == r)
    h[mask] = (g[mask] - b[mask]) / delta[mask]
    mask = nz & (maxc == g)
    h[mask] = 2.0 + (b[mask] - r[mask]) / delta[mask]
    mask = nz & (maxc == b)
    h[mask] = 4.0 + (r[mask] - g[mask]) / delta[mask]
    h = (h / 6.0) % 1.0  # [0,1)

    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv


def _sobel_edge_stats(gray_f32):
    gy = np.zeros_like(gray_f32, dtype=np.float32)
    gx = np.zeros_like(gray_f32, dtype=np.float32)

    gx[1:-1, 1:-1] = (
        -1.0 * gray_f32[:-2, :-2]
        + 1.0 * gray_f32[:-2, 2:]
        - 2.0 * gray_f32[1:-1, :-2]
        + 2.0 * gray_f32[1:-1, 2:]
        - 1.0 * gray_f32[2:, :-2]
        + 1.0 * gray_f32[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        -1.0 * gray_f32[:-2, :-2]
        - 2.0 * gray_f32[:-2, 1:-1]
        - 1.0 * gray_f32[:-2, 2:]
        + 1.0 * gray_f32[2:, :-2]
        + 2.0 * gray_f32[2:, 1:-1]
        + 1.0 * gray_f32[2:, 2:]
    )

    mag = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
    return np.array(
        [
            float(mag.mean()),
            float(mag.std()),
            float(np.quantile(mag, 0.90)),
            float(np.quantile(mag, 0.99)),
        ],
        dtype=np.float32,
    )


_FEATURE_CACHE = {}


def image_features(image_path, rgb_small=(16, 16), grid=(4, 4)):
    cached = _FEATURE_CACHE.get(image_path)
    if cached is not None:
        return cached

    with Image.open(image_path) as im0:
        im0 = im0.convert("RGB")

        arr_u8 = np.asarray(im0, dtype=np.uint8)  # HxWx3, uint8
        arr = arr_u8.astype(np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        means_rgb = flat.mean(axis=0)
        stds_rgb = flat.std(axis=0)

        hsv_arr = _rgb_to_hsv_np(arr_u8)
        hsv_flat = hsv_arr.reshape(-1, 3)
        means_hsv = hsv_flat.mean(axis=0)
        stds_hsv = hsv_flat.std(axis=0)

        gray = (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(np.float32)
        gmean = float(gray.mean())
        gstd = float(gray.std())

        rgb_q = (
            np.quantile(flat, [0.1, 0.5, 0.9], axis=0).astype(np.float32).reshape(-1)
        )
        hsv_q = (
            np.quantile(hsv_flat, [0.1, 0.5, 0.9], axis=0)
            .astype(np.float32)
            .reshape(-1)
        )
        gray_q = np.array(
            [
                float(np.quantile(gray, 0.1)),
                float(np.quantile(gray, 0.5)),
                float(np.quantile(gray, 0.9)),
            ],
            dtype=np.float32,
        )

        edge_stats = _sobel_edge_stats(gray)

        im_s = im0.resize(rgb_small, resample=Image.BILINEAR)
        rgb_thumb = (
            (np.asarray(im_s, dtype=np.float32) / 255.0).reshape(-1).astype(np.float32)
        )

    gh, gw = grid
    H, W = arr.shape[0], arr.shape[1]

    Hc = (H // gh) * gh
    Wc = (W // gw) * gw
    if Hc == 0 or Wc == 0:
        grid_feats = np.zeros((gh * gw * 3,), dtype=np.float32)
    else:
        arr_c = arr[:Hc, :Wc, :]
        bh = Hc // gh
        bw = Wc // gw
        means_grid = arr_c.reshape(gh, bh, gw, bw, 3).mean(axis=(1, 3))
        grid_feats = means_grid.reshape(-1).astype(np.float32)

    feats = np.concatenate(
        [
            means_rgb.astype(np.float32),
            stds_rgb.astype(np.float32),
            means_hsv.astype(np.float32),
            stds_hsv.astype(np.float32),
            np.array([gmean, gstd], dtype=np.float32),
            rgb_q,
            hsv_q,
            gray_q,
            edge_stats,
            rgb_thumb,
            grid_feats,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    _FEATURE_CACHE[image_path] = feats
    return feats


def build_feature_matrix(df, images_dir):
    ids = df["image_id"].astype(str).values
    paths = [_resolve_image_path_fast(i) for i in ids]

    missing = [ids[i] for i, p in enumerate(paths) if p is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images. Example: {missing[:5]}"
        )

    d0 = image_features(paths[0]).shape[0]
    X = np.empty((len(paths), d0), dtype=np.float32)

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(image_features, paths, chunksize=16)):
            X[i] = feats

    return X




## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

X_train = build_feature_matrix(train_df, IMAGES_DIR)
y_train = train_df[target_cols].values.astype(np.int64)
X_test = build_feature_matrix(test_df, IMAGES_DIR)

y_single = y_train.argmax(axis=1).astype(np.int64)

C_grid = [0.3, 1.0, 3.0]
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


def _eval_multinomial_C(C):
    oof = np.zeros((X_train.shape[0], 4), dtype=np.float64)
    for tr_idx, va_idx in skf.split(X_train, y_single):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        penalty="l2",
                        multi_class="multinomial",
                        n_jobs=-1,
                        C=C,
                    ),
                ),
            ]
        )
        clf.fit(X_train[tr_idx], y_single[tr_idx])
        proba_va = clf.predict_proba(X_train[va_idx])

        classes = clf.named_steps["lr"].classes_
        if not np.array_equal(classes, np.array([0, 1, 2, 3], dtype=np.int64)):
            proba_va = proba_va[:, np.argsort(classes)]

        oof[va_idx] = proba_va

    aucs = []
    for j in range(4):
        aucs.append(roc_auc_score(y_train[:, j], oof[:, j]))
    return float(np.mean(aucs))


best_C = None
best_cv = -1.0
for C in C_grid:
    cv_score = _eval_multinomial_C(C)
    print(f"CV mean AUC for multinomial LR (C={C}): {cv_score:.6f}")
    if cv_score > best_cv:
        best_cv = cv_score
        best_C = C

print("Selected best C for multinomial LR:", best_C, "with CV mean AUC:", best_cv)

base_mult = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                penalty="l2",
                multi_class="multinomial",
                n_jobs=-1,
                C=best_C,
            ),
        ),
    ]
)
clf_mult = CalibratedClassifierCV(estimator=base_mult, method="sigmoid", cv=3)

clf_balanced = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000,
                    solver="saga",
                    penalty="l2",
                    n_jobs=-1,
                    class_weight="balanced",
                )
            ),
        ),
    ]
)

clf_alt = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    penalty="l2",
                    n_jobs=-1,
                    C=0.7,
                    class_weight="balanced",
                )
            ),
        ),
    ]
)

clf_mult.fit(X_train, y_single)
proba_mult = clf_mult.predict_proba(X_test)

classes = clf_mult.classes_
if not np.array_equal(classes, np.array([0, 1, 2, 3], dtype=np.int64)):
    proba_mult = proba_mult[:, np.argsort(classes)]

clf_balanced.fit(X_train, y_train)
proba1 = clf_balanced.predict_proba(X_test)

clf_alt.fit(X_train, y_train)
proba2 = clf_alt.predict_proba(X_test)

proba_ovr_blend = 0.6 * proba1 + 0.4 * proba2
proba = 0.80 * proba_mult + 0.20 * proba_ovr_blend
proba = np.clip(proba, 1e-6, 1 - 1e-6)

print("proba:", proba.shape, float(proba.min()), float(proba.max()))



## === cell 4
out = pd.DataFrame(proba, columns=target_cols)
out.insert(0, "image_id", test_df["image_id"].values)

out = sub[["image_id"]].merge(out, on="image_id", how="left")
assert out.shape[0] == sub.shape[0], "Submission row count mismatch."
assert list(out.columns) == list(sub.columns), "Submission column mismatch."

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())

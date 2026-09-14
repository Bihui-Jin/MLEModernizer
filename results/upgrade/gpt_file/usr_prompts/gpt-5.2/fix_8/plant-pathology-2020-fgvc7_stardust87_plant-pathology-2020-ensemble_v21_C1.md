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
import glob
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 2
def _find_candidate_submission_csvs():
    """
    Bug fix: /kaggle/input/submissions/ is often absent in Kaggle notebooks.
    We search broadly for CSVs that look like submissions (contain target columns).
    """
    candidates = []

    if os.path.isdir(SUBMISSIONS_PATH):
        for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    candidates.append(os.path.join(dirname, filename))

    for root in DATA_ROOTS:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
            )

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    filtered = []
    for p in unique:
        base = os.path.basename(p).lower()
        if base in ("train.csv", "test.csv", "sample_submission.csv"):
            continue
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df_head.columns)
        if "image_id" in cols and all(c in cols for c in TARGET_COLS):
            filtered.append(p)

    filtered.sort()
    return filtered


submissions_all = _find_candidate_submission_csvs()
print("Discovered candidate submission CSVs:")
print(submissions_all if submissions_all else "(none found)")




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble 0 submissions.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must equal sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    max_idx = max(sub_idx)
    if max_idx >= len(submissions_all):
        raise IndexError(
            f"Requested submission index {max_idx}, but only {len(submissions_all)} files found."
        )

    weights = np.array(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    weights = weights / weights.sum()

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]:.6f}")
        submission = pd.read_csv(path)

        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")

        arr = submission.loc[:, TARGET_COLS].to_numpy(dtype=float)
        submission_with_weight.append(arr * weights[i])

    submission_avg = np.sum(submission_with_weight, axis=0)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all, out_path="submission.csv"):
    sample_paths = [
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    sample_path = None
    for p in sample_paths:
        if os.path.exists(p):
            sample_path = p
            break

    if sample_path is None:
        if len(submissions_all) == 0:
            raise FileNotFoundError(
                "Could not locate sample_submission.csv and no submissions_all available."
            )
        sample_path = submissions_all[0]

    submission_df = pd.read_csv(sample_path)

    if submission_avg.shape[0] != len(submission_df):
        raise ValueError(
            f"Row count mismatch: submission_avg has {submission_avg.shape[0]} rows "
            f"but template has {len(submission_df)} rows."
        )
    if submission_avg.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Column count mismatch: submission_avg has {submission_avg.shape[1]} cols but expected {len(TARGET_COLS)}."
        )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0
    submission_df[TARGET_COLS] = submission_avg

    submission_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path}")
    print(submission_df.head())




## === cell 5
def _resolve_competition_root():
    roots = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for r in roots:
        if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
            os.path.join(r, "test.csv")
        ):
            return r
    raise FileNotFoundError(
        "Could not find a root containing train.csv and test.csv in known locations."
    )


def _image_path(root, image_id):
    candidates = [
        os.path.join(root, "images", f"{image_id}.jpg"),
        os.path.join(root, "plant-pathology-2020-fgvc7", "images", f"{image_id}.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _resize_rgb(img_rgb, size=(96, 96)):
    try:
        import cv2  # type: ignore

        return cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(img_rgb).resize(size, resample=Image.BILINEAR))


def _rgb_to_hsv01(x_rgb01):
    """
    Minimal, dependency-free RGB->HSV in [0,1].
    """
    r = x_rgb01[..., 0]
    g = x_rgb01[..., 1]
    b = x_rgb01[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    s = np.zeros_like(cmax, dtype=np.float32)
    v = cmax.astype(np.float32)

    mask = delta > 1e-8
    s[mask] = (delta[mask] / (cmax[mask] + 1e-12)).astype(np.float32)

    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g)
    mask_b = mask & (cmax == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-12)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-12)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-12)) + 4.0
    h = (h / 6.0).astype(np.float32)
    return np.stack([h, s, v], axis=-1)


def _block_pool_2d(x2d, out_h, out_w):
    """
    Unambiguous block-mean pooling to get true (out_h, out_w) grids from 96x96 grayscale.
    """
    h, w = x2d.shape
    if h % out_h != 0 or w % out_w != 0:
        raise ValueError(f"Cannot block-pool {h}x{w} to {out_h}x{out_w}.")
    bh = h // out_h
    bw = w // out_w
    return x2d.reshape(out_h, bh, out_w, bw).mean(axis=(1, 3))


def _extract_simple_features_single(img):
    """
    Features: simple color stats, histograms, edge stats, and low-res pooled grayscale grids.
    """
    img = _resize_rgb(img, size=(96, 96))

    x = img.astype(np.float32) / 255.0
    feats = []

    feats.extend(x.mean(axis=(0, 1)).tolist())
    feats.extend(x.std(axis=(0, 1)).tolist())

    gray = (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).astype(
        np.float32
    )
    gmean = float(gray.mean())
    gstd = float(gray.std())
    feats.append(gmean)
    feats.append(gstd)

    z = (gray - gmean) / (gstd + 1e-6)
    feats.append(float(np.mean(z**3)))  # skewness proxy
    feats.append(float(np.mean(z**4)))  # kurtosis proxy

    bins = 16
    for c in range(3):
        hist, _ = np.histogram(x[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        hist = np.log1p(hist)
        feats.extend(hist.tolist())

    hsv = _rgb_to_hsv01(x)
    feats.extend(hsv.mean(axis=(0, 1)).tolist())
    feats.extend(hsv.std(axis=(0, 1)).tolist())
    for c in range(3):
        hist, _ = np.histogram(hsv[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        hist = np.log1p(hist)
        feats.extend(hist.tolist())

    feats.append(float((x[..., 1] - x[..., 0]).mean()))
    feats.append(float((x[..., 0] - x[..., 1]).mean()))
    feats.append(float((x[..., 1] - x[..., 2]).mean()))
    feats.append(float((x[..., 2] - x[..., 1]).mean()))

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge = (gx.mean() + gy.mean()) / 2.0
    feats.append(float(edge))
    feats.append(float(gx.std()))
    feats.append(float(gy.std()))

    pooled_12 = _block_pool_2d(gray, 12, 12)  # 144 dims
    feats.extend(np.log1p(pooled_12.astype(np.float32)).ravel().tolist())

    pooled_8 = _block_pool_2d(gray, 8, 8)  # 64 dims
    feats.extend(np.log1p(pooled_8.astype(np.float32)).ravel().tolist())

    return np.array(feats, dtype=np.float32)


def _extract_simple_features(img):
    """
    Deterministic test-time augmentation at feature level: average features over a few transforms.
    """
    img0 = img
    img1 = img[:, ::-1, :]  # horizontal flip
    img2 = img[::-1, :, :]  # vertical flip
    img3 = np.rot90(img, k=1)  # 90-degree rotation

    f0 = _extract_simple_features_single(img0)
    f1 = _extract_simple_features_single(img1)
    f2 = _extract_simple_features_single(img2)
    f3 = _extract_simple_features_single(img3)

    return ((f0 + f1 + f2 + f3) / 4.0).astype(np.float32)


def _build_feature_matrix(image_ids, root):
    try:
        import cv2  # type: ignore

        use_cv2 = True
    except Exception:
        use_cv2 = False
        from PIL import Image  # pillow is typically available on Kaggle

    X = []
    missing = 0
    for image_id in image_ids:
        p = _image_path(root, image_id)
        if p is None:
            missing += 1
            X.append(None)
            continue
        if use_cv2:
            img = cv2.imread(p)  # BGR uint8
            if img is None:
                missing += 1
                X.append(None)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(p).convert("RGB"))
        X.append(_extract_simple_features(img))

    available = [x for x in X if x is not None]
    if len(available) == 0:
        raise RuntimeError("Could not read any images to build features.")
    mean_feat = np.mean(np.stack(available, axis=0), axis=0)
    X2 = np.stack([x if x is not None else mean_feat for x in X], axis=0)
    if missing:
        print(
            f"Warning: {missing} images missing/unreadable; imputed features with mean."
        )
    return X2


def train_and_predict_baseline(out_path="submission.csv"):
    root = _resolve_competition_root()
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    sample_path = os.path.join(root, "sample_submission.csv")
    if not os.path.exists(sample_path):
        sample_paths = [
            "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
            "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
            "/kaggle/data/sample_submission.csv",
        ]
        sample_path = next((p for p in sample_paths if os.path.exists(p)), None)
    if sample_path is None:
        raise FileNotFoundError(
            "sample_submission.csv not found for templating output."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train_df.columns:
            raise KeyError(f"train.csv missing required column: {c}")
    if "image_id" not in test_df.columns:
        raise KeyError("test.csv missing required column: image_id")

    X_train = _build_feature_matrix(train_df["image_id"].tolist(), root)
    X_test = _build_feature_matrix(test_df["image_id"].tolist(), root)

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    y_train = train_df[TARGET_COLS].astype(np.int32).to_numpy(copy=False)

    base_lr = LogisticRegression(
        max_iter=6000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        random_state=0,
    )

    clf = OneVsRestClassifier(
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", base_lr),
            ]
        ),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    sub = pd.read_csv(sample_path)
    sub = sub.set_index("image_id").reindex(test_df["image_id"].values).reset_index()

    for i, c in enumerate(TARGET_COLS):
        sub[c] = np.clip(proba[:, i].astype(np.float64), 0.0, 1.0)

    sub.to_csv(out_path, index=False)
    print(f"Wrote baseline image-model submission to: {out_path}")
    print(sub.head())




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.04, 0.92, 0.04])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
elif len(submissions_all) >= 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
else:
    train_and_predict_baseline(out_path="submission.csv")

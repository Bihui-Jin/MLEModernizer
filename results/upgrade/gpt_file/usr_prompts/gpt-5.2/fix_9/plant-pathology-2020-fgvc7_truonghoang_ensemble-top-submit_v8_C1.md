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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]


def find_file(filename, base_dirs):
    for bd in base_dirs:
        p = os.path.join(bd, filename)
        if os.path.exists(p):
            return p
    return None


def find_images_dir(base_dirs):
    for bd in base_dirs:
        p = os.path.join(bd, "images")
        if os.path.isdir(p):
            return p
    for bd in base_dirs:
        p = os.path.join(bd, "plant-pathology-2020-fgvc7", "images")
        if os.path.isdir(p):
            return p
    return None


train_path = find_file("train.csv", BASE_DIR_CANDIDATES)
test_path = find_file("test.csv", BASE_DIR_CANDIDATES)
sample_sub_path = find_file("sample_submission.csv", BASE_DIR_CANDIDATES)
images_dir = find_images_dir(BASE_DIR_CANDIDATES)

if (
    train_path is None
    or test_path is None
    or sample_sub_path is None
    or images_dir is None
):
    raise FileNotFoundError(
        f"Could not locate required files. Found:"
        f"\ntrain_path={train_path}"
        f"\ntest_path={test_path}"
        f"\nsample_sub_path={sample_sub_path}"
        f"\nimages_dir={images_dir}"
    )

print("Using:")
print("train:", train_path)
print("test:", test_path)
print("sample_submission:", sample_sub_path)
print("images_dir:", images_dir)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
print("Targets:", target_cols)
print(
    "Train shape:",
    train_df.shape,
    "Test shape:",
    test_df.shape,
    "Sample sub shape:",
    sample_sub.shape,
)



## === cell 2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

IMG_SIZE = (96, 96)
FEAT_SIZE = (
    48,
    48,
)  # compact per-pixel feature maps (still deterministic, no learning)


def _rgb_to_hsv01(arr01):
    """
    Deterministic RGB->HSV conversion for arr01 in [0,1], returns hsv in [0,1].
    Implemented in numpy to avoid extra dependencies.
    """
    r = arr01[..., 0]
    g = arr01[..., 1]
    b = arr01[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-12

    rc = np.zeros_like(cmax, dtype=np.float32)
    gc = np.zeros_like(cmax, dtype=np.float32)
    bc = np.zeros_like(cmax, dtype=np.float32)
    rc[mask] = ((g - b) / delta)[mask]
    gc[mask] = ((b - r) / delta)[mask]
    bc[mask] = ((r - g) / delta)[mask]

    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask

    h[r_is_max] = (rc[r_is_max] % 6.0) / 6.0
    h[g_is_max] = (gc[g_is_max] + 2.0) / 6.0
    h[b_is_max] = (bc[b_is_max] + 4.0) / 6.0

    s = np.zeros_like(cmax, dtype=np.float32)
    nonzero = cmax > 1e-12
    s[nonzero] = (delta / cmax)[nonzero]

    v = cmax.astype(np.float32)

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _color_texture_stats(arr01):
    """
    Minimal, fast summary features:
    - per-channel mean/std (6)
    - grayscale mean/std (2)
    - HSV mean/std (6)
    - simple gradient magnitude mean/std on grayscale (2)
    Total: 16 features
    """
    ch_mean = arr01.reshape(-1, 3).mean(axis=0)
    ch_std = arr01.reshape(-1, 3).std(axis=0)

    gray = (
        0.2989 * arr01[..., 0] + 0.5870 * arr01[..., 1] + 0.1140 * arr01[..., 2]
    ).astype(np.float32)
    g_mean = np.array([gray.mean()], dtype=np.float32)
    g_std = np.array([gray.std()], dtype=np.float32)

    hsv = _rgb_to_hsv01(arr01)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    if gx.shape[0] > 1 and gy.shape[1] > 1:
        grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    else:
        grad_mag = np.zeros((0,), dtype=np.float32)

    if grad_mag.size == 0:
        gm_mean = np.array([0.0], dtype=np.float32)
        gm_std = np.array([0.0], dtype=np.float32)
    else:
        gm_mean = np.array([grad_mag.mean()], dtype=np.float32)
        gm_std = np.array([grad_mag.std()], dtype=np.float32)

    return np.concatenate(
        [ch_mean, ch_std, g_mean, g_std, hsv_mean, hsv_std, gm_mean, gm_std], axis=0
    ).astype(np.float32)


def _hog_like_hist_from_gray(gray01, n_bins=9):
    """
    Adds a tiny HOG-like orientation histogram from grayscale gradients.
    Kept to preserve your current feature set + model pipeline.
    """
    gx = np.diff(gray01, axis=1)
    gy = np.diff(gray01, axis=0)
    if gx.shape[0] <= 1 or gy.shape[1] <= 1:
        return np.zeros((2 * n_bins,), dtype=np.float32)

    gx = gx[:-1, :]
    gy = gy[:, :-1]
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    ang = (np.arctan2(gy, gx) + np.pi).astype(np.float32)  # [0, 2pi)
    bins = (ang / (2.0 * np.pi) * n_bins).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)

    hist = np.zeros((n_bins,), dtype=np.float32)
    flat_bins = bins.reshape(-1)
    flat_mag = mag.reshape(-1)
    np.add.at(hist, flat_bins, flat_mag)

    s = float(hist.sum())
    if s > 0:
        hist_sum = hist / s
    else:
        hist_sum = hist

    hist_mean = hist_sum / max(1.0, float(flat_bins.size))
    return np.concatenate([hist_sum, hist_mean], axis=0).astype(np.float32)


def _color_histograms(arr01, n_bins=16):
    """
    Compact global color histograms (RGB + HSV), L1-normalized.
    Total: 6*n_bins = 96 features when n_bins=16.
    """
    eps = 1e-12
    arr01 = np.clip(arr01.astype(np.float32), 0.0, 1.0)

    rgb = arr01.reshape(-1, 3)
    hists = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, c], bins=n_bins, range=(0.0, 1.0))
        h = h.astype(np.float32)
        h /= h.sum() + eps
        hists.append(h)

    hsv = _rgb_to_hsv01(arr01).reshape(-1, 3)
    for c in range(3):
        h, _ = np.histogram(hsv[:, c], bins=n_bins, range=(0.0, 1.0))
        h = h.astype(np.float32)
        h /= h.sum() + eps
        hists.append(h)

    return np.concatenate(hists, axis=0).astype(np.float32)


def _fit_color_codebook(
    image_ids,
    images_dir,
    img_size=(96, 96),
    sample_size_per_image=256,
    n_clusters=32,
    batch_size=4096,
    random_state=RANDOM_STATE,
):
    rng = np.random.RandomState(random_state)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=3,
        reassignment_ratio=0.01,
    )

    fitted_any = False
    for img_id in image_ids:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            alt = os.path.join(images_dir, f"{img_id}.JPG")
            if os.path.exists(alt):
                img_path = alt
            else:
                continue

        img = (
            Image.open(img_path)
            .convert("RGB")
            .resize(img_size, resample=Image.BILINEAR)
        )
        arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue

        idx = rng.choice(
            arr.shape[0], size=min(sample_size_per_image, arr.shape[0]), replace=False
        )
        km.partial_fit(arr[idx])
        fitted_any = True

    if not fitted_any:
        raise RuntimeError("Failed to fit color codebook: no images found/readable.")
    return km


def _color_code_hist(arr01, km, eps=1e-12):
    flat = arr01.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] == 0:
        return np.zeros((km.n_clusters,), dtype=np.float32)
    labels = km.predict(flat)
    hist = np.bincount(labels, minlength=km.n_clusters).astype(np.float32)
    hist /= hist.sum() + eps
    return hist


def load_image_features(
    image_ids,
    images_dir,
    img_size=(96, 96),
    feat_size=(48, 48),
    color_km=None,
):
    """
    Features = [
      downsampled grayscale (feat_size),
      downsampled edge magnitude (feat_size),
      downsampled RGB (3 * feat_size),
      + summary stats (16),
      + HOG-like gradient orientation hist (18),
      + RGB+HSV color histograms (96),
      + optional color-codebook histogram (n_clusters)
    ]
    """
    n_map = feat_size[0] * feat_size[1]
    n_stats = 16
    n_hog = 18  # 2*9 bins
    n_hist = 96  # 6*16 bins
    n_code = 0 if color_km is None else int(color_km.n_clusters)

    X = np.zeros(
        (len(image_ids), 5 * n_map + n_stats + n_hog + n_hist + n_code),
        dtype=np.float32,
    )
    missing = 0

    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            alt = os.path.join(images_dir, f"{img_id}.JPG")
            if os.path.exists(alt):
                img_path = alt
            else:
                missing += 1
                continue

        img = (
            Image.open(img_path)
            .convert("RGB")
            .resize(img_size, resample=Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0

        gray = (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(np.float32)

        hog_feat = _hog_like_hist_from_gray(gray, n_bins=9)

        gray_img = Image.fromarray(
            np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L"
        ).resize(feat_size, resample=Image.BILINEAR)
        gray_small = (np.asarray(gray_img, dtype=np.float32) / 255.0).reshape(-1)

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        if gx.shape[0] > 1 and gy.shape[1] > 1:
            grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2).astype(np.float32)
        else:
            grad_mag = np.zeros((img_size[1] - 1, img_size[0] - 1), dtype=np.float32)

        gmin, gmax = float(grad_mag.min()), float(grad_mag.max())
        if gmax > gmin:
            grad_mag01 = (grad_mag - gmin) / (gmax - gmin)
        else:
            grad_mag01 = np.zeros_like(grad_mag, dtype=np.float32)

        edge_img = Image.fromarray(
            np.clip(grad_mag01 * 255.0, 0, 255).astype(np.uint8), mode="L"
        ).resize(feat_size, resample=Image.BILINEAR)
        edge_small = (np.asarray(edge_img, dtype=np.float32) / 255.0).reshape(-1)

        rgb_small_img = img.resize(feat_size, resample=Image.BILINEAR)
        rgb_small = (np.asarray(rgb_small_img, dtype=np.float32) / 255.0).reshape(-1)

        stats = _color_texture_stats(arr)
        hists = _color_histograms(arr, n_bins=16)

        X[i, 0:n_map] = gray_small
        X[i, n_map : 2 * n_map] = edge_small
        X[i, 2 * n_map : 5 * n_map] = rgb_small
        off = 5 * n_map
        X[i, off : off + n_stats] = stats
        off += n_stats
        X[i, off : off + n_hog] = hog_feat
        off += n_hog
        X[i, off : off + n_hist] = hists
        off += n_hist

        if color_km is not None:
            X[i, off : off + n_code] = _color_code_hist(arr, color_km)

    if missing:
        print(f"Warning: {missing} images were missing and left as zeros.")
    return X


color_km = _fit_color_codebook(
    train_df["image_id"].values,
    images_dir,
    img_size=IMG_SIZE,
    sample_size_per_image=256,
    n_clusters=32,
    batch_size=4096,
    random_state=RANDOM_STATE,
)
print("Fitted color codebook with n_clusters =", color_km.n_clusters)

X_train = load_image_features(
    train_df["image_id"].values, images_dir, IMG_SIZE, FEAT_SIZE, color_km=color_km
)
y_train = train_df[target_cols].values.astype(np.int32)
X_test = load_image_features(
    test_df["image_id"].values, images_dir, IMG_SIZE, FEAT_SIZE, color_km=color_km
)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)



## === cell 3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

row_sum = y_train.sum(axis=1)
strat_labels = np.where(row_sum > 0, y_train.argmax(axis=1), 0)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=strat_labels,
)

candidates = [
    {"C": 0.6, "l1_ratio": 0.0, "max_iter": 2500},  # pure L2 (baseline-compatible)
    {"C": 0.8, "l1_ratio": 0.15, "max_iter": 3000},  # mild sparsity
    {"C": 1.0, "l1_ratio": 0.30, "max_iter": 3500},  # a bit more sparsity
]

best_cfg = None
best_auc = -1.0

for cfg in candidates:
    base_lr = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=cfg["l1_ratio"],
        C=cfg["C"],
        max_iter=cfg["max_iter"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    estimator = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        base_lr,
    )

    clf = MultiOutputClassifier(estimator, n_jobs=1)
    clf.fit(X_tr, y_tr)

    va_pred = np.zeros((X_va.shape[0], len(target_cols)), dtype=np.float32)
    for k in range(len(target_cols)):
        va_pred[:, k] = clf.estimators_[k].predict_proba(X_va)[:, 1]
    va_pred = np.clip(va_pred, 1e-6, 1 - 1e-6)

    aucs = []
    for k in range(len(target_cols)):
        aucs.append(roc_auc_score(y_va[:, k], va_pred[:, k]))
    mean_auc = float(np.mean(aucs))

    print(
        f"Candidate {cfg} -> val mean AUC: {mean_auc:.6f} (per-class: {[round(a,6) for a in aucs]})"
    )

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_cfg = cfg

print("Selected config:", best_cfg, "with val mean AUC:", best_auc)

final_base_lr = LogisticRegression(
    solver="saga",
    penalty="elasticnet",
    l1_ratio=best_cfg["l1_ratio"],
    C=best_cfg["C"],
    max_iter=best_cfg["max_iter"],
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)
final_estimator = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    final_base_lr,
)
final_clf = MultiOutputClassifier(final_estimator, n_jobs=1)
final_clf.fit(X_train, y_train)

pred = np.zeros((X_test.shape[0], len(target_cols)), dtype=np.float32)
for k in range(len(target_cols)):
    pred[:, k] = final_clf.estimators_[k].predict_proba(X_test)[:, 1]
pred = np.clip(pred, 1e-6, 1 - 1e-6)

sub = sample_sub.copy()
sub["image_id"] = test_df["image_id"].values
sub[target_cols] = pred

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

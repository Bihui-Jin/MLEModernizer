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



## === cell 1
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "plant-pathology-2020-fgvc7", "images"),
]

print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
for p in IMAGES_DIR_CANDIDATES:
    print("Exists images dir candidate:", p, os.path.exists(p))




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have the same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty. Provide valid submission file paths to ensemble."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} out of range for submissions_all of length {len(submissions_all)}"
            )
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file_from_df(pred_df, out_path="submission.csv"):
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if list(sample.columns) != required_cols:
        sample = sample[required_cols]

    if "image_id" not in pred_df.columns:
        raise ValueError("pred_df must contain an 'image_id' column")

    pred_df = pred_df.copy()
    pred_df = pred_df[required_cols]

    merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left", sort=False)

    for c in required_cols[1:]:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.25)

    for c in required_cols[1:]:
        merged[c] = merged[c].clip(0.0, 1.0)

    merged.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {merged.shape} and columns {list(merged.columns)}"
    )




## === cell 4
import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL is required to read images. Please ensure Pillow is available in the environment."
    ) from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

np.random.seed(42)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def image_path(image_id: str) -> str:
    fname = f"{image_id}.jpg"
    for d in IMAGES_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return os.path.join(IMAGES_DIR_CANDIDATES[0], fname)


def _rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delt = maxc - minc
    s = np.where(maxc > 1e-8, delt / (maxc + 1e-8), 0.0)

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delt > 1e-8

    rc = np.where(mask, (maxc - r) / (delt + 1e-8), 0.0)
    gc = np.where(mask, (maxc - g) / (delt + 1e-8), 0.0)
    bc = np.where(mask, (maxc - b) / (delt + 1e-8), 0.0)

    h = np.where((maxc == r) & mask, (bc - gc), h)
    h = np.where((maxc == g) & mask, 2.0 + (rc - bc), h)
    h = np.where((maxc == b) & mask, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def _rgb_to_lab_np(rgb01: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_to_linear(rgb01)
    r = rgb_lin[..., 0]
    g = rgb_lin[..., 1]
    b = rgb_lin[..., 2]

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    eps = 216.0 / 24389.0
    k = 24389.0 / 27.0

    def f(t):
        return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0).astype(np.float32)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    lab = np.stack(
        [L / 100.0, (a + 128.0) / 255.0, (bb + 128.0) / 255.0], axis=-1
    ).astype(np.float32)
    return lab


def _foreground_crop(arr01: np.ndarray) -> np.ndarray:
    hsv = _rgb_to_hsv_np(arr01)
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    mask = (S > 0.18) & (V > 0.2)
    mask = mask | (((H > 0.18) & (H < 0.48)) & (V > 0.15))

    ys, xs = np.where(mask)
    if ys.size < 200:
        return arr01

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = arr01.shape[0], arr01.shape[1]
    dy = int(0.05 * (y1 - y0 + 1))
    dx = int(0.05 * (x1 - x0 + 1))
    y0 = max(0, y0 - dy)
    y1 = min(h - 1, y1 + dy)
    x0 = max(0, x0 - dx)
    x1 = min(w - 1, x1 + dx)

    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return arr01
    return arr01[y0 : y1 + 1, x0 : x1 + 1, :]


def extract_features(
    img_path: str, size_rgb=(96, 96), size_rgb2=(160, 160), thumb_gray=(32, 32)
) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        arr_full = np.asarray(im, dtype=np.float32) / 255.0

    arr_full = _foreground_crop(arr_full)

    im1 = Image.fromarray(
        np.clip(arr_full * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    ).resize(size_rgb, resample=_RESAMPLE)
    arr1 = (np.asarray(im1, dtype=np.float32) / 255.0).astype(np.float32)

    im2 = Image.fromarray(
        np.clip(arr_full * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    ).resize(size_rgb2, resample=_RESAMPLE)
    arr2 = (np.asarray(im2, dtype=np.float32) / 255.0).astype(np.float32)

    def stats_rgb_hsv_lab(arr):
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))

        hsv = _rgb_to_hsv_np(arr)
        hsv_means = hsv.mean(axis=(0, 1))
        hsv_stds = hsv.std(axis=(0, 1))

        lab = _rgb_to_lab_np(arr)
        lab_means = lab.mean(axis=(0, 1))
        lab_stds = lab.std(axis=(0, 1))

        a = lab[..., 1]
        b = lab[..., 2]
        eps = 1e-6
        ab_ratio_mean = np.array([float(np.mean(a / (b + eps)))], dtype=np.float32)
        ba_ratio_mean = np.array([float(np.mean(b / (a + eps)))], dtype=np.float32)

        r = arr[..., 0]
        g = arr[..., 1]
        bch = arr[..., 2]
        rg = (r / (g + eps)).mean()
        rb = (r / (bch + eps)).mean()
        gb = (g / (bch + eps)).mean()
        r_minus_g = (r - g).mean()
        r_minus_b = (r - bch).mean()
        g_minus_b = (g - bch).mean()

        overall_mean = np.array([arr.mean()], dtype=np.float32)
        overall_std = np.array([arr.std()], dtype=np.float32)

        extra_scalars = np.array(
            [rg, rb, gb, r_minus_g, r_minus_b, g_minus_b],
            dtype=np.float32,
        )

        return (
            means.astype(np.float32),
            stds.astype(np.float32),
            hsv_means.astype(np.float32),
            hsv_stds.astype(np.float32),
            lab_means.astype(np.float32),
            lab_stds.astype(np.float32),
            overall_mean,
            overall_std,
            extra_scalars,
            ab_ratio_mean,
            ba_ratio_mean,
        )

    (
        means1,
        stds1,
        hsv_means1,
        hsv_stds1,
        lab_means1,
        lab_stds1,
        overall_mean1,
        overall_std1,
        extra_scalars1,
        ab_ratio_mean1,
        ba_ratio_mean1,
    ) = stats_rgb_hsv_lab(arr1)

    (
        means2,
        stds2,
        hsv_means2,
        hsv_stds2,
        lab_means2,
        lab_stds2,
        overall_mean2,
        overall_std2,
        extra_scalars2,
        ab_ratio_mean2,
        ba_ratio_mean2,
    ) = stats_rgb_hsv_lab(arr2)

    r2 = arr2[..., 0]
    g2 = arr2[..., 1]
    b2 = arr2[..., 2]
    gray = (0.2989 * r2 + 0.5870 * g2 + 0.1140 * b2).astype(np.float32)

    gray_q25 = np.array([np.quantile(gray, 0.25)], dtype=np.float32)
    gray_q50 = np.array([np.quantile(gray, 0.50)], dtype=np.float32)
    gray_q75 = np.array([np.quantile(gray, 0.75)], dtype=np.float32)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    grad_abs_mean = np.array(
        [np.mean(np.abs(dx)) + np.mean(np.abs(dy))], dtype=np.float32
    )
    grad_sq_mean = np.array([np.mean(dx * dx) + np.mean(dy * dy)], dtype=np.float32)

    grad_mag = np.sqrt(
        np.pad(dx, ((0, 0), (0, 1)), mode="edge") ** 2
        + np.pad(dy, ((0, 1), (0, 0)), mode="edge") ** 2
    ).astype(np.float32)
    grad_q90 = np.array([np.quantile(grad_mag, 0.90)], dtype=np.float32)
    grad_q99 = np.array([np.quantile(grad_mag, 0.99)], dtype=np.float32)

    q10 = np.quantile(gray, 0.10)
    q90 = np.quantile(gray, 0.90)
    frac_dark = np.array([float(np.mean(gray <= q10))], dtype=np.float32)
    frac_bright = np.array([float(np.mean(gray >= q90))], dtype=np.float32)

    gpad = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gmean3 = (
        gpad[0:-2, 0:-2]
        + gpad[0:-2, 1:-1]
        + gpad[0:-2, 2:]
        + gpad[1:-1, 0:-2]
        + gpad[1:-1, 1:-1]
        + gpad[1:-1, 2:]
        + gpad[2:, 0:-2]
        + gpad[2:, 1:-1]
        + gpad[2:, 2:]
    ) / 9.0
    local_resid = (gray - gmean3).astype(np.float32)
    local_contrast_mean = np.array(
        [float(np.mean(np.abs(local_resid)))], dtype=np.float32
    )
    local_contrast_std = np.array([float(np.std(local_resid))], dtype=np.float32)

    gray_img = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    gray_img = gray_img.resize(thumb_gray, resample=_RESAMPLE)
    gray_thumb = (np.asarray(gray_img, dtype=np.float32) / 255.0).reshape(-1)
    gray_thumb_mean = np.array([gray_thumb.mean()], dtype=np.float32)
    gray_thumb_std = np.array([gray_thumb.std()], dtype=np.float32)

    feat = np.concatenate(
        [
            means1,
            stds1,
            hsv_means1,
            hsv_stds1,
            lab_means1,
            lab_stds1,
            overall_mean1,
            overall_std1,
            extra_scalars1,
            ab_ratio_mean1,
            ba_ratio_mean1,
            means2,
            stds2,
            hsv_means2,
            hsv_stds2,
            lab_means2,
            lab_stds2,
            overall_mean2,
            overall_std2,
            extra_scalars2,
            ab_ratio_mean2,
            ba_ratio_mean2,
            gray_q25,
            gray_q50,
            gray_q75,
            grad_abs_mean,
            grad_sq_mean,
            grad_q90,
            grad_q99,
            frac_dark,
            frac_bright,
            local_contrast_mean,
            local_contrast_std,
            gray_thumb_mean,
            gray_thumb_std,
            gray_thumb,
        ],
        axis=0,
    ).astype(np.float32)

    return feat


_FEATURE_CACHE = {}


def build_features(image_ids: pd.Series) -> np.ndarray:
    first_feat = None
    for img_id in image_ids.tolist():
        p = image_path(img_id)
        if os.path.exists(p):
            first_feat = extract_features(p)
            break
    if first_feat is None:
        raise RuntimeError(
            "No images found to infer feature dimension. Check IMAGES_DIR paths."
        )

    X = np.zeros((len(image_ids), first_feat.shape[0]), dtype=np.float32)
    for i, img_id in enumerate(image_ids.tolist()):
        if img_id in _FEATURE_CACHE:
            X[i] = _FEATURE_CACHE[img_id]
            continue
        p = image_path(img_id)
        if not os.path.exists(p):
            X[i] = 0.0
        else:
            X[i] = extract_features(p)
        _FEATURE_CACHE[img_id] = X[i]
    return X


def iterative_multilabel_split(y_bin: np.ndarray, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n = y_bin.shape[0]
    n_test = int(round(n * test_size))
    n_test = max(1, min(n - 1, n_test))

    pos_total = y_bin.sum(axis=0).astype(int)
    desired_test_pos = np.rint(pos_total * (n_test / n)).astype(int)

    remaining = np.ones(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    current_test_pos = np.zeros(y_bin.shape[1], dtype=int)

    for _ in range(n_test):
        candidates = np.where(remaining)[0]
        need = desired_test_pos - current_test_pos
        label_order = np.argsort(-need)
        chosen = None
        for lbl in label_order:
            if need[lbl] <= 0:
                continue
            cand_lbl = candidates[y_bin[candidates, lbl] == 1]
            if cand_lbl.size > 0:
                sums = y_bin[cand_lbl].sum(axis=1)
                mx = sums.max()
                top = cand_lbl[sums == mx]
                chosen = int(rng.choice(top))
                break
        if chosen is None:
            chosen = int(rng.choice(candidates))
        test_mask[chosen] = True
        remaining[chosen] = False
        current_test_pos += y_bin[chosen].astype(int)

    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


X = build_features(train_df["image_id"])
y = train_df[TARGET_COLS].values.astype(int)

tr_idx, va_idx = iterative_multilabel_split(y, test_size=0.2, random_state=42)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000, solver="lbfgs", class_weight="balanced", C=3.0
                )
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)
auc_cols = []
for j, c in enumerate(TARGET_COLS):
    if len(np.unique(y_va[:, j])) < 2:
        continue
    auc_cols.append(roc_auc_score(y_va[:, j], va_pred[:, j]))
if len(auc_cols) > 0:
    print("Local mean column-wise ROC AUC (valid):", float(np.mean(auc_cols)))
else:
    print("Local AUC could not be computed (constant labels in validation split).")

clf.fit(X, y)

X_test = build_features(test_df["image_id"])
test_pred = clf.predict_proba(X_test)

pred_df = pd.DataFrame(test_pred, columns=TARGET_COLS)
pred_df.insert(0, "image_id", test_df["image_id"].values)

make_submission_file_from_df(pred_df, out_path="submission.csv")

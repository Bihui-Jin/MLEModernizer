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




## === cell 1
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




## === cell 2
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




## === cell 3
import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL is required to read images. Please ensure Pillow is available in the environment."
    ) from e

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed, cpu_count

np.random.seed(42)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _build_image_path_map(image_ids):
    id_set = set(image_ids)
    mapping = {}
    for d in IMAGES_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".jpg"):
                continue
            img_id = fn[:-4]
            if img_id in id_set and img_id not in mapping:
                mapping[img_id] = os.path.join(d, fn)
    default_dir = IMAGES_DIR_CANDIDATES[0]
    for img_id in id_set:
        if img_id not in mapping:
            mapping[img_id] = os.path.join(default_dir, f"{img_id}.jpg")
    return mapping


_all_ids = (
    pd.concat([train_df["image_id"], test_df["image_id"]], axis=0).astype(str).tolist()
)
_IMAGE_PATH_MAP = _build_image_path_map(_all_ids)


def image_path(image_id: str) -> str:
    return _IMAGE_PATH_MAP.get(
        str(image_id), os.path.join(IMAGES_DIR_CANDIDATES[0], f"{image_id}.jpg")
    )


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


def _gray_world(arr01: np.ndarray) -> np.ndarray:
    eps = 1e-6
    means = arr01.mean(axis=(0, 1))
    m = float(means.mean())
    gains = m / (means + eps)
    out = arr01 * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _resize01(arr01: np.ndarray, size_hw) -> np.ndarray:
    im = Image.fromarray(np.clip(arr01 * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    im = im.resize((size_hw[1], size_hw[0]), resample=_RESAMPLE)
    return (np.asarray(im, dtype=np.float32) / 255.0).astype(np.float32)


def extract_features_from_arr01(
    arr_full01: np.ndarray,
    size_rgb=(96, 96),
    size_rgb2=(160, 160),
    thumb_gray=(32, 32),
) -> np.ndarray:
    arr_full01 = _foreground_crop(arr_full01)
    arr_full01 = _gray_world(arr_full01)

    arr1 = _resize01(arr_full01, (size_rgb[0], size_rgb[1]))
    arr2 = _resize01(arr_full01, (size_rgb2[0], size_rgb2[1]))

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
    gray_img = gray_img.resize((thumb_gray[1], thumb_gray[0]), resample=_RESAMPLE)
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


def extract_features(
    img_path: str,
    size_rgb=(96, 96),
    size_rgb2=(160, 160),
    thumb_gray=(32, 32),
    do_flip: bool = False,
) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        arr_full = (np.asarray(im, dtype=np.float32) / 255.0).astype(np.float32)

    if do_flip:
        arr_full = arr_full[:, ::-1, :]

    return extract_features_from_arr01(
        arr_full, size_rgb=size_rgb, size_rgb2=size_rgb2, thumb_gray=thumb_gray
    )


_FEATURE_CACHE = {}


def _extract_both(img_id: str):
    img_id = str(img_id)
    k0 = (img_id, False)
    k1 = (img_id, True)
    if k0 in _FEATURE_CACHE and k1 in _FEATURE_CACHE:
        return (k0, _FEATURE_CACHE[k0]), (k1, _FEATURE_CACHE[k1])

    p = image_path(img_id)
    if not os.path.exists(p):
        return (k0, None), (k1, None)

    with Image.open(p) as im:
        im = im.convert("RGB")
        arr_full = (np.asarray(im, dtype=np.float32) / 255.0).astype(np.float32)

    feat0 = extract_features_from_arr01(
        arr_full, size_rgb=(96, 96), size_rgb2=(160, 160), thumb_gray=(32, 32)
    )
    feat1 = extract_features_from_arr01(
        arr_full[:, ::-1, :],
        size_rgb=(96, 96),
        size_rgb2=(160, 160),
        thumb_gray=(32, 32),
    )
    return (k0, feat0), (k1, feat1)


def build_features(image_ids: pd.Series, do_flip: bool = False) -> np.ndarray:
    ids = image_ids.astype(str).tolist()

    feat_len = None
    for img_id in ids:
        cache_key = (img_id, bool(do_flip))
        if cache_key in _FEATURE_CACHE:
            feat_len = _FEATURE_CACHE[cache_key].shape[0]
            break
        p = image_path(img_id)
        if os.path.exists(p):
            feat_len = extract_features(p, do_flip=do_flip).shape[0]
            break
    if feat_len is None:
        raise RuntimeError(
            "No images found to infer feature dimension. Check IMAGES_DIR paths."
        )

    X = np.zeros((len(ids), feat_len), dtype=np.float32)

    missing_img_ids = []
    for img_id in ids:
        k0 = (img_id, False)
        k1 = (img_id, True)
        if (k0 not in _FEATURE_CACHE) or (k1 not in _FEATURE_CACHE):
            missing_img_ids.append(img_id)

    if missing_img_ids:
        n_jobs = min(max(1, cpu_count() - 1), 8)
        results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=8)(
            delayed(_extract_both)(img_id) for img_id in missing_img_ids
        )
        for (k0, f0), (k1, f1) in results:
            if f0 is None:
                _FEATURE_CACHE[k0] = np.zeros((feat_len,), dtype=np.float32)
            else:
                _FEATURE_CACHE[k0] = f0
            if f1 is None:
                _FEATURE_CACHE[k1] = np.zeros((feat_len,), dtype=np.float32)
            else:
                _FEATURE_CACHE[k1] = f1

    for i, img_id in enumerate(ids):
        X[i] = _FEATURE_CACHE[(img_id, bool(do_flip))]
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


X0 = build_features(train_df["image_id"], do_flip=False)
X1 = build_features(train_df["image_id"], do_flip=True)
X = 0.5 * X0 + 0.5 * X1

y_bin = train_df[TARGET_COLS].values.astype(int)
y_class = y_bin.argmax(axis=1).astype(int)

row_sums = y_bin.sum(axis=1)
is_single_label = bool(np.all(row_sums == 1))
print("Detected single-label one-hot dataset:", is_single_label)

tr_idx, va_idx = iterative_multilabel_split(y_bin, test_size=0.2, random_state=42)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr_bin, y_va_bin = y_bin[tr_idx], y_bin[va_idx]
y_tr_class, y_va_class = y_class[tr_idx], y_class[va_idx]

class_counts = np.bincount(y_tr_class, minlength=len(TARGET_COLS)).astype(np.float64)
class_counts = np.maximum(class_counts, 1.0)
class_weight_per_class = (len(y_tr_class) / (len(TARGET_COLS) * class_counts)).astype(
    np.float64
)
sample_weight_tr = class_weight_per_class[y_tr_class]

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                C=3.0,
                multi_class="multinomial",
                n_jobs=None,
            ),
        ),
    ]
)

if is_single_label:
    clf.fit(X_tr, y_tr_class, lr__sample_weight=sample_weight_tr)
    va_proba = clf.predict_proba(X_va)
    va_pred = va_proba
else:
    from sklearn.multiclass import OneVsRestClassifier

    ovr_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight=None,
                        C=3.0,
                    )
                ),
            ),
        ]
    )
    pos_rate = np.clip(y_tr_bin.mean(axis=0).astype(np.float64), 1e-6, 1.0 - 1e-6)
    w_pos = (1.0 - pos_rate) / pos_rate
    sample_weight_tr_ml = (
        (y_tr_bin * w_pos.reshape(1, -1) + (1 - y_tr_bin) * 1.0)
        .mean(axis=1)
        .astype(np.float64)
    )
    try:
        ovr_clf.fit(X_tr, y_tr_bin, ovr__estimator__sample_weight=sample_weight_tr_ml)
    except TypeError as e:
        print(
            "Warning: sample_weight could not be passed to estimator (continuing unweighted). Error:",
            repr(e),
        )
        ovr_clf.fit(X_tr, y_tr_bin)
    va_pred = ovr_clf.predict_proba(X_va)
    clf = ovr_clf

auc_cols = []
for j, c in enumerate(TARGET_COLS):
    if len(np.unique(y_va_bin[:, j])) < 2:
        continue
    auc_cols.append(roc_auc_score(y_va_bin[:, j], va_pred[:, j]))
if len(auc_cols) > 0:
    print("Local mean column-wise ROC AUC (valid):", float(np.mean(auc_cols)))
else:
    print("Local AUC could not be computed (constant labels in validation split).")

if is_single_label:
    class_counts_full = np.bincount(y_class, minlength=len(TARGET_COLS)).astype(
        np.float64
    )
    class_counts_full = np.maximum(class_counts_full, 1.0)
    class_weight_per_class_full = (
        len(y_class) / (len(TARGET_COLS) * class_counts_full)
    ).astype(np.float64)
    sample_weight_full = class_weight_per_class_full[y_class]
    clf.fit(X, y_class, lr__sample_weight=sample_weight_full)
else:
    pos_rate_full = np.clip(y_bin.mean(axis=0).astype(np.float64), 1e-6, 1.0 - 1e-6)
    w_pos_full = (1.0 - pos_rate_full) / pos_rate_full
    sample_weight_full_ml = (
        (y_bin * w_pos_full.reshape(1, -1) + (1 - y_bin) * 1.0)
        .mean(axis=1)
        .astype(np.float64)
    )
    try:
        clf.fit(X, y_bin, ovr__estimator__sample_weight=sample_weight_full_ml)
    except TypeError as e:
        print(
            "Warning: sample_weight could not be passed to estimator (continuing unweighted). Error:",
            repr(e),
        )
        clf.fit(X, y_bin)

X_test = build_features(test_df["image_id"], do_flip=False)
X_test_flip = build_features(test_df["image_id"], do_flip=True)

if is_single_label:
    test_pred = 0.5 * clf.predict_proba(X_test) + 0.5 * clf.predict_proba(X_test_flip)
else:
    test_pred = 0.5 * clf.predict_proba(X_test) + 0.5 * clf.predict_proba(X_test_flip)

pred_df = pd.DataFrame(test_pred, columns=TARGET_COLS)
pred_df.insert(0, "image_id", test_df["image_id"].values)

make_submission_file_from_df(pred_df, out_path="submission.csv")

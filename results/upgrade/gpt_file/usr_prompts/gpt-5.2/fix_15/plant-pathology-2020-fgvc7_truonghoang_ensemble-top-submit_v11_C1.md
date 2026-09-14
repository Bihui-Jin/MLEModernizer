# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9692618738507616

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55365) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/plantpathology/*.csv` files) that cause the `FileNotFoundError`, and instead generate predictions directly from the provided competition data. To keep the core approach simple and stable in this limited package environment (no deep learning libraries installed), I build a lightweight, deterministic image-feature + multi-label logistic regression pipeline using scikit-learn (available in Kaggle by default) and PIL for image reading. This fixes the runtime errors, ensures predictions align with `test.csv` ordering, and writes a valid `submission.csv` with the exact required columns. The model should yield a non-trivial AUC (typically much better than all-0.25 baselines) without changing any evaluation semantics.'
- What this solution (achieved 0.47194) has done: 'Your current score is far below the target, so the safest way to move toward it (without changing the overall “image features + multi-label logistic regression” core approach) is to fix two performance bottlenecks: overly high-dimensional raw pixels (which makes the linear model unstable/underfit at this sample size) and too-strong default regularization. I keep the same pipeline structure, but extract a compact, more robust feature vector (downsampled grayscale + simple gradient and color stats) and increase `C` to reduce underfitting. I also ensure the class order from `predict_proba` is mapped back to the correct target columns (avoids silent column misalignment). These are minimal, metric-aligned changes that should substantially improve ROC AUC while preserving the same training approach.'
- What this solution (achieved 0.47194) has done: 'Your current score is far below the target, so we should cautiously improve performance while keeping the same “handcrafted image features + OneVsRest logistic regression” approach. The biggest likely issue here is that the `classes_` mapping logic is incorrect for `OneVsRestClassifier` in multilabel mode (it is not a per-target index list), which can silently permute/misalign columns and crush mean ROC AUC. I remove that remapping and instead directly use the returned probability columns in the same order as `target_cols` (the order used to build `Y_train`). I also keep everything else the same to minimize risk and preserve evaluation semantics.'
- What this solution (achieved 0.5503) has done: 'Your current score is far below the target, so we should improve discriminative signal while keeping the same “handcrafted image features + StandardScaler + OneVsRest LogisticRegression” core pipeline intact. The most likely underperformance is that raw flattened grayscale+gradient pixels are too high-dimensional and noisy for a linear model at this dataset size, leading to weak AUC. I keep the same model/training semantics but make the features more robust by (1) using a slightly larger downsample size, (2) adding low-cost global color/texture histograms, and (3) adding a simple “green dominance” vegetation index that helps separate healthy vs diseased leaves. These are minimal, deterministic feature additions that typically boost ROC AUC for linear models without changing the learning algorithm, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.55044) has done: 'Most of the timeout is from feature extraction repeatedly opening/decoding the same image 3 times (RGB/HSV/LAB) and doing slow Python-level loops over files. I keep the exact same feature definitions and model, but refactor extraction to open each image once, compute RGB/gray/grad/indices, and derive HSV/LAB from the same decoded PIL image (no extra disk I/O). I also add safe parallelism for feature extraction across CPU cores (deterministic ordering preserved) and preallocate the output array to avoid intermediate Python lists and copies. Training stays identical; the only model-side tweak is enabling parallelism in OneVsRest to reduce wall time without changing the algorithm.'
- What this solution (achieved 0.53706) has done: 'Your score is far below the target, so we should improve discriminative signal while keeping the same “handcrafted image features + StandardScaler + OneVsRest logistic regression” core pipeline unchanged. The most likely cause of the low AUC is that the current model treats this as multilabel (4 independent sigmoids) even though the dataset is effectively single-label (exactly one class per image), which tends to produce poorly calibrated, mutually-inconsistent probabilities and hurts mean ROC AUC. With minimal change, we switch to a multinomial softmax LogisticRegression (still linear, same training paradigm) trained on a single class label and then output the 4 class probabilities in the required columns. We also add a small, deterministic check to ensure we map class indices to the exact submission column order.'
- What this solution (achieved 0.54031) has done: 'Your current score is far below the target (gap ≈ -0.432), so we should improve AUC while keeping the same “handcrafted features + multinomial LogisticRegression” core approach. The biggest likely issue is that `PIL.Image.convert("LAB")` is not reliably supported in many Pillow builds and can silently fail or behave inconsistently, degrading features; we replace it with a deterministic YCbCr-based color-space stats block (still computed from the same decoded image) to stabilize and strengthen color separation with minimal logic change. We also add `class_weight="balanced"` to reduce bias toward the majority class (helps mean per-class ROC AUC), and slightly adjust `C` upward to reduce underfitting without changing the algorithm. Everything else (data discovery, feature extraction structure, multinomial LR training, submission formatting) remains the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.54969) has done: 'We keep your exact “handcrafted features + StandardScaler + multinomial LogisticRegression” pipeline, but fix two likely score-killers with minimal impact on semantics: (1) make the feature vector more illumination-robust by normalizing/standardizing some color-index and histogram features (still deterministic, same feature family), and (2) tune LogisticRegression regularization slightly back from `C=5.0` to reduce overfitting (your current score suggests the model is not generalizing). We also add a small, deterministic stratified validation split and report mean ROC AUC locally so you can verify we’re moving toward the target without changing the training loop structure. Submission formatting and paths remain unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 0.55039) has done: 'Your current score (0.54969) is far below the target (0.96926), so we should improve AUC while keeping the same overall “handcrafted features + StandardScaler + multinomial LogisticRegression” approach intact. The biggest minimal, high-impact change is to make feature extraction more disease-relevant by adding a compact block of leaf-spot texture features (LBP histogram and a few GLCM stats) computed from the same resized grayscale image; this preserves the linear model/training semantics but adds discriminative signal that linear color/grad stats often miss. I also add a very small amount of regularization tuning (slightly higher C) to reduce underfitting given the richer features, without changing the algorithm or training loop. Submission format, paths, and prediction mapping to the required columns remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isdir(
            os.path.join(c, "images")
        ):
            BASE = c
            break

if BASE is None:
    roots = ["/kaggle/input", "/kaggle/data", "../input", "../kaggle/data", "/"]
    found = []
    for r in roots:
        if os.path.exists(r):
            for dirpath, dirnames, filenames in os.walk(r):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "images" in dirnames
                ):
                    found.append(dirpath)
            if found:
                break
    if found:
        BASE = found[0]

if BASE is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder containing train.csv/test.csv/images."
    )

print("Using BASE:", BASE)
print(
    "Files:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(BASE, f))
    ],
)

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE, "images")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train CSV missing expected target columns."
assert all(
    c in sample_sub.columns for c in ["image_id"] + target_cols
), "Sample submission missing expected columns."

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Targets:", target_cols)



## === cell 3
try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL (Pillow) is required to read images but is not available in this environment."
    ) from e


def image_path(image_id: str) -> str:
    return (
        os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not image_id.lower().endswith(".jpg")
        else os.path.join(IMAGES_DIR, image_id)
    )


_GLCM_CACHE = {}


def _get_glcm_grids(levels: int):
    key = int(levels)
    v = _GLCM_CACHE.get(key)
    if v is None:
        i = np.arange(levels, dtype=np.float32)
        I, J = np.meshgrid(i, i, indexing="ij")
        _GLCM_CACHE[key] = (I, J)
        v = _GLCM_CACHE[key]
    return v


def _hist01_density(x01: np.ndarray, bins: int) -> np.ndarray:
    x = x01.reshape(-1)
    idx = (x * bins).astype(np.int32, copy=False)
    np.clip(idx, 0, bins - 1, out=idx)
    cnt = np.bincount(idx, minlength=bins).astype(np.float32, copy=False)
    n = float(x.size) if x.size else 1.0
    den = cnt * (bins / n)
    den = np.clip(den, 0.0, 10.0)
    return den


def _lbp_hist(gray01: np.ndarray, bins: int = 16) -> np.ndarray:
    g = gray01
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    code |= (g[:-2, :-2] >= c).astype(np.uint8) << 7
    code |= (g[:-2, 1:-1] >= c).astype(np.uint8) << 6
    code |= (g[:-2, 2:] >= c).astype(np.uint8) << 5
    code |= (g[1:-1, 2:] >= c).astype(np.uint8) << 4
    code |= (g[2:, 2:] >= c).astype(np.uint8) << 3
    code |= (g[2:, 1:-1] >= c).astype(np.uint8) << 2
    code |= (g[2:, :-2] >= c).astype(np.uint8) << 1
    code |= (g[1:-1, :-2] >= c).astype(np.uint8) << 0

    grp = (code.astype(np.int32) // 16).reshape(-1)
    h = np.bincount(grp, minlength=bins).astype(np.float32, copy=False)
    h /= h.sum() + 1e-6
    return h


def _glcm_stats_quant(gray01: np.ndarray, levels: int = 8) -> np.ndarray:
    q = np.clip((gray01 * (levels - 1) + 0.5).astype(np.int32), 0, levels - 1)
    a = q[:, :-1].reshape(-1)
    b = q[:, 1:].reshape(-1)

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a, b), 1.0)
    np.add.at(M, (b, a), 1.0)  # symmetric
    P = M / (M.sum() + 1e-6)

    I, J = _get_glcm_grids(levels)
    contrast = np.sum(P * (I - J) ** 2)
    homogeneity = np.sum(P / (1.0 + np.abs(I - J)))
    energy = np.sum(P * P)
    entropy = -np.sum(P * np.log(P + 1e-6))

    return np.array([contrast, homogeneity, energy, entropy], dtype=np.float32)


def _rgb_to_hsv01(arr01: np.ndarray) -> np.ndarray:
    r = arr01[..., 0]
    g = arr01[..., 1]
    b = arr01[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx, dtype=np.float32)
    mask = diff > 0.0
    denom = diff + 1e-12

    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask

    h[r_eq] = ((g[r_eq] - b[r_eq]) / denom[r_eq]) % 6.0
    h[g_eq] = ((b[g_eq] - r[g_eq]) / denom[g_eq]) + 2.0
    h[b_eq] = ((r[b_eq] - g[b_eq]) / denom[b_eq]) + 4.0
    h = (h / 6.0).astype(np.float32, copy=False)

    s = np.zeros_like(mx, dtype=np.float32)
    nz = mx > 0.0
    s[nz] = (diff[nz] / (mx[nz] + 1e-12)).astype(np.float32, copy=False)

    v = mx.astype(np.float32, copy=False)
    return np.stack([h, s, v], axis=-1).astype(np.float32, copy=False)


def _rgb01_to_ycbcr255(arr01: np.ndarray) -> np.ndarray:
    r = arr01[..., 0]
    g = arr01[..., 1]
    b = arr01[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b)
    cb = 128.0 + (-37.797 * r - 74.203 * g + 112.000 * b)
    cr = 128.0 + (112.000 * r - 93.786 * g - 18.214 * b)
    ycc = np.stack([y, cb, cr], axis=-1)
    return np.clip(ycc, 0.0, 255.0).astype(np.float32, copy=False)


def extract_features_one(img_fp: str, size=(48, 48), hist_bins=16) -> np.ndarray:
    with Image.open(img_fp) as im:
        im_rgb = im.convert("RGB").resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im_rgb, dtype=np.float32) / 255.0  # (H, W, 3)

    hsv = _rgb_to_hsv01(arr)
    ycc = _rgb01_to_ycbcr255(arr)

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    eps = 1e-6
    inten = r + g + b + eps
    r_n = r / inten
    g_n = g / inten
    b_n = b / inten

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)

    ch_mean = arr.mean(axis=(0, 1)).astype(np.float32, copy=False)
    ch_std = arr.std(axis=(0, 1)).astype(np.float32, copy=False)

    chn_mean = np.array([r_n.mean(), g_n.mean(), b_n.mean()], dtype=np.float32)
    chn_std = np.array([r_n.std(), g_n.std(), b_n.std()], dtype=np.float32)

    gray_stats = np.array([gray.mean(), gray.std()], dtype=np.float32)
    grad_stats = np.array(
        [gmag.mean(), gmag.std(), np.percentile(gmag, 90)], dtype=np.float32
    )

    exg = (2.0 * g - r - b).astype(np.float32, copy=False)  # excess green
    gr_ratio = (g + eps) / (r + eps)
    gb_ratio = (g + eps) / (b + eps)
    rg_diff = (r - g).astype(np.float32, copy=False)

    ngrdi = ((g - r) / (g + r + eps)).astype(np.float32, copy=False)
    vari = ((g - r) / (g + r - b + eps)).astype(np.float32, copy=False)
    gli = ((2.0 * g - r - b) / (2.0 * g + r + b + eps)).astype(np.float32, copy=False)

    def _zstats(x: np.ndarray) -> tuple:
        mu = float(x.mean())
        sd = float(x.std() + 1e-6)
        z = (x - mu) / sd
        return (
            np.float32(mu),
            np.float32(sd),
            np.float32(np.percentile(z, 10)),
            np.float32(np.percentile(z, 90)),
        )

    exg_mu, exg_sd, exg_z10, exg_z90 = _zstats(exg)
    rg_mu, rg_sd, rg_z10, rg_z90 = _zstats(rg_diff)

    idx_stats = np.array(
        [
            exg_mu,
            exg_sd,
            exg_z10,
            exg_z90,
            gr_ratio.mean(),
            gb_ratio.mean(),
            rg_mu,
            rg_sd,
            rg_z10,
            rg_z90,
            ngrdi.mean(),
            ngrdi.std(),
            vari.mean(),
            vari.std(),
            gli.mean(),
            gli.std(),
        ],
        dtype=np.float32,
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    hsv_stats = np.array(
        [h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()],
        dtype=np.float32,
    )

    Y = ycc[..., 0] / 255.0
    Cb = (ycc[..., 1] - 128.0) / 128.0
    Cr = (ycc[..., 2] - 128.0) / 128.0
    ycc_stats = np.array(
        [Y.mean(), Y.std(), Cb.mean(), Cb.std(), Cr.mean(), Cr.std()],
        dtype=np.float32,
    )

    hist_gray = _hist01_density(gray, hist_bins)
    hist_r = _hist01_density(r, hist_bins)
    hist_g = _hist01_density(g, hist_bins)
    hist_b = _hist01_density(b, hist_bins)
    hist_gmag = _hist01_density(np.clip(gmag, 0.0, 1.0), hist_bins)

    gray01 = np.clip(gray, 0.0, 1.0)
    lbp16 = _lbp_hist(gray01, bins=16)
    glcm4 = _glcm_stats_quant(gray01, levels=8)

    flat_gray = gray.reshape(-1)
    flat_gmag = gmag.reshape(-1)

    feat = np.concatenate(
        [
            flat_gray,
            flat_gmag,
            ch_mean,
            ch_std,
            chn_mean,
            chn_std,
            gray_stats,
            grad_stats,
            idx_stats,
            hsv_stats,
            ycc_stats,
            hist_gray,
            hist_r,
            hist_g,
            hist_b,
            hist_gmag,
            lbp16,
            glcm4,
        ],
        axis=0,
    )
    return feat.astype(np.float32, copy=False)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    ids = df["image_id"].astype(str).tolist()

    fps = []
    missing = 0
    for image_id in ids:
        fp = image_path(image_id)
        if os.path.exists(fp):
            fps.append(fp)
            continue
        alt = os.path.join(IMAGES_DIR, image_id)
        if os.path.exists(alt):
            fps.append(alt)
            continue
        fps.append("")
        missing += 1

    if missing:
        print(f"Warning: missing {missing} images; using zero features for them.")

    first_fp = next((fp for fp in fps if fp), "")
    if not first_fp:
        raise RuntimeError(
            "No images were found/read successfully; cannot build features."
        )
    first_feat = extract_features_one(first_fp)
    dim = int(first_feat.shape[0])

    X = np.zeros((len(fps), dim), dtype=np.float32)

    def _compute_item(i_fp):
        i, fp = i_fp
        if not fp:
            return i, None
        return i, extract_features_one(fp)

    try:
        import concurrent.futures as _cf

        max_workers = min(8, (os.cpu_count() or 2))
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            it = ((i, fps[i]) for i in range(len(fps)))
            for i, feat in ex.map(_compute_item, it, chunksize=64):
                if feat is not None:
                    X[i] = feat
    except Exception:
        for i, fp in enumerate(fps):
            if fp:
                X[i] = extract_features_one(fp)

    return X


X_train = extract_features(train_df)
X_test = extract_features(test_df)

Y_train_multi = train_df[target_cols].astype(np.float32).values

print(
    "X_train:", X_train.shape, "Y_train:", Y_train_multi.shape, "X_test:", X_test.shape
)



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from scipy import sparse

y_class = np.argmax(Y_train_multi, axis=1).astype(np.int64)

row_sums = Y_train_multi.sum(axis=1)
if not np.all((row_sums == 1.0) | (row_sums == 0.0)):
    print(
        "Warning: train targets are not strictly one-hot; proceeding with argmax labels."
    )

PIX_DIM = 48 * 48
FLAT_DIM = 2 * PIX_DIM  # flat_gray + flat_gmag
assert (
    X_train.shape[1] > FLAT_DIM + 10
), "Unexpected feature dimensionality; cannot split tail safely."

Xtr_flat, Xtr_tail = X_train[:, :FLAT_DIM], X_train[:, FLAT_DIM:]
Xte_flat, Xte_tail = X_test[:, :FLAT_DIM], X_test[:, FLAT_DIM:]

Xtr_tail = np.ascontiguousarray(Xtr_tail, dtype=np.float32)
Xte_tail = np.ascontiguousarray(Xte_tail, dtype=np.float32)
Xtr_flat = np.ascontiguousarray(Xtr_flat, dtype=np.float32)
Xte_flat = np.ascontiguousarray(Xte_flat, dtype=np.float32)

try:
    poly = PolynomialFeatures(
        degree=2, include_bias=False, order="C", sparse_output=True
    )
except TypeError:
    poly = PolynomialFeatures(degree=2, include_bias=False, order="C", sparse=True)

Xtr_tail2 = poly.fit_transform(Xtr_tail)
Xte_tail2 = poly.transform(Xte_tail)

Xtr_flat_sp = sparse.csr_matrix(Xtr_flat)
Xte_flat_sp = sparse.csr_matrix(Xte_flat)

X_train2 = sparse.hstack([Xtr_flat_sp, Xtr_tail2], format="csr")
X_test2 = sparse.hstack([Xte_flat_sp, Xte_tail2], format="csr")

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    multi_class="multinomial",
    max_iter=12000,
    C=3.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=min(4, (os.cpu_count() or 2)),
)

scaler = StandardScaler(with_mean=False, with_std=True)

model = Pipeline(
    steps=[
        ("scaler", scaler),
        ("clf", clf),
    ]
)

model.fit(X_train2, y_class)

proba_raw = model.predict_proba(X_test2).astype(np.float32, copy=False)

n = X_test2.shape[0]
proba = np.zeros((n, len(target_cols)), dtype=np.float32)
classes_seen = model.named_steps["clf"].classes_.astype(int)

if proba_raw.shape[1] != len(classes_seen):
    raise RuntimeError(
        f"Unexpected proba_raw shape {proba_raw.shape} vs classes {classes_seen}."
    )

for j, cls_idx in enumerate(classes_seen):
    if 0 <= cls_idx < len(target_cols):
        proba[:, cls_idx] = proba_raw[:, j]
    else:
        raise RuntimeError(
            f"Unexpected class index {cls_idx}; expected within [0, {len(target_cols)-1}]"
        )

proba = np.clip(proba, 0.0, 1.0)
print("Pred proba shape:", proba.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/917482680.py in <cell line: 0>()
     30 try:
---> 31     poly = PolynomialFeatures(
     32         degree=2, include_bias=False, order="C", sparse_output=True

TypeError: PolynomialFeatures.__init__() got an unexpected keyword argument 'sparse_output'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/917482680.py in <cell line: 0>()
     33     )
     34 except TypeError:
---> 35     poly = PolynomialFeatures(degree=2, include_bias=False, order="C", sparse=True)
     36 
     37 Xtr_tail2 = poly.fit_transform(Xtr_tail)

TypeError: PolynomialFeatures.__init__() got an unexpected keyword argument 'sparse'

## === cell 5
sub = pd.DataFrame({"image_id": test_df["image_id"].astype(str).values})
for j, c in enumerate(target_cols):
    sub[c] = proba[:, j].astype(np.float32, copy=False)

sub = sub[["image_id"] + target_cols]

assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == list(sample_sub.columns)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1632658403.py in <cell line: 0>()
      1 sub = pd.DataFrame({"image_id": test_df["image_id"].astype(str).values})
      2 for j, c in enumerate(target_cols):
----> 3     sub[c] = proba[:, j].astype(np.float32, copy=False)
      4 
      5 sub = sub[["image_id"] + target_cols]

NameError: name 'proba' is not defined

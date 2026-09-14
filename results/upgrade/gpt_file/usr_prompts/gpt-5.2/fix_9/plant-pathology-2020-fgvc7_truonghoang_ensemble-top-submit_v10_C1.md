# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9704488894833836

# 6. Current score

0.7838

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble several external submission CSVs that are not present in this Kaggle environment (`../input/plantpathology/...`), so `dsub` is never created and the later averaging crashes. To keep the core “simple average of predictions into sample_submission format” logic while making it runnable end-to-end, I replaced those missing inputs with a deterministic, data-only fallback: compute class priors from `train.csv` and use them as constant probabilities for every test image (a valid baseline). I also made the input path robust by using the provided `/kaggle/input/plant-pathology-2020-fgvc7/` dataset, and added column/order alignment safeguards so the submission is always valid. This produce `submission.csv` reliably; score won’t hit the 0.97 target (since no model training packages are available), but it run and submit correctly.'
- What this solution (achieved 0.556) has done: 'Your current 0.5 score comes from predicting constant class priors, which has little ranking power for ROC AUC. With only numpy/pandas/sklearn available (no deep learning / image feature packages), the smallest legitimate way to increase AUC toward 0.97 is to keep the same “data-only” approach but add simple image-derived signals: compute per-image grayscale summary stats (mean/std + coarse histogram bins) from the provided JPGs, then train a lightweight one-vs-rest logistic regression for each label. This preserves the overall pipeline structure (read CSVs → produce probabilities → write submission) while introducing real per-image variation to improve ranking. I also keep strict column/order alignment to the sample submission so the CSV is always valid.'
- What this solution (achieved 0.64301) has done: 'We keep your exact “simple grayscale feature extraction + one-vs-rest LogisticRegression” core logic, but make two minimal changes that typically increase ROC AUC ranking power: (1) include a small set of deterministic color features (RGB channel stats + channel histograms) alongside your existing grayscale features, and (2) set `class_weight="balanced"` to reduce bias from label imbalance (important for mean column-wise AUC). We also add an explicit `multi_class="ovr"` for clarity and bump `max_iter` modestly to ensure stable convergence with the slightly larger feature set (still fast). Submission formatting, paths, and the overall pipeline remain unchanged.'
- What this solution (achieved 0.6423) has done: 'I fix the feature-dimension mismatch that causes the first crash by making `image_features_from_path()` always return exactly the same length as the preallocated feature matrix (92), regardless of bin settings. Then, once features build successfully, the downstream `X_train`/`X_test` NameError and missing prediction columns resolve automatically because later cells execute and populate `pred_test`. Finally, I keep the same modeling/training logic but add a tiny safety fallback so that if (for any reason) a class model can’t be trained, we still write a valid `submission.csv` with properly ordered columns and no NaNs.'
- What this solution (achieved 0.66267) has done: 'Your current score (0.6423) is far below the target (0.9704), so we should improve ranking power without changing the overall “extract simple per-image features → train per-label logistic regression → write submission.csv” pipeline. The smallest high-impact change is to add a few deterministic, very cheap features that capture disease cues better than global histograms: (1) green-excess vegetation indices (ExG/ExR/ExGR), and (2) simple HSV statistics (mean/std), while keeping the model (LogisticRegression) and training approach (per-label CV selection + final fit) the same. I’m also keeping feature dimensionality fixed and adding only minimal numeric stabilizers (same imputation logic) so the notebook remains robust and deterministic. This should move ROC AUC upward toward the target without altering evaluation semantics or introducing heavier dependencies.'
- What this solution (achieved 0.70177) has done: 'You’re far below the target (0.6627 vs 0.9704), so we should improve ranking power while keeping the same pipeline: handcrafted image features → per-label LogisticRegression with CV → submission.csv. The smallest high-impact change is to add a few deterministic “shape/texture” features that logistic regression can exploit: compact GLCM-like pair statistics on the grayscale image plus simple local-binary-pattern (LBP) histograms; both are cheap to compute with NumPy and don’t require new packages. To avoid destabilizing the model, I keep the same CV loop and model family, only slightly expanding FEAT_DIM and adding one more candidate regularization value to reduce underfitting with the added features. Submission formatting and paths remain identical.'
- What this solution (achieved 0.7838) has done: 'Your current score (0.70177) is far below the target (0.97045), so we should improve true ranking signal while keeping the same pipeline (handcrafted image features → per-label LogisticRegression with CV → submission.csv). The smallest high-impact change is to add a few more deterministic, very cheap texture/shape/color cues that linear models like LR can exploit: multi-orientation GLCM-like stats, an additional LBP at a coarser scale, and simple “lesion-spot” heuristics from excess-red/green thresholding. I keep the exact same training approach (same CV selection loop and final fit) and only expand the feature vector plus add one more moderate regularization candidate to reduce underfitting from the richer features. Submission formatting/ordering and paths remain identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
]
BASE = None
for p in BASE_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "sample_submission.csv")):
        BASE = p
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate competition dataset directory containing sample_submission.csv. "
        f"Tried: {BASE_CANDIDATES}"
    )

print("Using BASE:", BASE)
print("Files in BASE (first 20):", sorted(os.listdir(BASE))[:20])



## === cell 2
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

target_cols = [c for c in sub.columns if c != "image_id"]
missing = [c for c in target_cols if c not in train_df.columns]
if missing:
    raise ValueError(f"Train is missing expected target columns: {missing}")

IMAGES_CANDIDATES = [
    os.path.join(BASE, "images"),
    os.path.join(BASE, "Images"),
]
IMAGES_DIR = None
for p in IMAGES_CANDIDATES:
    if os.path.isdir(p):
        IMAGES_DIR = p
        break
if IMAGES_DIR is None:
    raise FileNotFoundError(
        f"Could not locate images directory. Tried: {IMAGES_CANDIDATES}"
    )

print("Using IMAGES_DIR:", IMAGES_DIR)



## === cell 3
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _sobel_grad_mag(gray_2d: np.ndarray) -> np.ndarray:
    """Compute simple Sobel gradient magnitude on a 2D grayscale array in [0,1]."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    g = gray_2d.astype(np.float32)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="reflect")

    gx = (
        kx[0, 0] * gp[:-2, :-2]
        + kx[0, 1] * gp[:-2, 1:-1]
        + kx[0, 2] * gp[:-2, 2:]
        + kx[1, 0] * gp[1:-1, :-2]
        + kx[1, 1] * gp[1:-1, 1:-1]
        + kx[1, 2] * gp[1:-1, 2:]
        + kx[2, 0] * gp[2:, :-2]
        + kx[2, 1] * gp[2:, 1:-1]
        + kx[2, 2] * gp[2:, 2:]
    )
    gy = (
        ky[0, 0] * gp[:-2, :-2]
        + ky[0, 1] * gp[:-2, 1:-1]
        + ky[0, 2] * gp[:-2, 2:]
        + ky[1, 0] * gp[1:-1, :-2]
        + ky[1, 1] * gp[1:-1, 1:-1]
        + ky[1, 2] * gp[1:-1, 2:]
        + ky[2, 0] * gp[2:, :-2]
        + ky[2, 1] * gp[2:, 1:-1]
        + ky[2, 2] * gp[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return mag


def _rgb_to_hsv(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB->HSV for arr_rgb in [0,1], shape (H,W,3).
    Returns HSV in [0,1], same shape.
    """
    r = arr_rgb[..., 0]
    g = arr_rgb[..., 1]
    b = arr_rgb[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-12

    m = mask & (cmax == r)
    h[m] = ((g[m] - b[m]) / (delta[m] + 1e-12)) % 6.0
    m = mask & (cmax == g)
    h[m] = ((b[m] - r[m]) / (delta[m] + 1e-12)) + 2.0
    m = mask & (cmax == b)
    h[m] = ((r[m] - g[m]) / (delta[m] + 1e-12)) + 4.0
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(cmax, dtype=np.float32)
    nz = cmax > 1e-12
    s[nz] = (delta[nz] / (cmax[nz] + 1e-12)).astype(np.float32)

    v = cmax.astype(np.float32)

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _glcm_like_features(
    gray01: np.ndarray, levels: int = 8, dx: int = 1, dy: int = 0
) -> np.ndarray:
    """
    Very small GLCM-like statistics for a single offset (dx, dy).
    Quantize grayscale to `levels` and compute a normalized co-occurrence matrix.
    Returns [contrast, homogeneity, energy, entropy].
    """
    g = np.clip(gray01, 0.0, 1.0)
    q = np.floor(g * (levels - 1 + 1e-12)).astype(np.int32)  # 0..levels-1

    h, w = q.shape
    y0 = max(0, dy)
    y1 = h + min(0, dy)
    x0 = max(0, dx)
    x1 = w + min(0, dx)

    a = q[y0:y1, x0:x1]
    b = q[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

    if a.size == 0 or b.size == 0:
        return np.zeros(4, dtype=np.float32)

    idx = a * levels + b
    counts = np.bincount(idx.ravel(), minlength=levels * levels).astype(np.float32)
    P = counts.reshape(levels, levels)
    s = P.sum()
    if s <= 0:
        return np.zeros(4, dtype=np.float32)
    P = P / s

    i = np.arange(levels, dtype=np.float32).reshape(-1, 1)
    j = np.arange(levels, dtype=np.float32).reshape(1, -1)
    diff2 = (i - j) ** 2

    contrast = float((P * diff2).sum())
    homogeneity = float((P / (1.0 + diff2)).sum())
    energy = float((P * P).sum())
    entropy = float(-(P * np.log(P + 1e-12)).sum())
    return np.asarray([contrast, homogeneity, energy, entropy], dtype=np.float32)


def _lbp_hist_features(gray01: np.ndarray, bins: int = 16, step: int = 1) -> np.ndarray:
    """
    Simple 8-neighbor LBP histogram with optional subsampling `step` (>=1).
    Returns `bins` normalized counts.
    """
    g = np.clip(gray01, 0.0, 1.0).astype(np.float32)
    if step < 1:
        step = 1
    g = g[::step, ::step]
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.zeros(bins, dtype=np.float32)

    gc = g[1:-1, 1:-1]
    code = np.zeros_like(gc, dtype=np.uint8)

    code |= (g[:-2, :-2] >= gc).astype(np.uint8) << 7
    code |= (g[:-2, 1:-1] >= gc).astype(np.uint8) << 6
    code |= (g[:-2, 2:] >= gc).astype(np.uint8) << 5
    code |= (g[1:-1, 2:] >= gc).astype(np.uint8) << 4
    code |= (g[2:, 2:] >= gc).astype(np.uint8) << 3
    code |= (g[2:, 1:-1] >= gc).astype(np.uint8) << 2
    code |= (g[2:, :-2] >= gc).astype(np.uint8) << 1
    code |= (g[1:-1, :-2] >= gc).astype(np.uint8) << 0

    if bins <= 0:
        bins = 16
    group = max(1, 256 // bins)
    bucket = (code.astype(np.int32) // group).clip(0, bins - 1)
    hist = np.bincount(bucket.ravel(), minlength=bins).astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)
    return hist


GRAY_HIST_BINS = 16
RGB_HIST_BINS = 10  # 3*10 = 30
EDGE_HIST_BINS = 8

EXTRA_HSV_STATS = 6
EXTRA_VEG_STATS = 3

GLCM_LEVELS = 8
GLCM_FEATS_PER = 4
GLCM_DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 4 offsets
GLCM_FEATS = GLCM_FEATS_PER * len(GLCM_DIRS)  # 16

LBP_BINS = 16
LBP_SCALES = [1, 2]
LBP_FEATS = LBP_BINS * len(LBP_SCALES)  # 32

SPOT_FEATS = 8

FEAT_DIM = (
    2
    + GRAY_HIST_BINS
    + (4 * 4 * 2)
    + 6
    + (3 * RGB_HIST_BINS)
    + 2
    + EDGE_HIST_BINS
    + EXTRA_HSV_STATS
    + EXTRA_VEG_STATS
    + GLCM_FEATS
    + LBP_FEATS
    + SPOT_FEATS
)  # 157


def image_features_from_path(
    img_path,
    gray_hist_bins=GRAY_HIST_BINS,
    rgb_hist_bins=RGB_HIST_BINS,
    edge_hist_bins=EDGE_HIST_BINS,
    size=(128, 128),
):
    """
    Deterministic features for ROC AUC ranking.
    Total dims (fixed): FEAT_DIM (currently 157)
    """
    try:
        im_rgb = Image.open(img_path).convert("RGB").resize(size)
        arr_rgb = np.asarray(im_rgb, dtype=np.float32) / 255.0  # (H,W,3)
        arr_gray = (
            0.2989 * arr_rgb[..., 0]
            + 0.5870 * arr_rgb[..., 1]
            + 0.1140 * arr_rgb[..., 2]
        ).astype(np.float32)
    except Exception:
        return np.full(FEAT_DIM, np.nan, dtype=np.float32)

    mean_g = float(arr_gray.mean())
    std_g = float(arr_gray.std())

    hist_g, _ = np.histogram(
        arr_gray, bins=gray_hist_bins, range=(0.0, 1.0), density=False
    )
    hist_g = hist_g.astype(np.float32)
    hist_g = hist_g / (hist_g.sum() + 1e-12)

    h, w = arr_gray.shape
    gh, gw = h // 4, w // 4
    grid_feats = []
    for i in range(4):
        for j in range(4):
            patch = arr_gray[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
            grid_feats.append(float(patch.mean()))
            grid_feats.append(float(patch.std()))
    grid_feats = np.asarray(grid_feats, dtype=np.float32)

    ch_stats = []
    for k in range(3):
        ch = arr_rgb[..., k]
        ch_stats.append(float(ch.mean()))
        ch_stats.append(float(ch.std()))
    ch_stats = np.asarray(ch_stats, dtype=np.float32)

    ch_hists = []
    for k in range(3):
        ch = arr_rgb[..., k]
        hist_c, _ = np.histogram(
            ch, bins=rgb_hist_bins, range=(0.0, 1.0), density=False
        )
        hist_c = hist_c.astype(np.float32)
        hist_c = hist_c / (hist_c.sum() + 1e-12)
        ch_hists.append(hist_c)
    ch_hists = np.concatenate(ch_hists).astype(np.float32)  # 3*rgb_hist_bins

    mag = _sobel_grad_mag(arr_gray)
    mag_mean = float(mag.mean())
    mag_std = float(mag.std())
    lo = float(np.quantile(mag, 0.01))
    hi = float(np.quantile(mag, 0.99))
    if hi <= lo:
        lo, hi = 0.0, float(mag.max() + 1e-6)
    mag_hist, _ = np.histogram(mag, bins=edge_hist_bins, range=(lo, hi), density=False)
    mag_hist = mag_hist.astype(np.float32)
    mag_hist = mag_hist / (mag_hist.sum() + 1e-12)

    hsv = _rgb_to_hsv(arr_rgb)
    hsv_stats = np.asarray(
        [
            float(hsv[..., 0].mean()),
            float(hsv[..., 0].std()),
            float(hsv[..., 1].mean()),
            float(hsv[..., 1].std()),
            float(hsv[..., 2].mean()),
            float(hsv[..., 2].std()),
        ],
        dtype=np.float32,
    )

    r = arr_rgb[..., 0]
    g = arr_rgb[..., 1]
    b = arr_rgb[..., 2]
    exg = (2.0 * g - r - b).astype(np.float32)
    exr = (1.4 * r - g).astype(np.float32)
    exgr = (exg - exr).astype(np.float32)
    veg_stats = np.asarray(
        [float(exg.mean()), float(exr.mean()), float(exgr.mean())],
        dtype=np.float32,
    )

    glcm_all = []
    for dx, dy in GLCM_DIRS:
        glcm_all.append(_glcm_like_features(arr_gray, levels=GLCM_LEVELS, dx=dx, dy=dy))
    glcm_feats = np.concatenate(glcm_all).astype(np.float32)

    lbp_all = []
    for st in LBP_SCALES:
        lbp_all.append(_lbp_hist_features(arr_gray, bins=LBP_BINS, step=st))
    lbp_hist = np.concatenate(lbp_all).astype(np.float32)

    exg_n = (exg - exg.min()) / (exg.max() - exg.min() + 1e-12)
    exr_n = (exr - exr.min()) / (exr.max() - exr.min() + 1e-12)
    t1, t2, t3 = 0.65, 0.75, 0.85
    spot_exr1 = float((exr_n > t1).mean())
    spot_exr2 = float((exr_n > t2).mean())
    spot_exr3 = float((exr_n > t3).mean())
    spot_exg1 = float((exg_n < (1.0 - t1)).mean())
    spot_exg2 = float((exg_n < (1.0 - t2)).mean())
    spot_exg3 = float((exg_n < (1.0 - t3)).mean())
    mask = (exr_n > t2).astype(np.float32)
    frag = float((mag * mask).mean())
    mask_area = float(mask.mean())
    spot_feats = np.asarray(
        [
            spot_exr1,
            spot_exr2,
            spot_exr3,
            spot_exg1,
            spot_exg2,
            spot_exg3,
            frag,
            mask_area,
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            np.asarray([mean_g, std_g], dtype=np.float32),
            hist_g,
            grid_feats,
            ch_stats,
            ch_hists,
            np.asarray([mag_mean, mag_std], dtype=np.float32),
            mag_hist,
            hsv_stats,
            veg_stats,
            glcm_feats,
            lbp_hist,
            spot_feats,
        ]
    ).astype(np.float32)

    if feats.shape[0] != FEAT_DIM:
        fixed = np.full(FEAT_DIM, np.nan, dtype=np.float32)
        n = min(FEAT_DIM, feats.shape[0])
        fixed[:n] = feats[:n]
        feats = fixed

    return feats


def build_feature_matrix(image_ids):
    X = np.zeros((len(image_ids), FEAT_DIM), dtype=np.float32)
    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        X[i] = image_features_from_path(img_path)

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_means, inds[1])
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

print("Feature dim:", FEAT_DIM)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)



## === cell 4
pred_test = pd.DataFrame({"image_id": test_df["image_id"].values})

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for c in target_cols:
    y = train_df[c].astype(int).values

    if y.min() == y.max():
        const_p = float(y.mean())
        pred_test[c] = const_p
        continue

    candidates = [
        dict(C=1.0, solver="lbfgs", max_iter=4000),
        dict(C=2.0, solver="liblinear", max_iter=6000),
        dict(C=4.0, solver="liblinear", max_iter=8000),
        dict(C=8.0, solver="liblinear", max_iter=10000),
    ]

    best_params = None
    best_auc = -1.0

    for params in candidates:
        oof = np.zeros(len(y), dtype=np.float64)

        for tr_idx, va_idx in skf.split(X_train, y):
            clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            **params,
                            random_state=0,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            clf.fit(X_train[tr_idx], y[tr_idx])
            oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1].astype(np.float64)

        try:
            auc = roc_auc_score(y, oof)
        except Exception:
            auc = -1.0

        if auc > best_auc:
            best_auc = auc
            best_params = params

    if best_params is None:
        best_params = dict(C=1.0, solver="lbfgs", max_iter=4000)

    final_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    **best_params,
                    random_state=0,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    final_clf.fit(X_train, y)
    pred_test[c] = final_clf.predict_proba(X_test)[:, 1].astype(np.float64)

pred_test[target_cols] = pred_test[target_cols].clip(1e-6, 1 - 1e-6)



## === cell 5
sub_out = sub[["image_id"] + target_cols].copy()
sub_out = sub_out.merge(pred_test, on="image_id", how="left", suffixes=("", "_pred"))

priors = train_df[target_cols].mean().to_dict()

for c in target_cols:
    if f"{c}_pred" not in sub_out.columns:
        sub_out[c] = float(priors.get(c, 0.25))
    else:
        sub_out[c] = sub_out[f"{c}_pred"].astype(float)
        sub_out.drop(columns=[f"{c}_pred"], inplace=True)

sub_out[target_cols] = sub_out[target_cols].clip(0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## === cell 6
assert os.path.exists("submission.csv"), "submission.csv was not created."
loaded = pd.read_csv("submission.csv")
assert (
    list(loaded.columns) == ["image_id"] + target_cols
), "Submission columns mismatch."
assert len(loaded) == len(
    pd.read_csv(sample_path)
), "Submission row count mismatch vs sample_submission."
assert loaded["image_id"].equals(
    pd.read_csv(sample_path)["image_id"]
), "image_id order mismatch vs sample_submission."
assert loaded[target_cols].notnull().all().all(), "Found NaNs in prediction columns."
print("Submission format validated.")

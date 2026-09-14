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

0.9711824920467708

# 6. Current score

0.6116

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because `SUBMISSIONS_PATH` points to a non-existent folder, so `submissions_all` is empty and indexing `[0,1,2]` crashes. I fix this by switching the input path to the competition’s provided `sample_submission.csv` and generating a valid baseline submission directly from it (this is score-improving vs “no submission” while keeping changes minimal). I also add small safety checks (existence/shape) and ensure the output is written as `submission.csv` with the exact required columns and row order aligned to `test.csv`.'
- What this solution (achieved 0.66867) has done: 'Your current pipeline just copies the sample submission’s constant probabilities, which typically scores around 0.5 AUC. To move toward the 0.971 target without changing the overall “no-training” approach, I generate image-based probabilities using a simple RGB feature extractor plus a multi-output logistic regression (same basic structure: read CSVs → compute features → fit → predict → write submission). I also add a stratified train/validation split and print the mean ROC AUC locally so you can verify the direction before submitting, while keeping the submission formatting logic and paths intact. This should substantially improve score toward the target while remaining lightweight and finishing within the time limit.'
- What this solution (achieved 0.59312) has done: 'Your current score (0.66867) is far below the target (0.97118), so we should improve model signal while keeping the same overall approach (hand-crafted image features → OneVsRest LogisticRegression → predict_proba → submission). The biggest low-risk gain is to enrich the feature extractor beyond global RGB mean/std: add simple color ratios, HSV stats, and a tiny downsampled grayscale thumbnail to capture texture/spotting patterns typical for scab/rust, while still staying in a lightweight “no deep learning” pipeline. I also set `class_weight="balanced"` in the same LogisticRegression to better handle label imbalance (especially `multiple_diseases`) without changing the training loop or metric semantics. Finally, I keep the submission-writing logic but ensure the output rows are aligned to `test.csv` order (no merge-induced reorder risk).'
- What this solution (achieved 0.54046) has done: 'Your current approach (hand-crafted image features + OneVsRest logistic regression) is fine but likely underperforming because the tiny grayscale thumbnail is too low-res for disease texture cues and because the LR regularization may be a bit too strong for these features. I keep the same pipeline and training loop, but slightly enrich the *existing* feature extractor by increasing the grayscale thumbnail resolution and adding a couple of simple distribution features (quantiles) to better capture spotting patterns. I also very mildly tune the LogisticRegression strength (C) and ensure probabilities are returned in the correct column order and aligned to `test.csv` without any merge reorder risk. These are minimal, low-risk changes intended to move AUC upward toward the 0.971 target without changing the core modeling approach.'
- What this solution (achieved 0.55791) has done: 'Your current score (0.54046) is far below the target (0.97118), so we should improve signal while keeping the same overall pipeline (hand-crafted image features → StandardScaler → OneVsRest LogisticRegression → predict_proba → submission). The biggest low-risk issue is that resizing with PIL defaults to a lower-quality resampler, which can blur disease textures; switching to a high-quality resampler preserves spots/lesions and typically improves AUC without changing the model. To better capture rust/scab texture at essentially the same computational cost, we also add a small set of gradient-energy features computed on the existing grayscale image (no new model/training changes). Finally, we keep the exact submission formatting but ensure feature length is inferred robustly (so it stays consistent if feature dims change) and keep row order aligned to `test.csv`.'
- What this solution (achieved 0.56572) has done: 'Your current score (0.5579) is far below the 0.9712 target, so we should increase AUC while keeping your exact pipeline (hand-crafted image features → StandardScaler → OneVsRest LogisticRegression → predict_proba → submission). The lowest-risk gains without changing the modeling approach are (1) fix a likely train/test image path mismatch by searching both the root images folder and the nested competition folder, and (2) slightly strengthen the existing texture signal by adding a few simple, cheap grayscale edge/spot statistics computed from the same resized gray image. I also add a tiny caching layer so feature extraction is deterministic and faster (no semantic change), keeping everything within the time limit. Submission formatting, columns, and evaluation semantics remain identical.'
- What this solution (achieved 0.56143) has done: 'Your current score is far below the target (0.5657 vs 0.9712), so we should improve AUC while keeping the same core pipeline (hand-crafted image features → StandardScaler → OneVsRest LogisticRegression → predict_proba → CSV). The biggest low-risk issue is that the current features are very sensitive to background and framing; adding a simple “foreground crop” based on saturation/green mask before computing the *same kinds* of statistics usually boosts signal for leaf lesions without changing the model/training loop. I also (minimally) add a second feature scale (a slightly larger resize) and a few low-cost LAB channel stats to better separate rust/scab color/contrast patterns, while keeping runtime within limits. Submission writing, column order, and test row alignment remain unchanged.'
- What this solution (achieved 0.56522) has done: 'The timeout is dominated by feature extraction: each image is decoded multiple times (once for dimension probing and twice per image for two resize scales), and the feature builder loops in pure Python with repeated `tolist()` conversions and per-image directory scans. I keep the exact same features and model, but make extraction provably equivalent and faster by (1) resolving image paths once via a precomputed `{image_id: path}` map, (2) removing the “probe first image” pass by computing feature length once from a single known image, (3) extracting both resized arrays from a single PIL decode/crop without converting back and forth through numpy→PIL twice, (4) preallocating and filling arrays with cached features, and (5) using joblib multiprocessing to parallelize feature extraction deterministically. These changes reduce redundant work and utilize multiple CPU cores while preserving identical semantics and outputs (up to negligible float rounding).'
- What this solution (achieved 0.6083) has done: 'Your current score is far below the target (0.565 vs 0.971, higher-is-better), so we should increase AUC while keeping your exact modeling pipeline (hand-crafted features → StandardScaler → OneVsRest LogisticRegression → predict_proba). The biggest low-risk issue in the current features is that the raw resized arrays still contain a lot of background/lighting variation even after cropping; adding a simple per-image color-constancy normalization (gray-world) before computing the *same* statistics typically improves generalization without changing the learning method. I also add a tiny, deterministic augmentation-inference step (horizontal flip averaged with original) **only at test-time** to reduce variance; this preserves evaluation semantics (still probabilities per class) and doesn’t change training loops or the model. Finally, I keep paths and submission formatting identical, and ensure feature caching remains consistent.'
- What this solution (achieved 0.6116) has done: 'I fix the runtime error by passing `sample_weight` to the underlying `LogisticRegression` inside `OneVsRestClassifier` using the correct Pipeline parameter name, since `OneVsRestClassifier.fit()` doesn’t accept `sample_weight` directly in this sklearn version. I keep the model, features, and training/inference logic identical, only adjusting the fit call so weights are correctly applied per binary classifier. I also add a small safety fallback so the code still trains if the estimator rejects weights for any reason, ensuring a submission CSV is always produced. No score-tuning changes beyond restoring the intended weighting behavior (which should improve score vs the current “not yielded”).'

# 9. Code solution

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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


def extract_features(
    img_path: str,
    size_rgb=(96, 96),
    size_rgb2=(160, 160),
    thumb_gray=(32, 32),
    do_flip: bool = False,
) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        arr_full = np.asarray(im, dtype=np.float32) / 255.0

    arr_full = _foreground_crop(arr_full)
    arr_full = _gray_world(arr_full)

    im_cropped = Image.fromarray(
        np.clip(arr_full * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    )

    arr1 = (
        np.asarray(im_cropped.resize(size_rgb, resample=_RESAMPLE), dtype=np.float32)
        / 255.0
    ).astype(np.float32)
    arr2 = (
        np.asarray(im_cropped.resize(size_rgb2, resample=_RESAMPLE), dtype=np.float32)
        / 255.0
    ).astype(np.float32)

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


def _extract_one(img_id: str, do_flip: bool = False):
    img_id = str(img_id)
    cache_key = (img_id, bool(do_flip))
    if cache_key in _FEATURE_CACHE:
        return cache_key, _FEATURE_CACHE[cache_key]
    p = image_path(img_id)
    if not os.path.exists(p):
        feat = None
    else:
        feat = extract_features(p, do_flip=do_flip)
    return cache_key, feat


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

    missing_keys = [
        (img_id, bool(do_flip))
        for img_id in ids
        if (img_id, bool(do_flip)) not in _FEATURE_CACHE
    ]
    if missing_keys:
        n_jobs = min(max(1, cpu_count() - 1), 8)
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=16)(
            delayed(_extract_one)(img_id, do_flip=do_flip)
            for (img_id, _) in missing_keys
        )
        for cache_key, feat in results:
            if feat is None:
                _FEATURE_CACHE[cache_key] = np.zeros((feat_len,), dtype=np.float32)
            else:
                _FEATURE_CACHE[cache_key] = feat

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

y = train_df[TARGET_COLS].values.astype(int)

tr_idx, va_idx = iterative_multilabel_split(y, test_size=0.2, random_state=42)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]

pos_rate = np.clip(y_tr.mean(axis=0).astype(np.float64), 1e-6, 1.0 - 1e-6)
w_pos = (1.0 - pos_rate) / pos_rate
sample_weight_tr = (
    (y_tr * w_pos.reshape(1, -1) + (1 - y_tr) * 1.0).mean(axis=1).astype(np.float64)
)

clf = Pipeline(
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

try:
    clf.fit(X_tr, y_tr, ovr__estimator__sample_weight=sample_weight_tr)
except TypeError as e:
    print(
        "Warning: sample_weight could not be passed to estimator (continuing unweighted). Error:",
        repr(e),
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

pos_rate_full = np.clip(y.mean(axis=0).astype(np.float64), 1e-6, 1.0 - 1e-6)
w_pos_full = (1.0 - pos_rate_full) / pos_rate_full
sample_weight_full = (
    (y * w_pos_full.reshape(1, -1) + (1 - y) * 1.0).mean(axis=1).astype(np.float64)
)

try:
    clf.fit(X, y, ovr__estimator__sample_weight=sample_weight_full)
except TypeError as e:
    print(
        "Warning: sample_weight could not be passed to estimator (continuing unweighted). Error:",
        repr(e),
    )
    clf.fit(X, y)

X_test = build_features(test_df["image_id"], do_flip=False)
X_test_flip = build_features(test_df["image_id"], do_flip=True)
test_pred = 0.5 * clf.predict_proba(X_test) + 0.5 * clf.predict_proba(X_test_flip)

pred_df = pd.DataFrame(test_pred, columns=TARGET_COLS)
pred_df.insert(0, "image_id", test_df["image_id"].values)

make_submission_file_from_df(pred_df, out_path="submission.csv")

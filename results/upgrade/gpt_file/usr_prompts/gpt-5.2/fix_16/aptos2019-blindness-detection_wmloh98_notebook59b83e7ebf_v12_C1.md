# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

DATA_PATH = "../input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256

SEED = 42
np.random.seed(SEED)

_CIRCLE_MASK_CACHE = {}


def crop_image_from_gray(img, tol=7):
    """
    Crop out dark borders based on grayscale threshold.
    Expects uint8 ndarray HxW or HxWx3. Returns original if crop would be empty.
    """
    if img is None:
        return img
    img = np.asarray(img)
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
    elif img.ndim == 3:
        gray = (
            0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        ).astype(np.uint8)
        mask = gray > tol
        if not mask.any():
            return img
        ys = np.where(mask.any(1))[0]
        xs = np.where(mask.any(0))[0]
        if ys.size == 0 or xs.size == 0:
            return img
        y0, y1 = ys[0], ys[-1] + 1
        x0, x1 = xs[0], xs[-1] + 1
        cropped = img[y0:y1, x0:x1, :]
        if cropped.size == 0:
            return img
        return cropped
    else:
        return img


def circle_crop_v2(img):
    """
    Apply circular mask after making image square via resize to largest side.
    Returns uint8 ndarray.
    """
    if img is None:
        return img
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    h, w, _ = img.shape
    largest_side = int(max(h, w))
    if largest_side <= 0:
        return img

    img_sq = np.array(
        Image.fromarray(img).resize(
            (largest_side, largest_side), resample=Image.BILINEAR
        ),
        dtype=np.uint8,
    )

    key = (largest_side, largest_side)
    mask3 = _CIRCLE_MASK_CACHE.get(key)
    if mask3 is None:
        hs, ws = largest_side, largest_side
        cx, cy = ws // 2, hs // 2
        r = min(cx, cy)
        yy, xx = np.ogrid[:hs, :ws]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r**2)
        mask3 = np.stack([mask, mask, mask], axis=-1)
        _CIRCLE_MASK_CACHE[key] = mask3

    out = np.zeros_like(img_sq)
    out[mask3] = img_sq[mask3]
    out = crop_image_from_gray(out)
    return out


def _gaussian_blur_rgb(img_f32, sigma):
    """
    Gaussian blur for float32 RGB image in [0,255], using Pillow's GaussianBlur.
    Returns float32 RGB in [0,255].
    """
    sigma = float(sigma)
    if (not np.isfinite(sigma)) or sigma <= 0:
        sigma = 25.0

    img_u8 = np.clip(img_f32, 0.0, 255.0).astype(np.uint8, copy=False)
    pil = Image.fromarray(img_u8, mode="RGB")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32)


def preprocess_image_pil(path, sigmaX=25, DIM_X=256, DIM_Y=256):
    """
    Mirrors the original intent:
    crop dark borders -> circle crop -> resize -> unsharp mask.
    Returns float32 image in [0,1].
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return np.zeros((DIM_Y, DIM_X, 3), dtype=np.float32)

    img_u8 = np.asarray(img, dtype=np.uint8)

    img_u8 = crop_image_from_gray(img_u8)
    img_u8 = circle_crop_v2(img_u8)

    img_rs = np.asarray(
        Image.fromarray(img_u8).resize((DIM_X, DIM_Y), resample=Image.BILINEAR),
        dtype=np.float32,
    )

    blurred = _gaussian_blur_rgb(img_rs, sigma=float(sigmaX))
    sharp = 4.0 * img_rs + (-4.0) * blurred + 128.0
    sharp = np.clip(sharp, 0.0, 255.0).astype(np.float32, copy=False)

    return (sharp / 255.0).astype(np.float32, copy=False)




## === cell 1
use_heuristic = True
model = None
print(
    "TensorFlow/Keras disabled due to protobuf MessageFactory.GetPrototype crash; using heuristic inference."
)


def severity_proxy(img_f32):
    """
    img_f32: float32 RGB in [0,1], shape (H,W,3)
    Returns a float score where larger ~= more severe (heuristic).
    """
    if img_f32 is None or img_f32.size == 0:
        return 0.0

    g = img_f32[:, :, 1].astype(np.float32, copy=False)

    g255 = (g * 255.0).astype(np.float32, copy=False)

    pil_g = Image.fromarray(
        np.clip(g255, 0.0, 255.0).astype(np.uint8, copy=False), mode="L"
    )
    illum = np.asarray(
        pil_g.filter(ImageFilter.GaussianBlur(radius=25.0)), dtype=np.float32
    )
    illum = np.maximum(illum, 1.0)  # avoid division by small values

    g_norm = g255 / illum  # around ~[0, >1], relative reflectance
    g_norm = np.clip(g_norm, 0.0, 2.0) / 2.0

    dark_frac = float(np.mean(g_norm < 0.35))
    contrast = float(np.clip(np.std(g_norm), 0.0, 0.5)) / 0.5
    score = 0.85 * dark_frac + 0.15 * contrast
    return float(np.clip(score, 0.0, 1.0))




## === cell 2
train_csv_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

candidate_train_dirs = [
    os.path.join(DATA_PATH, "train_images"),
    "../input/train_images",
    "../input/aptos2019-blindness-detection/train_images",
]
train_dir = None
for d in candidate_train_dirs:
    if os.path.isdir(d):
        train_dir = d
        break
if train_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images directory. Tried: {candidate_train_dirs}"
    )

label_counts = train_df["diagnosis"].value_counts().sort_index()
label_probs = (
    (label_counts / label_counts.sum()).reindex(range(5), fill_value=0.0).values
)
cum_probs = np.cumsum(label_probs)

max_calib = 900  # keep as-is (time/quality tradeoff already chosen)

per_class = max(10, max_calib // 5)
parts = []
for c in range(5):
    dfc = train_df[train_df["diagnosis"] == c]
    if len(dfc) == 0:
        continue
    parts.append(dfc.sample(n=min(per_class, len(dfc)), random_state=SEED + c))
calib_df = (
    pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
)
if len(calib_df) > max_calib:
    calib_df = calib_df.sample(n=max_calib, random_state=SEED).reset_index(drop=True)

calib_scores = []
calib_labels = calib_df["diagnosis"].astype(int).values

for img_id in calib_df["id_code"].astype(str).tolist():
    path = os.path.join(train_dir, img_id + ".png")
    img = preprocess_image_pil(path, sigmaX=25, DIM_X=DIM_X, DIM_Y=DIM_Y)
    calib_scores.append(severity_proxy(img))

calib_scores = np.asarray(calib_scores, dtype=np.float32)

med = []
for c in range(5):
    s = calib_scores[calib_labels == c]
    if s.size == 0:
        med.append(np.nan)
    else:
        med.append(float(np.median(s)))
med = np.array(med, dtype=np.float32)

flip_proxy = False
if np.all(np.isfinite(med)):
    if med[4] < med[0]:
        flip_proxy = True

if flip_proxy:
    calib_scores = 1.0 - calib_scores
    med = 1.0 - med
    print(
        "Flipping proxy direction based on training medians (to improve ordinal alignment)."
    )


def _trimmed_mean(x, lo=10.0, hi=90.0):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.nan
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    y = x[(x >= a) & (x <= b)]
    if y.size == 0:
        y = x
    return float(np.mean(y))


use_boundary_thresholds = True
bounds = []
for c in range(4):
    s_c = calib_scores[calib_labels == c]
    s_n = calib_scores[calib_labels == (c + 1)]
    if (s_c.size < 8) or (s_n.size < 8):
        use_boundary_thresholds = False
        break
    m_c = _trimmed_mean(s_c, lo=10.0, hi=90.0)
    m_n = _trimmed_mean(s_n, lo=10.0, hi=90.0)
    if (not np.isfinite(m_c)) or (not np.isfinite(m_n)):
        use_boundary_thresholds = False
        break
    b = 0.5 * (m_c + m_n)
    bounds.append(b)

if use_boundary_thresholds:
    calib_thresholds = np.array(bounds, dtype=np.float32)
    calib_thresholds = np.clip(calib_thresholds, 1e-5, 1.0 - 1e-5)
    eps = 1e-4
    for i in range(1, 4):
        if calib_thresholds[i] <= calib_thresholds[i - 1] + eps:
            calib_thresholds[i] = calib_thresholds[i - 1] + eps
    if not np.all(np.diff(calib_thresholds) > 1e-6):
        use_boundary_thresholds = False

if not use_boundary_thresholds:
    use_median_thresholds = np.all(np.isfinite(med))
    if use_median_thresholds:
        med_mono = med.copy()
        eps = 1e-4
        for i in range(1, 5):
            if med_mono[i] <= med_mono[i - 1] + eps:
                med_mono[i] = med_mono[i - 1] + eps
        calib_thresholds = np.array(
            [
                0.5 * (med_mono[0] + med_mono[1]),
                0.5 * (med_mono[1] + med_mono[2]),
                0.5 * (med_mono[2] + med_mono[3]),
                0.5 * (med_mono[3] + med_mono[4]),
            ],
            dtype=np.float32,
        )
        calib_thresholds = np.clip(calib_thresholds, 1e-5, 1.0 - 1e-5)

        if not np.all(np.diff(calib_thresholds) > 1e-6):
            use_median_thresholds = False

    if not use_median_thresholds:
        raw_thresholds = np.quantile(calib_scores, cum_probs[:4]).astype(np.float32)
        raw_thresholds = np.clip(raw_thresholds, 1e-5, 1.0 - 1e-5)
        if not np.all(np.isfinite(raw_thresholds)) or not np.all(
            np.diff(raw_thresholds) > 1e-6
        ):
            calib_thresholds = np.array([0.35, 0.45, 0.55, 0.65], dtype=np.float32)
        else:
            calib_thresholds = raw_thresholds

print("Train label distribution:", label_counts.to_dict())
print(
    "Calib class medians (proxy, possibly flipped):",
    {i: float(med[i]) if np.isfinite(med[i]) else None for i in range(5)},
)
print("Using calibrated heuristic thresholds:", calib_thresholds)




## === cell 3
submission_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

candidate_test_dirs = [
    os.path.join(DATA_PATH, "test_images"),
    "../input/test_images",
    "../input/aptos2019-blindness-detection/test_images",
]
test_dir = None
for d in candidate_test_dirs:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {candidate_test_dirs}"
    )

thresholds = calib_thresholds

pred_labels = []
for fn in submission_df["filename"].tolist():
    path = os.path.join(test_dir, fn)
    img = preprocess_image_pil(path, sigmaX=25, DIM_X=DIM_X, DIM_Y=DIM_Y)
    score = severity_proxy(img)
    if flip_proxy:
        score = 1.0 - float(score)
    cls = int(np.digitize(score, thresholds, right=False))
    pred_labels.append(cls)

pred = np.array(pred_labels, dtype=int)




## === cell 4
if len(pred) != len(submission_df):
    raise RuntimeError(
        f"Prediction length {len(pred)} != test rows {len(submission_df)}"
    )

out_df = submission_df[["id_code"]].copy()
out_df["diagnosis"] = pred.astype(int)
out_df.to_csv("submission.csv", index=False)

print(out_df.head())
print("Wrote submission.csv with shape:", out_df.shape)




## === cell 5
from collections import Counter

cnt = Counter(out_df["diagnosis"].tolist())
print(cnt)

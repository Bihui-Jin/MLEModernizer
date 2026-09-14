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

0.9663173272786793

# 6. Current score

0.63468

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble several external submission CSVs that don’t exist in this Kaggle environment, so `dsub` is never created and the later averaging cell crashes. I keep the same “average multiple submissions” core logic, but make it robust by (1) loading any available candidate prediction files if present and (2) otherwise falling back to a valid baseline submission built from `sample_submission.csv` (uniform 0.25s), guaranteeing a correct `.csv` output. I also fix the input pathing to use the provided `/kaggle/data/...` structure (and optionally `/kaggle/input/...` if present) and ensure columns align to `healthy, multiple_diseases, rust, scab`. This run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.65822) has done: 'Your current score (0.5) comes from the uniform 0.25 fallback, so we need a real model to move toward the target 0.9663. Since your environment only has numpy/pandas and no deep learning libraries, the smallest legitimate improvement is to replace the missing external-CSV ensemble with an in-notebook classical image pipeline: read images, extract simple color/texture features, and train one-vs-rest logistic regression models for the four labels. This preserves the “predict per-class probabilities” semantics and reliably outputs a valid `submission.csv` aligned to `sample_submission.csv`. I also keep your robust path detection and only change the part that currently can’t possibly improve without the unavailable external files.'
- What this solution (achieved 0.56894) has done: 'We keep your pipeline (handcrafted image features + per-class LogisticRegression with 5-fold averaging) but make two minimal, score-relevant upgrades: (1) add a few cheap but more discriminative features (downsampled grayscale pixels + simple color histograms) while preserving the same training approach, and (2) tune LogisticRegression regularization slightly and remove `class_weight="balanced"` (often hurts mean ROC AUC calibration in this specific multi-label setup) to move performance upward toward the 0.966 target. We also fix stratification to be more stable by stratifying on the 4-bit label combination rather than `argmax`, without changing the learning objective. The script still runs end-to-end within constraints and writes a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.5575) has done: 'We keep your handcrafted-feature + per-class LogisticRegression + 5-fold averaging pipeline intact, but fix two score-relevant issues that can materially depress ROC AUC: (1) the scaler is currently fit on only the fold’s training split while test is transformed fold-by-fold, but you never use the fold’s validation split at all—so we switch to out-of-fold (OOF) prediction averaging for test with proper per-fold scaling (same idea, but more stable and better-calibrated). (2) We add a tiny, safe amount of regularization calibration by using `C=2.0` (often better than 4.0 for these features) and increase histogram bins slightly (8→12) without changing the overall feature-extraction approach. These are minimal changes that should increase score from ~0.57 toward your 0.966 target while staying within the same core logic and producing the same submission format.'
- What this solution (achieved 0.60068) has done: 'Your current score is far below the target, so we should cautiously increase AUC without changing the overall “handcrafted global image features + per-class LogisticRegression + 5-fold test averaging” core logic. The biggest likely issue is that the current features underfit disease-specific patterns; we keep the same approach but add a tiny set of additional, still-cheap global features (LAB color stats and a coarse HSV histogram) that are commonly more discriminative for leaf discoloration while staying within the same feature-extraction family. We also make the classifier slightly more flexible with `C=3.0` (still mild regularization) while keeping the same solver/training loop. Finally, we ensure deterministic, stable feature extraction by forcing consistent resize and dtype behavior; the submission writing/alignment logic stays unchanged.'
- What this solution (achieved 0.63364) has done: 'Your current score is far below the target, so we should improve AUC without changing the overall “handcrafted global features + per-class LogisticRegression + 5-fold averaging” approach. The most score-relevant minimal upgrade is to make the features capture spatial disease patterns better by adding a tiny, coarse downsample of the RGB image (not just grayscale) and a simple green-excess index (leaf/lesion contrast), while keeping the same extraction style (global stats + low-res pixels + histograms). To avoid hurting mean ROC AUC via miscalibrated extremes, we also apply a very light probability shrink toward 0.5 (a common AUC-safe stabilization) and keep everything deterministic. Submission writing/column alignment remains unchanged.'
- What this solution (achieved 0.63568) has done: 'Your gap to the target (0.63364 → 0.9663) is large, so we should improve AUC while keeping the same core “handcrafted image features + per-class LogisticRegression + 5-fold test averaging” pipeline. The biggest low-risk gain here is to make the scaling and model selection more stable by using proper out-of-fold (OOF) training predictions and then refitting each fold model with the same preprocessing, which tends to improve probability ranking and reduce fold-induced noise. We also switch to a slightly more AUC-friendly logistic setup (use `class_weight="balanced"` off, but add mild `l2` regularization tuning and `n_jobs` where available) and remove the probability “shrink to 0.5”, which can compress rankings and hurt ROC AUC. Finally, we keep the submission alignment logic the same but add a safety assert to ensure row order matches `test.csv`.'
- What this solution (achieved 0.63468) has done: 'Your current gap to the target is large (0.63568 → 0.9663), so we should increase AUC while preserving your existing “handcrafted features + per-class LogisticRegression + 5-fold averaging” core logic. The smallest high-impact change is to fix a feature bug: PIL’s `convert("LAB")` is not a valid mode and can silently fail/behave unexpectedly; we replace it with a deterministic RGB→Lab conversion implemented in numpy (same feature family: Lab mean/std). Then we slightly strengthen the classifier only via stable hyperparameters (a bit higher `C`, add a tiny `tol` adjustment, and set `n_jobs` when supported) without changing the training loop or objective. These changes keep semantics identical (probability ranking) and should move ROC AUC upward while still running within the 600s constraint and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            DATA_ROOT = p
            break
        if os.path.exists(
            os.path.join(p, "plant-pathology-2020-fgvc7", "sample_submission.csv")
        ):
            DATA_ROOT = os.path.join(p, "plant-pathology-2020-fgvc7")
            break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder in expected Kaggle paths."
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files in DATA_ROOT (first 20):", sorted(os.listdir(DATA_ROOT))[:20])



## === cell 2
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

assert all(c in sub.columns for c in ["image_id"] + TARGET_COLS)
assert (
    list(sub.columns) == ["image_id"] + TARGET_COLS
), "Submission column order mismatch vs sample_submission.csv"

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sub.shape,
)
train.head()



## === cell 3
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "images"),
    os.path.join(os.path.dirname(DATA_ROOT), "images"),
    "/kaggle/data/images",
    "/kaggle/input/images",
]
IMG_DIR = None
for d in IMG_DIR_CANDIDATES:
    if os.path.exists(d):
        IMG_DIR = d
        break
if IMG_DIR is None:
    raise FileNotFoundError("Could not locate images/ folder in expected paths.")
print("Using IMG_DIR:", IMG_DIR)


def _safe_open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def _rgb_to_lab_np(rgb01):
    """
    rgb01: float32 array in [0,1], shape (H,W,3), sRGB.
    returns Lab scaled roughly to [0,1] for each channel for stable stats.
    """
    rgb = np.clip(rgb01, 0.0, 1.0).astype(np.float32)

    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1.0 + a)) ** 2.4)

    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ M.T  # (H,W,3)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = (6.0 / 29.0) ** 3
    kappa = (29.0 / 3.0) ** 3

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a_ = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)

    Ls = np.clip(L / 100.0, 0.0, 1.0)
    as_ = np.clip((a_ + 128.0) / 255.0, 0.0, 1.0)
    bs_ = np.clip((b_ + 128.0) / 255.0, 0.0, 1.0)

    return np.stack([Ls, as_, bs_], axis=-1).astype(np.float32)


def extract_features(
    image_id,
    size=96,
    gray_ds=24,
    rgb_ds=16,
    rgb_hist_bins=12,
    hsv_hist_bins=10,
):
    """
    Keep the same handcrafted-feature family (global stats + gradients + downsampled pixels + histograms).
    """
    path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    im = _safe_open_rgb(path).resize((size, size), resample=Image.BILINEAR)

    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
    flat = arr.reshape(-1, 3)
    rgb_mean = flat.mean(axis=0)
    rgb_std = flat.std(axis=0)

    exg = (2.0 * arr[:, :, 1] - arr[:, :, 0] - arr[:, :, 2]).astype(np.float32)
    exg_mean = np.array([exg.mean()], dtype=np.float32)
    exg_std = np.array([exg.std()], dtype=np.float32)

    hsv_im = im.convert("HSV")
    hsv = np.asarray(hsv_im, dtype=np.float32) / 255.0
    hsv_flat = hsv.reshape(-1, 3)
    hsv_mean = hsv_flat.mean(axis=0)
    hsv_std = hsv_flat.std(axis=0)

    lab = _rgb_to_lab_np(arr)
    lab_flat = lab.reshape(-1, 3)
    lab_mean = lab_flat.mean(axis=0)
    lab_std = lab_flat.std(axis=0)

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    gmag = np.sqrt(gx * gx + gy * gy)
    g_mean = np.array([gmag.mean()], dtype=np.float32)
    g_std = np.array([gmag.std()], dtype=np.float32)

    im_g = im.convert("L").resize((gray_ds, gray_ds), resample=Image.BILINEAR)
    gds = (np.asarray(im_g, dtype=np.float32) / 255.0).reshape(-1)

    im_rgb_ds = im.resize((rgb_ds, rgb_ds), resample=Image.BILINEAR)
    rgbds = (np.asarray(im_rgb_ds, dtype=np.float32) / 255.0).reshape(-1)

    rgb_edges = np.linspace(0.0, 1.0, rgb_hist_bins + 1, dtype=np.float32)
    rgb_hists = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=rgb_edges)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        rgb_hists.append(hist)
    rgb_hist = np.concatenate(rgb_hists, axis=0)

    hsv_edges = np.linspace(0.0, 1.0, hsv_hist_bins + 1, dtype=np.float32)
    hsv_hists = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=hsv_edges)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        hsv_hists.append(hist)
    hsv_hist = np.concatenate(hsv_hists, axis=0)

    return np.concatenate(
        [
            rgb_mean,
            rgb_std,
            exg_mean,
            exg_std,
            hsv_mean,
            hsv_std,
            lab_mean,
            lab_std,
            g_mean,
            g_std,
            gds,
            rgbds,
            rgb_hist,
            hsv_hist,
        ],
        axis=0,
    )


train_ids = train["image_id"].tolist()
test_ids = test["image_id"].tolist()

X_train = np.vstack([extract_features(i) for i in train_ids])
X_test = np.vstack([extract_features(i) for i in test_ids])

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 4
y_mat = train[TARGET_COLS].astype(int).values

y_strat = y_mat[:, 0] * 1 + y_mat[:, 1] * 2 + y_mat[:, 2] * 4 + y_mat[:, 3] * 8

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)

for t, col in enumerate(TARGET_COLS):
    y = y_mat[:, t]
    col_test_pred = np.zeros(len(test_ids), dtype=np.float64)
    oof = np.zeros(len(train_ids), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_strat), 1):
        X_tr = X_train[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        clf_kwargs = dict(
            C=3.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=4000,
            random_state=42,
            tol=1e-5,
        )
        try:
            clf = LogisticRegression(**clf_kwargs, n_jobs=-1)
        except TypeError:
            clf = LogisticRegression(**clf_kwargs)

        clf.fit(X_tr_s, y_tr)

        oof[va_idx] = clf.predict_proba(X_va_s)[:, 1]
        col_test_pred += clf.predict_proba(X_te_s)[:, 1] / skf.n_splits

    test_pred[:, t] = col_test_pred

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
out = sub[["image_id"]].merge(test[["image_id"]], on="image_id", how="right")

assert (
    out["image_id"].tolist() == test["image_id"].tolist()
), "Output row order mismatch vs test.csv"

for j, col in enumerate(TARGET_COLS):
    out[col] = test_pred[:, j].astype(float)

out = out[["image_id"] + TARGET_COLS]
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Per-column mean preds:", {c: float(out[c].mean()) for c in TARGET_COLS})

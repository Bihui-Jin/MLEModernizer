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

0.9699448210609528

# 6. Current score

0.68377

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because the notebook expects pre-existing submission files in `/kaggle/input/submissions/submissions/`, but that directory doesn’t exist in your environment, so `submissions_all` is empty and indexing fails. I keep the ensemble logic intact, but add a minimal fallback that creates a valid baseline submission from `sample_submission.csv` when no external submissions are found. I also add small safety checks (weights length, indices range, and probability clipping) to prevent runtime errors and ensure the written `submission.csv` is always valid and correctly formatted.'
- What this solution (achieved 0.66583) has done: 'Your current 0.5 score comes from falling back to the sample submission (uniform 0.25), which yields near-random AUC; to move toward the 0.9699 target we need real image-based probabilities. Keeping changes minimal, I add a lightweight, deterministic sklearn baseline that extracts simple color/statistical features from each JPG and trains a one-vs-rest Logistic Regression model on the provided labels, then predicts probabilities for the test set. This preserves the overall “produce a submission.csv with the required columns” behavior and only replaces the fallback path when no external submission files exist. If external submissions are present, your original ensemble path is kept intact.'
- What this solution (achieved 0.62142) has done: 'Your current baseline is limited by weak hand-crafted features and by fitting four independent binary models without using any label correlations; we can move the ROC AUC up toward the 0.97 target with a minimal, still-sklearn approach. I keep the same image loading/resizing and feature extraction core, but add a small set of additional color/texture summary features and switch to a calibrated Linear SVM in a One-vs-Rest wrapper, which is typically stronger than plain LogisticRegression on this kind of feature space. I also ensure deterministic behavior (fixed seeds) and keep the submission formatting/alignment exactly as required. These are targeted upgrades that should improve generalization without changing the overall pipeline structure.'
- What this solution (achieved 0.65135) has done: 'Your current score (0.62142) is far below the target (0.96994), so we need a meaningful but still “same approach” improvement: keep the exact image loading/resizing + handcrafted feature extraction idea, but strengthen the classifier while staying in sklearn. The smallest high-impact change is to replace the calibrated LinearSVC with a stronger calibrated linear Logistic Regression (saga) in One-vs-Rest, and to slightly widen the feature vector with a few extra robust summary stats (HSV means/stds + simple color indices) without changing the overall pipeline structure. I also keep the external-submission ensemble path unchanged, and ensure the submission rows align exactly to `test.csv` order and columns match `sample_submission.csv`.'
- What this solution (achieved 0.65751) has done: 'Your current score (0.65135) is far below the target (0.96994), so we should improve the same sklearn, handcrafted-feature baseline without changing the overall pipeline structure. The most reliable minimal gain is to (1) extract a bit more disease-relevant information by adding simple per-channel quantiles and a compact HSV histogram (still just summary features), and (2) slightly strengthen regularization/optimization for the same OneVsRest LogisticRegression (same model family, same training flow). I also add a deterministic cache for feature extraction within the run to avoid any accidental recomputation variability and keep submission alignment strictly following `test.csv`. These changes keep the “no deep learning, no new training loops” core logic intact but should move ROC AUC upward toward your target.'
- What this solution (achieved 0.67076) has done: 'Your current score (0.6575) is far below the target (0.9699), so we should improve the same sklearn handcrafted-feature approach without changing the overall training/prediction flow. The smallest high-impact adjustment is to use a stronger non-linear classifier that still outputs calibrated probabilities: a One-vs-Rest RBF-kernel SVC with probability=True, keeping the exact same extracted features and the same end-to-end pipeline. To avoid breaking anything and keep behavior stable, the external-submission ensemble path is left untouched, and we only swap the fallback model used when no external submissions exist. I also keep strict alignment to `test.csv` ordering/`sample_submission.csv` columns and clip probabilities for a valid submission.'
- What this solution (achieved 0.68674) has done: 'Your current score (0.67076) is far below the target (0.96994), so we should improve the same sklearn handcrafted-feature pipeline without changing its overall structure. The most direct gain with minimal semantic changes is to make the RBF SVC probabilities better calibrated by wrapping the same SVC in `CalibratedClassifierCV`, which often improves ROC AUC when using `probability=True` outputs. To keep the core logic intact, we keep the same feature extraction, keep One-vs-Rest multi-label training, and keep the same submission formatting/alignment, only swapping how probabilities are produced. We also set `n_jobs=-1` for calibration to stay within the time budget and keep determinism via `random_state`.'
- What this solution (achieved 0.66514) has done: 'Your current score (0.68674) is far below the target (0.96994), so we should improve the same handcrafted-feature + sklearn pipeline with the smallest changes likely to increase mean ROC AUC. Keeping feature extraction and the One-vs-Rest + calibrated SVC approach intact, I only (1) switch the calibration method to isotonic (often better AUC calibration than sigmoid when you have enough data), (2) make the calibration CV stratified and deterministic per label to reduce fold pathologies, and (3) ensure perfect row/column alignment to `test.csv`/`sample_submission.csv` regardless of any external submissions. These are targeted changes to improve probability quality without changing the overall modeling approach or introducing early stopping/sampling. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.67337) has done: 'Your current gap to the target is large (0.665 → 0.970), so we need a real AUC lift while keeping the same handcrafted-feature + sklearn calibrated SVC core pipeline. The smallest high-impact change is to improve the calibration strategy: isotonic can overfit per-label with limited positives, so we switch to deterministic sigmoid calibration and increase CV folds slightly to stabilize probability ranking (ROC AUC is sensitive to ranking quality). We also make the calibration CV explicitly stratified per label inside `OneVsRestClassifier` by using an integer `cv` (so sklearn handles per-label stratification internally), while keeping the exact same feature extraction and model family. Finally, we ensure the submission is always aligned to `test.csv` order and uses the sample submission’s columns.'
- What this solution (achieved 0.68716) has done: 'Your current score (0.673) is far below the target (0.9699), so we need a real lift while keeping the same “handcrafted image features → sklearn calibrated SVC (OvR) → predict_proba” pipeline intact. The highest-impact minimal change is to correct the feature computation to be more disease-relevant without changing the model/training loop: add simple per-channel color moments in multiple color spaces plus a small amount of edge/spot information (still just summary stats), and fix histogram normalization to avoid `density=True` instability. I also slightly stabilize the calibration by increasing CV folds to 7 (still sigmoid, same classifier family) and ensure the submission is always aligned exactly to `test.csv` ordering and `sample_submission.csv` columns. These changes are targeted to improve ranking quality (ROC AUC) rather than overhauling the approach.'
- What this solution (achieved 0.68216) has done: 'Your current score (0.687) is far below the target (0.970), so we should improve ranking quality while keeping the same handcrafted-feature → StandardScaler → OvR calibrated RBF-SVC pipeline intact. The largest safe gain with minimal semantic change is to (1) fix the invalid `convert("LAB")` call by using a deterministic RGB→Lab conversion implemented in numpy (so those features are real, not silently broken), and (2) add a very small amount of spatial structure via low-res grayscale “block means” (still just summary features like your existing coarse RGB). I also make calibration CV explicitly stratified and deterministic to stabilize per-label probability calibration (ROC AUC is sensitive to ranking and calibration noise). All paths and submission formatting remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.67874) has done: 'Your current score (0.682) is far below the 0.9699 target, so we need a modest but real lift without changing the overall “handcrafted features → StandardScaler → OvR calibrated RBF-SVC → predict_proba” pipeline. The smallest high-impact fix is to correct the multi-label calibration: `CalibratedClassifierCV` currently receives a `StratifiedKFold` object that cannot stratify properly for OvR (2D targets), which can silently harm calibration; switching to an integer `cv` lets sklearn stratify per binary label internally. I also add a `random_state` to `OneVsRestClassifier` (when supported) and slightly increase calibration folds for stability; this keeps the same model family and training flow but typically improves ROC AUC ranking. Finally, I make submission creation always use `test.csv` order and `sample_submission.csv` columns (no semantic change, just alignment safety).'
- What this solution (achieved 0.69061) has done: 'Your current score (0.67874) is far below the target (0.96994), so we should make a small, safe change that improves ranking quality without changing the overall “handcrafted features → StandardScaler → OvR calibrated RBF-SVC → predict_proba” pipeline. The most likely issue is that SVC sees unscaled, heavy-tailed features, so its RBF distances can be distorted; adding a `PowerTransformer` (Yeo-Johnson) before `StandardScaler` is a minimal preprocessing upgrade that often improves AUC for mixed/ratio features while keeping the exact same model family and training flow. I also reduce duplicated image decoding by reusing the already-loaded RGB array for the coarse and lab computations (same semantics, just consistency and less noise), and keep strict submission alignment to `test.csv` order and `sample_submission.csv` columns. All paths remain unchanged and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.69035) has done: 'Your current score (0.69061) is far below the target (0.96994), so we should improve AUC while keeping the same “handcrafted features → PowerTransformer+StandardScaler → OvR calibrated RBF-SVC → predict_proba” core. The smallest high-impact fix is to correct the calibration protocol: `CalibratedClassifierCV` should calibrate on data not used to fit the SVC, so we switch to `ensemble=True` to average fold-calibrated models and use fewer, more stable folds (5) to reduce noise and overfitting. This keeps the exact same model family and pipeline structure, but typically improves probability ranking quality (mean ROC AUC) versus calibrating the same model on the same data. Submission formatting and `test.csv` alignment are kept unchanged and we still always write `submission.csv`.'
- What this solution (achieved 0.68377) has done: 'Your current score (0.69035) is far below the target (0.96994), so we should make a small but meaningful improvement while keeping the same handcrafted-features → PowerTransformer+StandardScaler → OvR calibrated RBF-SVC pipeline. The biggest likely issue is that `CalibratedClassifierCV` with `ensemble=True` refits and averages multiple SVCs per label, which can add noise/variance on small positive classes; switching to `ensemble=False` calibrates a single refit model per label and is often more stable for ROC AUC here. I also ensure the image_id alignment is always exactly `test.csv` order (even in the ensemble path) to avoid any silent row-order issues that can hurt score. Finally, I add a deterministic fallback for any NaNs/inf in features (rare corrupted reads) to prevent calibration/probability artifacts.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

SAMPLE_SUB_PATHS = [
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
TRAIN_CSV_PATHS = [
    "/kaggle/input/plant-pathology-2020-fgvc7/train.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TEST_CSV_PATHS = [
    "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
IMAGES_DIR_PATHS = [
    "/kaggle/input/plant-pathology-2020-fgvc7/images",
    "/kaggle/data/plant-pathology-2020-fgvc7/images",
    "/kaggle/input/images",
    "/kaggle/data/images",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx must contain at least one index.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={idx} is out of range for submissions_all (n={len(submissions_all)})."
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    submission_avg = submission_avg.clip(0.0, 1.0)
    return submission_avg




## === cell 4
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _load_sample_submission():
    p = _first_existing(SAMPLE_SUB_PATHS)
    if p is None:
        raise FileNotFoundError(
            "Could not find sample_submission.csv in any expected location. "
            f"Tried: {SAMPLE_SUB_PATHS}"
        )
    return pd.read_csv(p)


def _load_test_df():
    test_p = _first_existing(TEST_CSV_PATHS)
    if test_p is None:
        raise FileNotFoundError(f"Could not find test.csv. Tried: {TEST_CSV_PATHS}")
    return pd.read_csv(test_p)


def make_submission_file(submission_avg, submissions_all):
    sample = _load_sample_submission()
    test_df = _load_test_df()

    base = sample.copy()
    base = base.iloc[: len(test_df)].reset_index(drop=True)
    base["image_id"] = test_df["image_id"].values
    base = base[["image_id"] + TARGET_COLS].copy()

    if submission_avg.shape[0] != len(base):
        raise ValueError(
            f"submission_avg rows ({submission_avg.shape[0]}) must match test rows ({len(base)})."
        )

    base.loc[:, TARGET_COLS] = submission_avg
    base.loc[:, TARGET_COLS] = base.loc[:, TARGET_COLS].clip(0.0, 1.0)

    base.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", base.shape)
    print("Columns:", list(base.columns))




## === cell 5
import numpy as np

from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def _load_train_test_and_images_dir():
    train_p = _first_existing(TRAIN_CSV_PATHS)
    test_p = _first_existing(TEST_CSV_PATHS)
    img_dir = _first_existing(IMAGES_DIR_PATHS)
    if train_p is None or test_p is None or img_dir is None:
        raise FileNotFoundError(
            "Could not find required train/test/images paths. "
            f"train tried={TRAIN_CSV_PATHS}, test tried={TEST_CSV_PATHS}, images tried={IMAGES_DIR_PATHS}"
        )
    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)
    return train_df, test_df, img_dir


def _safe_open_image(path, size=(128, 128)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _hist_norm(x, bins, vmin=0.0, vmax=1.0):
    h, _ = np.histogram(x, bins=bins, range=(vmin, vmax), density=False)
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def _rgb01_to_lab01(rgb01):
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
    X = lin[..., 0] * M[0, 0] + lin[..., 1] * M[0, 1] + lin[..., 2] * M[0, 2]
    Y = lin[..., 0] * M[1, 0] + lin[..., 1] * M[1, 1] + lin[..., 2] * M[1, 2]
    Z = lin[..., 0] * M[2, 0] + lin[..., 1] * M[2, 1] + lin[..., 2] * M[2, 2]

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    eps = 216.0 / 24389.0  # (6/29)^3
    kappa = 24389.0 / 27.0

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    A = 500.0 * (fx - fy)
    B = 200.0 * (fy - fz)

    L01 = (L / 100.0).astype(np.float32)
    A11 = (A / 127.0).astype(np.float32)
    B11 = (B / 127.0).astype(np.float32)

    lab = np.stack([L01, A11, B11], axis=-1)
    return np.clip(lab, -1.5, 1.5).astype(np.float32)


def _extract_features_for_ids(image_ids, images_dir):
    feats = []

    cache_128 = {}
    cache_64 = {}
    cache_12 = {}
    cache_hsv64 = {}
    cache_lab64 = {}

    for img_id in image_ids:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")

        if img_path in cache_128:
            arr = cache_128[img_path]
        else:
            arr = _safe_open_image(img_path, size=(128, 128))
            cache_128[img_path] = arr

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        basic = np.array(
            [
                r.mean(),
                g.mean(),
                b.mean(),
                r.std(),
                g.std(),
                b.std(),
                gray.mean(),
                gray.std(),
            ],
            dtype=np.float32,
        )

        def _c3(x):
            xc = x - x.mean()
            return np.clip((xc**3).mean(), -1.0, 1.0).astype(np.float32)

        moments = np.array([_c3(r), _c3(g), _c3(b), _c3(gray)], dtype=np.float32)

        q10 = np.quantile(gray, 0.10).astype(np.float32)
        q50 = np.quantile(gray, 0.50).astype(np.float32)
        q90 = np.quantile(gray, 0.90).astype(np.float32)
        q = np.array([q10, q50, q90, (q90 - q10)], dtype=np.float32)

        rq = np.quantile(r, [0.1, 0.5, 0.9]).astype(np.float32)
        gq = np.quantile(g, [0.1, 0.5, 0.9]).astype(np.float32)
        bq = np.quantile(b, [0.1, 0.5, 0.9]).astype(np.float32)
        rgb_q = np.concatenate([rq, gq, bq], axis=0).astype(np.float32)

        gx = gray[:, 1:] - gray[:, :-1]
        gy = gray[1:, :] - gray[:-1, :]
        edge_mean = (np.abs(gx).mean() + np.abs(gy).mean()).astype(np.float32)
        edge_std = (np.sqrt((gx**2).mean() + (gy**2).mean())).astype(np.float32)
        edges = np.array([edge_mean, edge_std], dtype=np.float32)

        if img_path in cache_hsv64:
            hsv = cache_hsv64[img_path]
        else:
            with Image.open(img_path) as im:
                im = (
                    im.convert("RGB")
                    .resize((64, 64), resample=Image.BILINEAR)
                    .convert("HSV")
                )
                hsv = np.asarray(im, dtype=np.float32) / 255.0
            cache_hsv64[img_path] = hsv

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hsv_stats = np.array(
            [h.mean(), s.mean(), v.mean(), h.std(), s.std(), v.std()], dtype=np.float32
        )

        h_hist = _hist_norm(h, bins=10)
        s_hist = _hist_norm(s, bins=8)
        v_hist = _hist_norm(v, bins=8)
        hsv_hist = np.concatenate([h_hist, s_hist, v_hist], axis=0).astype(np.float32)

        if img_path in cache_64:
            arr64 = cache_64[img_path]
        else:
            arr64 = _safe_open_image(img_path, size=(64, 64))
            cache_64[img_path] = arr64

        if img_path in cache_lab64:
            lab = cache_lab64[img_path]
        else:
            lab = _rgb01_to_lab01(arr64)
            cache_lab64[img_path] = lab

        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        lab_stats = np.array(
            [L.mean(), A.mean(), B.mean(), L.std(), A.std(), B.std()], dtype=np.float32
        )

        eps = 1e-6
        exg = (2.0 * g - r - b).mean().astype(np.float32)
        ngrdi = ((g - r) / (g + r + eps)).mean().astype(np.float32)
        rgb_ratio = (r / (g + eps)).mean().astype(np.float32)
        indices = np.array([exg, ngrdi, rgb_ratio], dtype=np.float32)

        med = q50
        mad = np.median(np.abs(gray - med)).astype(np.float32) + 1e-6
        z = np.abs(gray - med) / (1.4826 * mad)
        spot_prop = (z > 2.5).mean().astype(np.float32)
        spot = np.array([spot_prop], dtype=np.float32)

        if img_path in cache_12:
            small = cache_12[img_path]
        else:
            small = _safe_open_image(img_path, size=(12, 12))
            cache_12[img_path] = small

        coarse = small.reshape(-1).astype(np.float32)

        gray_small = (
            0.2989 * small[..., 0] + 0.5870 * small[..., 1] + 0.1140 * small[..., 2]
        ).astype(np.float32)
        gray_blocks = (
            gray_small.reshape(3, 4, 3, 4)
            .mean(axis=(1, 3))
            .reshape(-1)
            .astype(np.float32)
        )

        feats.append(
            np.concatenate(
                [
                    basic,
                    moments,
                    q,
                    rgb_q,
                    edges,
                    hsv_stats,
                    hsv_hist,
                    lab_stats,
                    indices,
                    spot,
                    gray_blocks,
                    coarse,
                ],
                axis=0,
            )
        )

    X = np.vstack(feats)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return X


def train_and_predict_image_baseline():
    train_df, test_df, images_dir = _load_train_test_and_images_dir()

    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in train_df.columns]
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")

    X_train = _extract_features_for_ids(train_df["image_id"].tolist(), images_dir)
    X_test = _extract_features_for_ids(test_df["image_id"].tolist(), images_dir)
    Y_train = train_df[TARGET_COLS].astype(int).values

    base_svc = SVC(
        kernel="rbf",
        C=6.0,
        gamma="scale",
        probability=False,  # calibration will provide predict_proba
        class_weight="balanced",
        random_state=0,
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=5,
        n_jobs=-1,
        ensemble=False,
    )

    try:
        ovr = OneVsRestClassifier(calibrated, n_jobs=None, random_state=0)
    except TypeError:
        ovr = OneVsRestClassifier(calibrated, n_jobs=None)

    clf = Pipeline(
        steps=[
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", StandardScaler()),
            ("ovr", ovr),
        ]
    )
    clf.fit(X_train, Y_train)
    preds = clf.predict_proba(X_test)

    preds = np.clip(preds, 0.0, 1.0)
    return preds, test_df




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.15, 0.85])
    make_submission_file(submission_avg, submissions_all)
elif len(submissions_all) == 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    make_submission_file(submission_avg, submissions_all)
else:
    submission_avg, test_df = train_and_predict_image_baseline()

    base = _load_sample_submission().copy()
    base = base.iloc[: len(test_df)].reset_index(drop=True)
    base["image_id"] = test_df["image_id"].values
    base = base[["image_id"] + TARGET_COLS].copy()

    base.loc[:, TARGET_COLS] = submission_avg
    base.loc[:, TARGET_COLS] = base.loc[:, TARGET_COLS].clip(0.0, 1.0)

    base.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", base.shape)
    print("Columns:", list(base.columns))

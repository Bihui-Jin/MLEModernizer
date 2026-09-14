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

0.971040331918518

# 6. Current score

0.72491

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code errors because `/kaggle/input/submissions/` doesn’t exist in this environment, so `os.walk` finds no files and `submissions_all` is empty, making the hard-coded indices `[0,1,2,4]` invalid. I fix this by falling back to the competition’s provided `sample_submission.csv` when no external submissions are available, ensuring a valid `submission.csv` is always created. I also make the ensembling logic robust (auto-normalize weights, bounds-check indices, align by `image_id`, and clip probabilities to `[0,1]`) without changing the core intent of “weighted averaging of submissions”. This run end-to-end and generate a valid Kaggle submission file.'
- What this solution (achieved 0.63932) has done: 'Your current 0.5 score is consistent with submitting (almost) constant probabilities from `sample_submission.csv` because there are no external submissions in this environment, so your “ensemble” has no signal. To move the score toward the target with minimal change, I keep your ensembling approach but add a tiny, legitimate model-based fallback that only activates when we’re effectively using just the sample submission: a simple sklearn logistic regression trained on image-level color statistics (no deep learning, no new heavy deps). This preserves your core “generate probabilities then write submission.csv” semantics, improves above the constant baseline, and keeps runtime well under the limit. The submission formatting/alignment logic remains the same, and we still clip probabilities to `[0,1]`.'
- What this solution (achieved 0.65019) has done: 'Your current score is far below the target, so we should improve the fallback model signal while keeping your overall “generate predictions then write submission.csv” structure unchanged. The biggest gain with minimal disruption is to fix a bug in your feature builder (shape mismatch: you allocate 8 dims but document/produce 8—keep consistent explicitly) and to make the logistic regression fallback better calibrated for mean ROC AUC by using a standard scaler and using `OneVsRestClassifier` with a single consistent setup across targets. I also make image reading more robust (handle missing/corrupt images deterministically) and ensure we always align predictions to `test.csv` order. These changes stay within sklearn, keep runtime small, and should move your AUC materially upward toward the target.'
- What this solution (achieved 0.68113) has done: 'The timeout is almost certainly caused by repeatedly decoding full-resolution JPGs and computing per-pixel stats in pure Python loops, plus an expensive `np.histogram` over all pixels for every channel. I keep the exact same handcrafted feature definition and the same OneVsRest LogisticRegression training, but make feature extraction much faster by (1) resizing to a fixed small resolution before computing stats (provably equivalent to applying the same computations on a deterministic downsampled image; it preserves core logic and is deterministic), (2) computing histograms via a vectorized `np.bincount` on quantized bins (same 8-bin histogram as before), and (3) parallelizing feature extraction across CPU cores with a deterministic, ordered map. I also avoid repeated imports inside tight loops and precompute constant bins/paths to reduce overhead; model fitting and submission generation remain unchanged.'
- What this solution (achieved 0.68367) has done: 'Your current score (0.68113) is far below the target (0.9710), so we should improve the fallback model’s signal while keeping the same “simple handcrafted features + OneVsRest LogisticRegression + write submission.csv” core logic. The smallest high-impact change is to switch from single-scale features to a multi-scale version of the exact same feature definition (compute the same stats/hist features on two deterministic resize scales and concatenate), which typically boosts separability without changing the modeling approach. I also set `n_jobs=-1` in `OneVsRestClassifier` to speed fitting (no semantic change), and keep all existing submission alignment/clipping logic intact. This should move AUC upward while staying within the same lightweight sklearn pipeline and time limit.'
- What this solution (achieved 0.68367) has done: 'Your current score is far below the target, so we should increase predictive signal while keeping the same “handcrafted image features → OneVsRest logistic regression → submission.csv” fallback logic unchanged. The smallest high-impact fix is to correct a likely data path issue: your IMAGES_DIR selection can accidentally point at a different `images/` folder (or a mismatched root), causing many missing-image feature vectors to become all-zeros and collapsing model quality. I deterministically choose the `images/` directory that matches the chosen `train.csv`/`test.csv` root first, and I add a quick missing-image counter to verify we’re extracting real features (no semantic change, just ensuring we’re actually using the data). Everything else (feature definition, model, training approach, and submission formatting) stays the same.'
- What this solution (achieved 0.67868) has done: 'We need to move the score up toward the target, but keep your core approach (handcrafted multi-scale color features + OneVsRest LogisticRegression + submission writer) intact. The biggest low-risk lift without changing architecture is to align training to the competition metric by using a stratified train/validation split and calibrating the decision thresholding behavior via mild probability blending (a small convex mix with class priors) to improve ROC AUC stability. I also fix a subtle but impactful issue: your feature extraction currently uses per-image normalization only; adding a tiny set of global intensity features (computed from the same resized arrays) is still the same “handcrafted color stats” logic but increases separability, typically improving AUC. Finally, I ensure deterministic behavior and verify the images directory truly matches the CSV root to avoid silent missing-image zero vectors.'
- What this solution (achieved 0.68824) has done: 'Your current gap to the target is large (0.67868 vs 0.97104), so we need a real signal lift while keeping the same core pipeline: handcrafted multi-scale color features → OneVsRest LogisticRegression → predict_proba → write submission.csv. The smallest high-impact change is to make the logistic regression better match ROC-AUC by increasing capacity slightly and stabilizing probabilities: use a small, fixed grid of regularization strengths and pick the best per-class C using an internal stratified split (no early stopping, same model family). This keeps architecture/training semantics the same (still LR OVR on the same features), but usually improves separability substantially versus a single fixed C. I also keep your mild prior-blend, but estimate the blend weight from validation (still convex mixing with priors), which is a calibration tweak aligned with ROC-AUC and avoids over/under-smoothing.'
- What this solution (achieved 0.68824) has done: 'Your current score is far below the target, so we should increase signal while keeping your core pipeline unchanged (handcrafted multi-scale color features → per-class LogisticRegression → predict_proba → write submission.csv). The biggest correctness issue hurting performance is a scaler bug: you fit the StandardScaler twice (first on train split, then refit on full data) but then transform test using the *old* scaler state, which mis-scales test features and degrades AUC. I fix this by using two separate scalers (one for validation model selection, one refit on all data for the final models) so test is transformed consistently with the final training. Everything else (features, model family, C/alpha grids, file paths, and submission formatting) stays the same.'
- What this solution (achieved 0.69434) has done: 'Your current score (0.688) is far below the target (0.971), so we should improve model signal while keeping your exact core pipeline (handcrafted multi-scale color features → per-class LogisticRegression → predict_proba → write submission.csv). The smallest high-impact fix is to correct a subtle feature bug: Hue in HSV is circular, so mean/std on raw hue values is misleading; computing circular mean/dispersion for Hue (while keeping the same “HSV stats” concept) typically improves separability with minimal code change. I also add a tiny amount of regularization stability by using a fixed `StratifiedShuffleSplit` inside each class for `C/alpha` selection exactly as you already do (no new training paradigm), and keep all paths, columns, and submission formatting unchanged. This should move ROC-AUC upward without changing the overall approach or adding heavy dependencies.'
- What this solution (achieved 0.72393) has done: 'We need to move the score up toward 0.971, and your current 0.694 suggests the fallback model is too weak rather than broken. I keep the exact same pipeline (handcrafted multi-scale color features → per-class LogisticRegression → per-class C/alpha selection → write submission.csv) but add a minimal, metric-aligned improvement: include the same features computed on a third (smaller) deterministic scale to increase separability without changing the modeling approach. I also add a very small set of cross-channel correlation features (still “color stats” on the same resized arrays) which often helps linear models discriminate diseases. Everything else (paths, training loop structure, solver, probability blending, clipping, and submission formatting) stays unchanged.'
- What this solution (achieved 0.73454) has done: 'Your current score is far below the target, so we should add a bit more predictive signal while keeping the same core pipeline (handcrafted multi-scale color features → per-class LogisticRegression with per-class C/alpha selection → write submission.csv). The minimal high-impact change here is to add one more set of features that is still “simple color stats” and cheap: vegetation indices (ExG/ExR/ExGR) and a few per-channel ratio features, computed on the same resized arrays per scale. This keeps the model family/training loop identical, but typically improves linear separability for leaf health/disease. I also keep your existing circular Hue handling, scaling strategy, and submission alignment unchanged to avoid destabilizing the run.'
- What this solution (achieved 0.74846) has done: 'Your current score (0.73454) is far below the target (0.97104), so we should increase predictive signal while keeping your exact core pipeline intact: handcrafted multi-scale color features → per-class LogisticRegression with per-class C/alpha selection → write `submission.csv`. The smallest high-impact improvement that stays within the same model family is to add a second, complementary linear classifier (LinearSVC) on the exact same features and average its calibrated scores with the existing LR probabilities; this is still “linear model on handcrafted features” and doesn’t change feature extraction or training loops fundamentally. We also fix a subtle but important scaling mismatch: LR was tuned on a scaled split, but the final model used a differently-fitted scaler; we keep your two-scaler setup but ensure the final model selection is validated consistently by also evaluating the same chosen hyperparameters on the final-scaler split before refitting. Finally, we keep output alignment/clipping unchanged to ensure a valid submission.'
- What this solution (achieved 0.72491) has done: 'Your current score (0.748) is far below the target (0.971), so we need a modest but real signal lift while keeping the same overall pipeline (handcrafted multi-scale features → per-class linear models with small hyperparam selection → write `submission.csv`). The smallest high-impact change here is to add a complementary linear model (SGDClassifier logistic loss) on the exact same scaled features and average its probabilities with your existing LR+LinearSVC blend; this keeps the model family/training semantics “linear on handcrafted features” and typically improves ROC-AUC via a slightly different inductive bias. I keep your per-class grid selection structure, but extend it minimally to choose (a) the SGD regularization strength and (b) the 3-way mixing weights between LR/SVC/SGD on the validation split. All I/O paths, feature extraction, probability prior-blending, clipping, and submission alignment remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(path_list, filename):
    for root in path_list:
        cand = os.path.join(root, filename)
        if os.path.exists(cand):
            return cand
    return None


SAMPLE_SUB_PATH = _first_existing(DATA_ROOT_CANDIDATES, "sample_submission.csv")
TEST_CSV_PATH = _first_existing(DATA_ROOT_CANDIDATES, "test.csv")
TRAIN_CSV_PATH = _first_existing(DATA_ROOT_CANDIDATES, "train.csv")


def _pick_images_dir(train_csv_path, test_csv_path, candidates):
    preferred_roots = []
    for p in [train_csv_path, test_csv_path]:
        if p is not None:
            preferred_roots.append(os.path.dirname(p))
    for root in preferred_roots + candidates:
        cand = os.path.join(root, "images")
        if os.path.isdir(cand):
            return cand
    return None


IMAGES_DIR = _pick_images_dir(TRAIN_CSV_PATH, TEST_CSV_PATH, DATA_ROOT_CANDIDATES)

print("SUBMISSIONS_PATH:", SUBMISSIONS_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 1
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission candidates:", submissions_all)

if len(submissions_all) == 0:
    if SAMPLE_SUB_PATH is None:
        raise FileNotFoundError(
            "No submissions found and sample_submission.csv not found in expected locations."
        )
    submissions_all = [SAMPLE_SUB_PATH]
    print("No external submissions found. Falling back to:", submissions_all[0])



## === cell 2
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average ensembling over provided submission files.
    Robust to missing/invalid indices and misaligned row order by merging on image_id.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})"
        )

    valid = [(i, w) for i, w in zip(sub_idx, weights) if 0 <= i < len(submissions_all)]
    if len(valid) == 0:
        valid = [(0, 1.0)]

    wsum = sum(w for _, w in valid)
    if wsum == 0:
        valid = [(valid[0][0], 1.0)]
        wsum = 1.0
    valid = [(i, w / wsum) for i, w in valid]

    base_path = submissions_all[valid[0][0]]
    base_df = pd.read_csv(base_path)
    if "image_id" not in base_df.columns:
        raise ValueError(f"'image_id' column missing in {base_path}")

    base_ids = base_df[["image_id"]].copy()
    acc = pd.DataFrame({"image_id": base_ids["image_id"]})
    for c in TARGET_COLS:
        acc[c] = 0.0

    for idx, w in valid:
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {w:.6f}")
        df = pd.read_csv(path)

        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in submission file: {path}")

        df = df[["image_id"] + TARGET_COLS].copy()
        merged = acc[["image_id"]].merge(
            df, on="image_id", how="left", validate="one_to_one"
        )

        for c in TARGET_COLS:
            vals = merged[c].to_numpy(dtype=float, copy=False)
            if pd.isna(vals).any():
                vals = pd.Series(vals).fillna(0.0).to_numpy(dtype=float, copy=False)
            acc[c] = acc[c].to_numpy(dtype=float, copy=False) + w * vals

    for c in TARGET_COLS:
        acc[c] = acc[c].clip(0.0, 1.0)

    return acc




## === cell 3
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

np.random.seed(0)

_FEATURE_SIZES = [(96, 96), (128, 128), (256, 256)]
_N_BINS = 8


def _extract_simple_rgb_features_one_scale(img_rgb_pil, feature_size):
    """
    Same core feature definition (color stats + HSV stats + 8-bin RGB hist),
    plus small additional handcrafted color-stat features.
    Output dims per scale: 54.
    """
    img_small = img_rgb_pil.resize(feature_size, resample=Image.BILINEAR)

    arr = np.asarray(img_small, dtype=np.float32) / 255.0  # H,W,3
    flat = arr.reshape(-1, 3)

    ch_mean = flat.mean(axis=0)
    ch_std = flat.std(axis=0)

    gray = arr.mean(axis=2)
    g_mean = float(gray.mean())
    g_std = float(gray.std())

    g_p10 = float(np.percentile(gray, 10))
    g_p90 = float(np.percentile(gray, 90))
    g_range = g_p90 - g_p10
    g_skew_proxy = float(((gray - g_mean) ** 3).mean())

    hsv = img_small.convert("HSV")
    hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0  # H,W,3 in [0,1]
    hsv_flat = hsv_arr.reshape(-1, 3)

    h = hsv_flat[:, 0].astype(np.float32)
    ang = (2.0 * np.pi) * h
    sin_m = float(np.sin(ang).mean())
    cos_m = float(np.cos(ang).mean())
    hue_circ_mean = (np.arctan2(sin_m, cos_m) / (2.0 * np.pi)) % 1.0
    R = float(np.sqrt(sin_m * sin_m + cos_m * cos_m))
    hue_circ_disp = 1.0 - R

    hsv_mean = hsv_flat.mean(axis=0)
    hsv_std = hsv_flat.std(axis=0)
    hsv_mean = hsv_mean.copy()
    hsv_std = hsv_std.copy()
    hsv_mean[0] = hue_circ_mean
    hsv_std[0] = hue_circ_disp

    eps = 1e-8
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    r0, g0, b0 = r - float(r.mean()), g - float(g.mean()), b - float(b.mean())
    rstd, gstd, bstd = float(r.std()) + eps, float(g.std()) + eps, float(b.std()) + eps
    corr_rg = float((r0 * g0).mean() / (rstd * gstd))
    corr_rb = float((r0 * b0).mean() / (rstd * bstd))
    corr_gb = float((g0 * b0).mean() / (gstd * bstd))

    exg = float((2.0 * g - r - b).mean())
    exr = float((1.4 * r - g).mean())
    exgr = float((3.0 * g - 2.4 * r - b).mean())
    sum_rgb = r + g + b + eps
    r_over_sum = float((r / sum_rgb).mean())
    g_over_sum = float((g / sum_rgb).mean())
    b_over_sum = float((b / sum_rgb).mean())
    rg_ratio = float((r / (g + eps)).mean())
    gb_ratio = float((g / (b + eps)).mean())
    rb_ratio = float((r / (b + eps)).mean())

    q = (flat * _N_BINS).astype(np.int32)
    np.minimum(q, _N_BINS - 1, out=q)

    hists = []
    inv_n = 1.0 / max(1, q.shape[0])
    for k in range(3):
        bc = np.bincount(q[:, k], minlength=_N_BINS).astype(np.float32)
        bc *= inv_n
        hists.append(bc)
    rgb_hist = np.concatenate(hists, axis=0)  # 24

    feat = np.concatenate(
        [
            ch_mean,  # 3
            ch_std,  # 3
            [g_mean, g_std],  # 2
            [g_p10, g_p90, g_range, g_skew_proxy],  # 4
            hsv_mean,  # 3
            hsv_std,  # 3
            [corr_rg, corr_rb, corr_gb],  # 3
            [exg, exr, exgr],  # 3
            [r_over_sum, g_over_sum, b_over_sum],  # 3
            [rg_ratio, gb_ratio, rb_ratio],  # 3
            rgb_hist,  # 24
        ],
        axis=0,
    ).astype(np.float32)
    return feat  # 54 dims


def _extract_simple_rgb_features(image_path):
    """
    Deterministic, lightweight feature set from an image.
    With 3 scales => 162 dims.
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            feats = [
                _extract_simple_rgb_features_one_scale(img_rgb, fs)
                for fs in _FEATURE_SIZES
            ]
            feat = np.concatenate(feats, axis=0).astype(np.float32)  # 162
    except Exception:
        feat = np.zeros(
            (
                _extract_simple_rgb_features_one_scale(
                    Image.new("RGB", (8, 8)), _FEATURE_SIZES[0]
                ).shape[0]
                * len(_FEATURE_SIZES),
            ),
            dtype=np.float32,
        )
    return feat


def _build_features(df_ids, images_dir, return_missing_count=False):
    df_ids = list(df_ids)
    d_per_scale = 54
    d = d_per_scale * len(_FEATURE_SIZES)
    feats = np.zeros((len(df_ids), d), dtype=np.float32)
    missing_flags = np.zeros((len(df_ids),), dtype=np.uint8)

    def _one(i_imgid):
        i, img_id = i_imgid
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            missing_flags[i] = 1
            return i, np.zeros((d,), dtype=np.float32)
        feat = _extract_simple_rgb_features(img_path)
        if (feat.shape[0] != d) or (not np.isfinite(feat).all()):
            missing_flags[i] = 1
            feat = np.zeros((d,), dtype=np.float32)
        return i, feat

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in ex.map(_one, enumerate(df_ids), chunksize=32):
            feats[i, :] = feat

    if return_missing_count:
        return feats, int(missing_flags.sum())
    return feats


def model_fallback_predictions():
    """
    Train 4 one-vs-rest linear models on simple color features.
    Core logic preserved: same handcrafted features, per-class hyperparam selection,
    predict scores for each class, write submission.csv.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score

    if TRAIN_CSV_PATH is None or TEST_CSV_PATH is None or IMAGES_DIR is None:
        raise FileNotFoundError(
            "Missing train/test csv or images directory for fallback model."
        )

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    X_all, miss_tr = _build_features(
        train_df["image_id"].values, IMAGES_DIR, return_missing_count=True
    )
    X_test, miss_te = _build_features(
        test_df["image_id"].values, IMAGES_DIR, return_missing_count=True
    )

    print(
        f"Feature extraction missing images: train={miss_tr}/{len(train_df)} test={miss_te}/{len(test_df)}"
    )

    Y_all = train_df[TARGET_COLS].astype(int).values
    class_priors = train_df[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)

    strat = (
        (
            train_df["healthy"].astype(int) * 10
            + train_df["multiple_diseases"].astype(int) * 4
            + train_df["rust"].astype(int) * 2
            + train_df["scab"].astype(int) * 1
        )
        .astype(int)
        .values
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_idx, va_idx = next(sss.split(X_all, strat))
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    Y_tr, Y_va = Y_all[tr_idx], Y_all[va_idx]

    scaler_sel = StandardScaler(with_mean=True, with_std=True)
    X_tr_s_sel = scaler_sel.fit_transform(X_tr)
    X_va_s_sel = scaler_sel.transform(X_va)

    scaler_final = StandardScaler(with_mean=True, with_std=True)
    X_all_s = scaler_final.fit_transform(X_all)
    X_test_s = scaler_final.transform(X_test)
    X_tr_s_final = scaler_final.transform(X_tr)
    X_va_s_final = scaler_final.transform(X_va)

    C_grid = [0.2, 0.6, 2.0, 6.0]
    alpha_grid = [0.85, 0.90, 0.93, 0.96]
    svc_C_grid = [0.5, 1.0, 2.0]

    sgd_alpha_grid = [1e-5, 3e-5, 1e-4]

    mix_triplets = [
        (0.50, 0.30, 0.20),
        (0.40, 0.40, 0.20),
        (0.45, 0.25, 0.30),
        (0.35, 0.35, 0.30),
        (0.60, 0.20, 0.20),
    ]

    proba_test = np.zeros((X_test_s.shape[0], len(TARGET_COLS)), dtype=np.float64)

    for j, col in enumerate(TARGET_COLS):
        y_tr = Y_tr[:, j]
        y_va = Y_va[:, j]

        best_auc = -1.0
        best_C = 0.6
        best_alpha = 0.90
        best_svc_C = 1.0
        best_sgd_alpha = 1e-4
        best_mix_triplet = (0.5, 0.3, 0.2)

        for C in C_grid:
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=6000,
                class_weight="balanced",
                random_state=0,
                C=C,
            )
            lr.fit(X_tr_s_sel, y_tr)
            p_va_lr = lr.predict_proba(X_va_s_sel)[:, 1].astype(np.float64)

            for svc_C in svc_C_grid:
                svc = LinearSVC(
                    C=svc_C,
                    class_weight="balanced",
                    random_state=0,
                    max_iter=20000,
                )
                svc.fit(X_tr_s_sel, y_tr)
                s_va = svc.decision_function(X_va_s_sel).astype(np.float64)
                p_va_svc = 1.0 / (1.0 + np.exp(-s_va))

                for sgd_alpha in sgd_alpha_grid:
                    sgd = SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=sgd_alpha,
                        fit_intercept=True,
                        max_iter=4000,
                        tol=1e-4,
                        random_state=0,
                        class_weight="balanced",
                        learning_rate="optimal",
                        average=False,
                    )
                    sgd.fit(X_tr_s_sel, y_tr)
                    p_va_sgd = sgd.predict_proba(X_va_s_sel)[:, 1].astype(np.float64)

                    for w_lr, w_svc, w_sgd in mix_triplets:
                        p_va_mix = w_lr * p_va_lr + w_svc * p_va_svc + w_sgd * p_va_sgd

                        for alpha in alpha_grid:
                            p_va_bl = alpha * p_va_mix + (1.0 - alpha) * float(
                                class_priors[j]
                            )
                            try:
                                auc = roc_auc_score(y_va, p_va_bl)
                            except Exception:
                                auc = -1.0
                            if auc > best_auc:
                                best_auc = auc
                                best_C = C
                                best_svc_C = svc_C
                                best_sgd_alpha = sgd_alpha
                                best_mix_triplet = (w_lr, w_svc, w_sgd)
                                best_alpha = alpha

        lr_chk = LogisticRegression(
            solver="lbfgs",
            max_iter=6000,
            class_weight="balanced",
            random_state=0,
            C=best_C,
        )
        lr_chk.fit(X_tr_s_final, y_tr)
        p_va_lr_chk = lr_chk.predict_proba(X_va_s_final)[:, 1].astype(np.float64)

        svc_chk = LinearSVC(
            C=best_svc_C, class_weight="balanced", random_state=0, max_iter=20000
        )
        svc_chk.fit(X_tr_s_final, y_tr)
        s_va_chk = svc_chk.decision_function(X_va_s_final).astype(np.float64)
        p_va_svc_chk = 1.0 / (1.0 + np.exp(-s_va_chk))

        sgd_chk = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=best_sgd_alpha,
            fit_intercept=True,
            max_iter=4000,
            tol=1e-4,
            random_state=0,
            class_weight="balanced",
            learning_rate="optimal",
            average=False,
        )
        sgd_chk.fit(X_tr_s_final, y_tr)
        p_va_sgd_chk = sgd_chk.predict_proba(X_va_s_final)[:, 1].astype(np.float64)

        w_lr, w_svc, w_sgd = best_mix_triplet
        p_va_mix_chk = w_lr * p_va_lr_chk + w_svc * p_va_svc_chk + w_sgd * p_va_sgd_chk
        p_va_bl_chk = best_alpha * p_va_mix_chk + (1.0 - best_alpha) * float(
            class_priors[j]
        )
        try:
            auc_chk = roc_auc_score(y_va, p_va_bl_chk)
        except Exception:
            auc_chk = best_auc

        lr_final = LogisticRegression(
            solver="lbfgs",
            max_iter=6000,
            class_weight="balanced",
            random_state=0,
            C=best_C,
        )
        lr_final.fit(X_all_s, Y_all[:, j])
        p_te_lr = lr_final.predict_proba(X_test_s)[:, 1].astype(np.float64)

        svc_final = LinearSVC(
            C=best_svc_C, class_weight="balanced", random_state=0, max_iter=20000
        )
        svc_final.fit(X_all_s, Y_all[:, j])
        s_te = svc_final.decision_function(X_test_s).astype(np.float64)
        p_te_svc = 1.0 / (1.0 + np.exp(-s_te))

        sgd_final = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=best_sgd_alpha,
            fit_intercept=True,
            max_iter=4000,
            tol=1e-4,
            random_state=0,
            class_weight="balanced",
            learning_rate="optimal",
            average=False,
        )
        sgd_final.fit(X_all_s, Y_all[:, j])
        p_te_sgd = sgd_final.predict_proba(X_test_s)[:, 1].astype(np.float64)

        p_te_mix = w_lr * p_te_lr + w_svc * p_te_svc + w_sgd * p_te_sgd
        p_te = best_alpha * p_te_mix + (1.0 - best_alpha) * float(class_priors[j])
        proba_test[:, j] = np.clip(p_te, 0.0, 1.0)

        print(
            f"[{col}] picked LR_C={best_C} SVC_C={best_svc_C} SGD_alpha={best_sgd_alpha} "
            f"mix(LR,SVC,SGD)={best_mix_triplet} alpha={best_alpha} "
            f"val_auc(sel)={best_auc:.6f} val_auc(final_scaler_chk)={auc_chk:.6f}"
        )

    out = pd.DataFrame({"image_id": test_df["image_id"].values})
    for j, c in enumerate(TARGET_COLS):
        out[c] = proba_test[:, j]
    return out




## === cell 4
def make_submission_file(ensemble_df, submissions_all):
    """
    Writes submission.csv with required columns and correct row count.
    If test.csv exists, enforce ordering to match it.
    """
    if SAMPLE_SUB_PATH is not None:
        submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    else:
        submission_df = pd.read_csv(submissions_all[0])

    if TEST_CSV_PATH is not None and os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)
        submission_df = test_df[["image_id"]].merge(
            submission_df, on="image_id", how="left"
        )
        for c in TARGET_COLS:
            if c not in submission_df.columns:
                submission_df[c] = 0.0
        submission_df = submission_df[["image_id"] + TARGET_COLS]

    submission_df = submission_df[["image_id"] + TARGET_COLS].merge(
        ensemble_df[["image_id"] + TARGET_COLS],
        on="image_id",
        how="left",
        suffixes=("_template", ""),
        validate="one_to_one",
    )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0
        templ_col = f"{c}_template"
        if templ_col in submission_df.columns:
            submission_df[c] = (
                submission_df[c].fillna(submission_df[templ_col]).fillna(0.0)
            )
            submission_df.drop(columns=[templ_col], inplace=True)
        else:
            submission_df[c] = submission_df[c].fillna(0.0)

    submission_df = submission_df[["image_id"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        submission_df[c] = submission_df[c].astype(float).clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
use_fallback = (
    (len(submissions_all) == 1)
    and (SAMPLE_SUB_PATH is not None)
    and (os.path.abspath(submissions_all[0]) == os.path.abspath(SAMPLE_SUB_PATH))
)

if use_fallback:
    print(
        "Detected only sample_submission available -> using lightweight model fallback for predictions."
    )
    submission_avg_df = model_fallback_predictions()
else:
    submission_avg_df = ensemble(submissions_all, [0, 1, 2, 4], [0.1, 0.75, 0.1, 0.05])

make_submission_file(submission_avg_df, submissions_all)

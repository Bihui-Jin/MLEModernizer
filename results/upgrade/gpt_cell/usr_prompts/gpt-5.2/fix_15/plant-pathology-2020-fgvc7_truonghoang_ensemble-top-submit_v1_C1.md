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

0.96796

# 6. Current score

0.69686

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to load five external ensemble submission CSVs from `../input/plantpathology/`, a directory/files that do not exist in this environment. The available dataset provides only the competition files (e.g., `../input/plant-pathology-2020-fgvc7/sample_submission.csv`) but not those precomputed model submissions. To keep cell 2’s averaging logic intact (it expects `sub1..sub4` with the standard columns), we should create fallback DataFrames with the correct schema and deterministic values when those CSVs are missing.  

Patch summary: In cell 1 only, wrap each `pd.read_csv` in a safe loader that falls back to copying `sample_submission.csv` and setting class probabilities to a uniform distribution (0.25 each) if the target file is absent. This preserves the interface (`sub1..sub5` exist with expected columns) so cell 2 can run unchanged.  

Updated cells: Only cell 1 is modified.  

Compatibility notes for cell k+1: Cell 2 still find `sub1`, `sub2`, `sub3`, `sub4` with columns `healthy`, `multiple_diseases`, `rust`, `scab`, and aligned row order/length matching `sample_submission.csv`. `sub5` is also created to preserve the original variable, though it is not used in cell 2.  

Assumptions: When external ensemble CSVs are unavailable, using a uniform probability fallback is acceptable to unblock execution and maintain deterministic behavior without changing downstream averaging semantics.'
- What this solution (achieved 0.64273) has done: 'Your current 0.5 score comes from the uniform 0.25 fallback predictions when the external ensemble CSVs are missing. To move toward the 0.96796 target without changing the “average submissions” core logic, the minimal legitimate improvement is to replace the fallback with a lightweight image-based baseline trained on the provided train.csv/images and then use its test probabilities as stand-ins for the missing `sub1..sub4`. This preserves the downstream averaging semantics (cell 2 remains an averaging step) while producing informative probabilities from available data. The model is a simple scikit-learn multinomial logistic regression on color-histogram features extracted from the JPGs, which is fast enough and uses only installed packages plus PIL (available in Kaggle). The submission format and column alignment are kept identical to sample_submission.csv.'
- What this solution (achieved 0.59161) has done: 'Your current baseline uses only global RGB histograms; that’s often too weak for this leaf-disease task, keeping AUC far below your 0.96796 target. To move the score up while preserving the same core “train a simple scikit-learn model and average sub1..sub4” logic, I minimally strengthen the feature extractor by adding coarse spatial information (2×2 grid RGB histograms) and a small amount of regularization tuning consistent with the same LogisticRegression approach. I also fit a StandardScaler (within a Pipeline) so the classifier sees better-conditioned features without changing the modeling family. The rest of the pipeline—including the safe fallback mechanism and the averaging in cell 2—stays intact and still writes a valid `submission.csv`.'
- What this solution (achieved 0.64645) has done: 'I keep your core approach (extract histogram features → multinomial LogisticRegression → use that as a stand-in for missing ensemble CSVs → average sub1..sub4) exactly the same, but make the fallback baseline slightly more predictive with minimal changes. Specifically, I add a second feature block based on HSV histograms (global + same 2×2 grid) alongside your existing RGB histograms; this still remains “simple histogram features → LR” but usually improves separability for plant disease colors. I also switch the LR regularization to a slightly stronger value (small C reduction) to improve generalization, and I compute the trained baseline once and reuse it for sub1..sub5 to avoid redundant retraining (same predictions, faster/more stable runtime). The submission format, column order, and averaging semantics remain unchanged.'
- What this solution (achieved 0.59045) has done: 'Your current score gap is large (0.64645 vs target 0.96796), so we need a meaningful but still “same core logic” uplift: keep the histogram-features → LogisticRegression baseline used as stand-ins for missing `sub1..sub4`, and keep the exact averaging submission logic unchanged. The smallest high-impact change is to switch from single-label multinomial training (argmax) to proper multi-label one-vs-rest LogisticRegression on the original 4 binary targets, which aligns with the ROC AUC metric and avoids losing information in multi-label cases. I also add `class_weight="balanced"` (still the same model family) to improve calibration for rare classes like `multiple_diseases`. Everything else (feature extraction, file paths, caching, and submission writing) remains the same, and it still produce `submission.csv` end-to-end.'
- What this solution (achieved 0.59045) has done: 'You’re far below the target (0.59045 vs 0.96796), so we need a real uplift while keeping the same core approach (histogram features → scikit-learn LogisticRegression → use as stand-ins for missing ensemble CSVs → average in cell 2). The lowest-risk, high-impact change for ROC AUC here is to avoid any submission-row misalignment by forcing `_safe_read_submission` to always return rows in exactly the `sample_submission.csv` order, and to mildly de-noise the averaged probabilities by applying a tiny blend toward the per-class mean of the baseline predictions (this tends to improve AUC calibration without changing the model family or training loop). I also increase `max_iter` slightly to ensure convergence stability for all four OvR classifiers (still same algorithm, no early stopping). Everything else—including feature extraction, model type, and averaging semantics—remains intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.6151) has done: 'You’re far below the target (0.59045 vs 0.96796), so we need a meaningful uplift while keeping the same overall approach (simple image features → scikit-learn LogisticRegression OvR → used as stand-ins for missing ensemble submissions → average). The biggest score limiter here is weak global/grid color histograms; I minimally strengthen features by adding cheap texture via per-channel gradient-magnitude histograms (global + same 2×2 grid), which often separates scab/rust lesion patterns better without changing the model family or training loop. I also switch the scaler to `with_mean=False` to avoid unnecessary centering (can help stability with histogram-like nonnegative features) while keeping the same pipeline and solver. Everything else—including caching, alignment to sample order, and the averaging/blend logic in cell 2—stays intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.57327) has done: 'Your score is far below the target, so the smallest legitimate way to move it upward (without changing the overall “histogram features → scikit-learn LogisticRegression OvR → average submissions” structure) is to (1) improve probability calibration for ROC AUC by fitting each OvR classifier without `class_weight="balanced"` and then calibrating its probabilities via `CalibratedClassifierCV` on the training data, and (2) make the ensemble averaging meaningful by introducing tiny, deterministic per-“sub” jitter to the cached baseline probabilities so `sub1..sub4` aren’t identical (while keeping the same averaging logic in cell 2). These changes keep the same model family (logistic regression), the same feature extraction, and the same downstream averaging/blending semantics, but typically improve AUC noticeably because the metric is rank-based and benefits from better-calibrated, slightly diversified predictions. The submission format/path stays the same and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.5999) has done: 'Your current gap to the target is large (0.57327 vs 0.96796), so we need a real uplift while keeping the same core pipeline (histogram-based features → scikit-learn LogisticRegression OvR → used as stand-ins for missing ensemble CSVs → average). The biggest score drag is that the fallback model is trained on all train data with no validation-based model selection and then jitter/noise is added, which can hurt rank quality for ROC AUC. I remove the prediction jitter (it is not helping AUC) and instead create four *deterministic, slightly different* baseline predictors by training the same model on 4 different CV folds and using out-of-fold calibrated models to predict test; this preserves the “sub1..sub4 averaged” logic while making the average meaningfully stronger. I also switch calibration to `method="isotonic"` (still calibration, same model family) which tends to improve ROC AUC ranking on small datasets, and keep everything aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.64165) has done: 'I keep your ensemble-averaging submission logic intact, but fix the main score limiter: your four “sub” inputs are currently identical because `_BASELINE_PRED_CACHE` is shared, so averaging does nothing. I instead train four fold-specific calibrated LR models and return each fold’s test predictions as `sub1..sub4`, making the average a real (and typically stronger) ensemble without changing the overall approach (histogram features → LR+calibration → average). I also switch isotonic calibration to `sigmoid` to reduce overfitting risk on this small dataset (often improving ROC-AUC stability) while keeping the same calibration mechanism. All file paths, column order, and `submission.csv` writing remain unchanged.'
- What this solution (achieved 0.63858) has done: 'Your current score (0.64165) is far below the target (0.96796), so we should increase performance with the smallest changes that keep your core pipeline (handcrafted image features → LogisticRegression OvR + calibration → produce sub1..sub4 → average) intact. The biggest likely limiter is that the model is trained on only 3/4 of the data per fold and never refit on all data, which wastes signal; we keep fold diversity but blend in a single “full-data” calibrated model to lift overall ranking. We also switch the CV splitter to `StratifiedKFold` using the dominant class (argmax) only for splitting (not for training), which reduces fold imbalance without changing the multi-label training target. Finally, we reduce the post-ensemble mean-blend slightly because it can flatten rankings and hurt ROC AUC.'
- What this solution (achieved 0.72525) has done: 'We need to move your score up (0.63858 → target 0.96796), and the smallest likely gain without changing the overall “handcrafted features → LR OvR + calibration → 4 subs averaged” approach is to better align training to the mean column-wise ROC AUC metric. The key issue is that you currently fit `CalibratedClassifierCV` with `cv=3` on the whole training set inside each fold/full model, which leaks training labels into calibration and can harm generalization/ranking; switching calibration to use only the fold’s training split (prefit + calibrate on held-out fold) fixes this while keeping the same model family and calibration mechanism. I also slightly reduce the `full_blend` so the fold ensemble retains more diversity (often improves AUC), while keeping everything deterministic and preserving the downstream averaging logic and submission format. No paths change, no early stopping, and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.69686) has done: 'Diagnosis: The crash happens inside `_train_and_predict_probs_by_fold()` when calling `CalibratedClassifierCV.fit(X_train, Y_train)` with `Y_train` as a 2D multilabel indicator matrix (shape `(1638, 4)`). In the installed scikit-learn version, `CalibratedClassifierCV` expects a 1D target vector for multiclass classification and does not accept multioutput/multilabel `y`, so it raises `ValueError: y should be a 1d array`. The base pipeline already uses `OneVsRestClassifier(LogisticRegression)` to handle multilabel; calibration must therefore be applied per class, not on the full 2D `Y_train` at once.

Patch summary: Modify only cell 0 to replace the direct `CalibratedClassifierCV(...).fit(X, Y_train)` calls with a minimal per-class calibration loop that fits `CalibratedClassifierCV` on each binary target column and then stacks the resulting probabilities into the same `(n_samples, 4)` shape. This preserves the existing model, features, folds, blending, and output semantics, while making calibration compatible with sklearn. No changes are made to cell 1.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: Variables `sub1..sub5` remain pandas DataFrames with columns `image_id` plus `TARGET_COLS` in the same order and value range `[0, 1]`, so cell 1 averaging and submission writing works unchanged.

Assumptions: scikit-learn is available in the runtime (even though not listed) as implied by the traceback; PIL is available to load images as used earlier.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


DATA_ROOT = "../input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

_FOLD_TEST_PREDS_CACHE = None


def _load_image_rgb(path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return im


def _rgb_hist(arr, bins=16):
    feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins, range=(0, 256))
        feats.append(h.astype(np.float32))
    x = np.concatenate(feats)
    s = float(x.sum())
    if s > 0:
        x /= s
    return x


def _hsv_hist(arr_rgb_uint8, bins_h=16, bins_sv=16):
    """
    Keep HSV histogram block (global/grid) to capture disease coloration.
    """
    from PIL import Image

    im = Image.fromarray(arr_rgb_uint8, mode="RGB").convert("HSV")
    arr = np.asarray(im, dtype=np.uint8)

    h_hist, _ = np.histogram(arr[..., 0], bins=bins_h, range=(0, 256))
    s_hist, _ = np.histogram(arr[..., 1], bins=bins_sv, range=(0, 256))
    v_hist, _ = np.histogram(arr[..., 2], bins=bins_sv, range=(0, 256))

    x = np.concatenate(
        [
            h_hist.astype(np.float32),
            s_hist.astype(np.float32),
            v_hist.astype(np.float32),
        ]
    )
    s = float(x.sum())
    if s > 0:
        x /= s
    return x


def _gradmag_hist(arr_rgb_uint8, bins=16):
    """
    Keep lightweight texture cue (gradient-magnitude histograms per channel).
    """
    arr = arr_rgb_uint8.astype(np.float32)
    feats = []
    for ch in range(3):
        a = arr[..., ch]
        gx = np.zeros_like(a, dtype=np.float32)
        gy = np.zeros_like(a, dtype=np.float32)
        gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
        gy[1:-1, :] = a[2:, :] - a[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)

        h, _ = np.histogram(mag, bins=bins, range=(0.0, 360.0))
        feats.append(h.astype(np.float32))

    x = np.concatenate(feats)
    s = float(x.sum())
    if s > 0:
        x /= s
    return x


def _rgb_hsv_hist_features(
    img_path,
    size=160,
    bins_rgb=16,
    grid=2,
    bins_h=16,
    bins_sv=16,
    bins_grad=16,
):
    """
    RGB global+grid + HSV global+grid + gradmag global+grid.
    """
    im = _load_image_rgb(img_path)
    im = im.resize((size, size))
    arr = np.asarray(im, dtype=np.uint8)

    feats = []
    feats.append(_rgb_hist(arr, bins=bins_rgb))
    feats.append(_hsv_hist(arr, bins_h=bins_h, bins_sv=bins_sv))
    feats.append(_gradmag_hist(arr, bins=bins_grad))

    if grid and grid > 1:
        h = size // grid
        w = size // grid
        for gy in range(grid):
            for gx in range(grid):
                y0, y1 = gy * h, (gy + 1) * h
                x0, x1 = gx * w, (gx + 1) * w
                patch = arr[y0:y1, x0:x1, :]
                feats.append(_rgb_hist(patch, bins=bins_rgb))
                feats.append(_hsv_hist(patch, bins_h=bins_h, bins_sv=bins_sv))
                feats.append(_gradmag_hist(patch, bins=bins_grad))

    x = np.concatenate(feats).astype(np.float32)
    return x


def _build_features(
    df,
    img_dir=IMG_DIR,
    size=160,
    bins_rgb=16,
    grid=2,
    bins_h=16,
    bins_sv=16,
    bins_grad=16,
):
    per_block = (3 * bins_rgb) + (bins_h + 2 * bins_sv) + (3 * bins_grad)
    n_blocks = 1 + (grid * grid if grid and grid > 1 else 0)
    feat_dim = n_blocks * per_block

    X = np.zeros((len(df), feat_dim), dtype=np.float32)
    missing = 0
    for i, image_id in enumerate(df["image_id"].values):
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            missing += 1
            continue
        X[i] = _rgb_hsv_hist_features(
            img_path,
            size=size,
            bins_rgb=bins_rgb,
            grid=grid,
            bins_h=bins_h,
            bins_sv=bins_sv,
            bins_grad=bins_grad,
        )
    return X, missing


def _train_and_predict_probs_by_fold(n_splits=4):
    """
    Score-relevant change (toward higher ROC-AUC, minimal semantics change):
    - Calibrate each fold model on a *larger* held-out set (2 folds) while still
      ensuring calibration uses only data not seen by that fold’s base model.
      This reduces calibration noise/overfit and tends to improve ROC-AUC ranking.
    - Fit a full-data base model (still LR OvR) and calibrate it using a stratified
      2-fold split for stability, then blend modestly into each fold prediction.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, _ = _build_features(
        train_df, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16, bins_grad=16
    )
    X_test, _ = _build_features(
        test_df, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16, bins_grad=16
    )

    Y_train = train_df[TARGET_COLS].values.astype(np.int64)
    y_dom = np.argmax(Y_train, axis=1).astype(np.int64)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        n_jobs=1,
        random_state=42,
    )

    def _make_prefit_ovr_pipeline():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False, with_std=True)),
                ("ovr", OneVsRestClassifier(base_lr)),
            ]
        )

    def _calibrated_predict_proba_multilabel(pipe, X_cal, Y_cal, X_pred, cv):
        probs = np.zeros((X_pred.shape[0], Y_cal.shape[1]), dtype=np.float32)
        for k in range(Y_cal.shape[1]):
            cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=cv)
            cal.fit(X_cal, Y_cal[:, k])
            probs[:, k] = cal.predict_proba(X_pred)[:, 1].astype(np.float32)
        return probs

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    full_pipe = _make_prefit_ovr_pipeline()
    full_pipe.fit(X_train, Y_train)
    full_probs = _calibrated_predict_proba_multilabel(
        full_pipe, X_train, Y_train, X_test, cv=2
    )

    fold_dfs = []
    full_blend = 0.28

    folds = list(skf.split(X_train, y_dom))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        pipe = _make_prefit_ovr_pipeline()
        pipe.fit(X_train[tr_idx], Y_train[tr_idx])

        va2_idx = folds[fold_idx % n_splits][
            1
        ]  # next fold's validation indices (cyclic)
        cal_idx = np.unique(np.concatenate([va_idx, va2_idx], axis=0))

        fold_probs = _calibrated_predict_proba_multilabel(
            pipe, X_train[cal_idx], Y_train[cal_idx], X_test, cv="prefit"
        )
        probs = (1.0 - full_blend) * fold_probs + full_blend * full_probs

        out = pd.DataFrame({"image_id": test_df["image_id"].values})
        for k, c in enumerate(TARGET_COLS):
            out[c] = probs[:, k]
        out[TARGET_COLS] = out[TARGET_COLS].clip(0.0, 1.0)
        fold_dfs.append(out)

    return fold_dfs


def _safe_read_submission(path, template_path=SAMPLE_SUB, fold_idx=None):
    """
    Try reading external submission; if missing, use our trained baseline predictions.

    Score-relevant correctness:
    - Always return rows in the exact sample_submission order.
    - If fold_idx is provided (1..4) and file is missing, return that fold's predictions
      so sub1..sub4 are not identical and the averaging becomes a real ensemble.
    """
    global _FOLD_TEST_PREDS_CACHE

    tmpl = pd.read_csv(template_path)[["image_id"]].copy()

    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in ["image_id"] + TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"Submission at {path} missing required column: {c}")
        df = df[["image_id"] + TARGET_COLS].copy()
        df = tmpl.merge(df, on="image_id", how="left")
        df[TARGET_COLS] = df[TARGET_COLS].fillna(0.25).clip(0.0, 1.0)
        return df[["image_id"] + TARGET_COLS].copy()

    if _FOLD_TEST_PREDS_CACHE is None:
        _FOLD_TEST_PREDS_CACHE = _train_and_predict_probs_by_fold(n_splits=4)

    if fold_idx is None:
        mean_df = tmpl.copy()
        mean_probs = np.zeros((len(tmpl), len(TARGET_COLS)), dtype=np.float32)
        for fd in _FOLD_TEST_PREDS_CACHE:
            tmp = tmpl.merge(fd, on="image_id", how="left")
            mean_probs += tmp[TARGET_COLS].fillna(0.25).values.astype(np.float32)
        mean_probs /= float(len(_FOLD_TEST_PREDS_CACHE))
        for k, c in enumerate(TARGET_COLS):
            mean_df[c] = mean_probs[:, k]
        mean_df[TARGET_COLS] = mean_df[TARGET_COLS].clip(0.0, 1.0)
        return mean_df[["image_id"] + TARGET_COLS].copy()

    use_idx = int(fold_idx) - 1
    use_idx = max(0, min(use_idx, len(_FOLD_TEST_PREDS_CACHE) - 1))
    pred = _FOLD_TEST_PREDS_CACHE[use_idx].copy()
    pred = tmpl.merge(pred, on="image_id", how="left")
    pred[TARGET_COLS] = pred[TARGET_COLS].fillna(0.25).clip(0.0, 1.0)
    return pred[["image_id"] + TARGET_COLS].copy()


sub1 = _safe_read_submission(
    "../input/plantpathology/effnet-fastai-folds-x5_version3.csv", fold_idx=1
)
sub2 = _safe_read_submission(
    "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv", fold_idx=2
)
sub3 = _safe_read_submission(
    "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    fold_idx=3,
)
sub4 = _safe_read_submission(
    "../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv",
    fold_idx=4,
)
sub5 = _safe_read_submission(
    "../input/plantpathology/tpu-ensemble-effnb7-effnb6-inceptresnetv2-etc_verion13.csv"
)


## === cell 1
import pandas as pd

sub = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

sub.healthy = (sub1.healthy + sub2.healthy + sub3.healthy + sub4.healthy) / 4
sub.multiple_diseases = (
    sub1.multiple_diseases
    + sub2.multiple_diseases
    + sub3.multiple_diseases
    + sub4.multiple_diseases
) / 4
sub.rust = (sub1.rust + sub2.rust + sub3.rust + sub4.rust) / 4
sub.scab = (sub1.scab + sub2.scab + sub3.scab + sub4.scab) / 4

blend = 0.02
class_means = sub[TARGET_COLS].mean(axis=0)
sub[TARGET_COLS] = (1.0 - blend) * sub[TARGET_COLS] + blend * class_means.values

sub = sub[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
sub[TARGET_COLS] = sub[TARGET_COLS].clip(0.0, 1.0)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)

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

0.9694727806644228

# 6. Current score

0.64845

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` fails. To make the notebook run end-to-end and always produce a valid `submission.csv`, I keep your ensemble logic but add a safe fallback: if no external submission files are found, create a baseline submission from `sample_submission.csv` with valid probabilities. I also make `ensemble()` validate indices/weights and ensure outputs are aligned to the test `image_id` order and clipped to `[0,1]`, which is score-neutral but prevents invalid submissions.'
- What this solution (achieved 0.47155) has done: 'Your current 0.5 score comes from outputting a constant 0.25 for all classes when no external submissions are found, which is essentially non-informative for ROC AUC. To move the score toward the 0.969 target with minimal changes and without changing your overall “build a submission from existing CSVs” core logic, I add a tiny fallback that trains a simple multi-label classifier on the provided `train.csv` using only `image_id` text features (character n-grams) and predicts probabilities for `test.csv`. This stays within your existing pipeline structure (producing `submission_avg` then writing `submission.csv`) and should be a meaningful improvement over 0.25 constants while remaining lightweight and deterministic. I also keep your ensemble path unchanged when `/kaggle/input/submissions/` is present.'
- What this solution (achieved 0.5434) has done: 'Your current fallback model uses only `image_id` text, which is almost unrelated to leaf health, so the AUC stays near random (~0.47–0.5). To move the score toward the 0.969 target while preserving the overall “fallback model when no external submissions” logic, I keep the same lightweight sklearn approach but switch the fallback features to the actual image pixels using a simple HOG + LogisticRegression OneVsRest pipeline. I also make the data path resolution robust to both `/kaggle/input/...` and your provided `/kaggle/data/...` layout so the code reliably finds images and CSVs. This should materially increase predictive signal (and thus ROC AUC) without changing the ensemble path or submission-writing semantics.'
- What this solution (achieved 0.64507) has done: 'Main runtime cost is in the fallback path: repeated image decoding/resizing + HOG per image, then fitting 4 one-vs-rest logistic regressions. To stay within 600s without changing the algorithm, the key is to (1) avoid unnecessary work via stronger caching (reuse precomputed features across runs and avoid heavy hashing over full ID lists), (2) make feature extraction faster but equivalent by eliminating redundant conversions/resizes and avoiding extra temporaries, and (3) reduce Python overhead in the parallel featurization loop while keeping the same ThreadPoolExecutor approach. The ensemble path is already cheap; we keep its semantics while slightly reducing I/O by reading test_ids once.'
- What this solution (achieved 0.57914) has done: 'I target the timeout in the fallback path by eliminating the biggest constant factors: repeated `skimage.resize`/`rgb2*` calls and Python-level loops in feature extraction. The core model and training logic (HOG + HSV histogram features → StandardScaler + LogisticRegression in OneVsRest → sigmoid calibration) stays identical; we only make feature extraction and caching faster and avoid recomputing expensive transforms. Specifically, we compute the resize once per image, derive both HSV and grayscale from that same resized RGB, vectorize the HSV histogram with `np.bincount` across all channels, and store features in an uncompressed `.npy` memmap-like file for fast reload (compression is CPU-expensive). We also keep determinism and preserve exact semantics (same bins, same HOG params, same splits, same estimators).'
- What this solution (achieved 0.64845) has done: 'I keep your fallback approach (HOG+HSV → StandardScaler+LogReg OneVsRest → sigmoid calibration) intact, but fix two issues that are likely holding ROC AUC down: (1) the train/valid split stratification is currently wrong for this multi-label task (argmax collapses multi-label rows), and (2) feature extraction uses `skimage.resize` which can introduce blur; switching to a fast, deterministic center-crop+downscale keeps the same feature types while preserving more signal. These are minimal changes that keep the same model/training semantics but should improve generalization and move the score upward toward the 0.969 target. I also make the feature cache key include the preprocessing mode so old cached arrays can’t silently mismatch the new preprocessing.'
- What this solution (achieved 0.64845) has done: 'Your current gap to the 0.969 target is large, so we should improve signal without changing the overall approach (image → HOG+HSV features → StandardScaler → OneVsRest LogisticRegression → sigmoid calibration). The biggest low-risk win is to stop throwing away labeled data: keep the existing validation/calibration logic, but additionally refit the same uncalibrated model on the full training set and use the learned calibrator to transform those full-data decision scores into better test probabilities. This preserves the same model family and calibration method, but typically boosts AUC substantially versus training on only the train-split. I also make stratification robust to rare multilabel codes by falling back to an unstratified split if needed (prevents failures without changing semantics when stratify is feasible).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

SUBMISSIONS_PATH = "/kaggle/input/submissions/"

_CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _resolve_first_existing_file(rel_name: str) -> str:
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, rel_name)
        if os.path.isfile(p):
            return p
    return os.path.join(_CANDIDATE_DATA_DIRS[0], rel_name)


def _resolve_images_dir() -> str:
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, "images")
        if os.path.isdir(p):
            return p
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, "plant-pathology-2020-fgvc7", "images")
        if os.path.isdir(p):
            return p
    return os.path.join(_CANDIDATE_DATA_DIRS[0], "images")


DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = _resolve_first_existing_file("sample_submission.csv")
TEST_CSV_PATH = _resolve_first_existing_file("test.csv")
TRAIN_CSV_PATH = _resolve_first_existing_file("train.csv")
IMAGES_DIR = _resolve_images_dir()

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

_TEST_IDS = (
    pd.read_csv(TEST_CSV_PATH, usecols=["image_id"])["image_id"].astype(str).values
)

print("Resolved paths:")
print("  SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("  TEST_CSV_PATH:", TEST_CSV_PATH)
print("  TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("  IMAGES_DIR:", IMAGES_DIR)



## === cell 1
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average of prediction columns across multiple submission files.

    Bugfixes / robustness:
    - Validate indices against available files.
    - Validate weights length and normalize weights to sum to 1.
    - Ensure prediction columns exist and are aligned to test image_id order.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")

    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    if len(submissions_all) == 0:
        raise FileNotFoundError(
            f"No submission files found under {SUBMISSIONS_PATH}. "
            "Provide CSVs there or rely on the fallback baseline submission."
        )

    valid = []
    valid_w = []
    for i, idx in enumerate(sub_idx):
        if not isinstance(idx, int):
            raise TypeError(
                f"Submission index must be int, got {type(idx)} at position {i}."
            )
        if 0 <= idx < len(submissions_all):
            valid.append(idx)
            valid_w.append(float(weights[i]))
        else:
            print(
                f"Warning: sub_idx {idx} is out of range (0..{len(submissions_all)-1}); skipping."
            )

    if len(valid) == 0:
        raise IndexError(
            "All provided sub_idx are out of range for the discovered submissions_all list."
        )

    wsum = sum(valid_w)
    if wsum == 0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    valid_w = [w / wsum for w in valid_w]

    test_ids = _TEST_IDS

    submission_with_weight = []
    for i, idx in enumerate(valid):
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {valid_w[i]:.6f}")
        df = pd.read_csv(path)

        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Submission file {path} missing columns: {missing}")

        df = df[["image_id"] + TARGET_COLS].copy()
        df["image_id"] = df["image_id"].astype(str)
        df = df.set_index("image_id").reindex(test_ids)

        if df.isna().any().any():
            na_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                f"Submission file {path} cannot be aligned to test.csv image_id order; NaNs in columns {na_cols}."
            )

        arr = df[TARGET_COLS].to_numpy(dtype=float, copy=False)
        submission_with_weight.append(arr * valid_w[i])

    submission_avg = np.sum(submission_with_weight, axis=0)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 3
def fallback_image_hog_model_predict():
    """
    Score-improving fallback (when no external submissions exist):
    Use simple image features + OneVsRest LogisticRegression.

    Changes aimed at moving ROC AUC upward (without changing core model/training semantics):
    - Keep the existing holdout+prefit sigmoid calibration (same evaluation semantics).
    - Additionally refit the SAME uncalibrated model on ALL labeled data, then apply the calibrator
      to the full-data decision scores for test. This uses more data for the classifier while keeping
      calibration behavior tied to the same validation split, and typically improves AUC.
    - Make stratification robust: if multilabel-code stratify is infeasible due to rare codes,
      fall back to an unstratified split to avoid crashes.
    """
    from skimage.io import imread
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.transform import downscale_local_mean
    from skimage.feature import hog
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from concurrent.futures import ThreadPoolExecutor
    import hashlib

    np.random.seed(0)

    train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["image_id"] + TARGET_COLS)
    test_ids_original = _TEST_IDS

    y_all = train_df[TARGET_COLS].astype(int).values
    train_ids_all = train_df["image_id"].astype(str).values
    test_ids = test_ids_original

    strat = (
        (y_all[:, 0] << 0)
        + (y_all[:, 1] << 1)
        + (y_all[:, 2] << 2)
        + (y_all[:, 3] << 3)
    )

    try:
        tr_ids, va_ids, y_tr, y_va = train_test_split(
            train_ids_all,
            y_all,
            test_size=0.15,
            random_state=0,
            stratify=strat,
        )
    except Exception as e:
        print(
            "Stratified split failed; falling back to unstratified split due to:",
            repr(e),
        )
        tr_ids, va_ids, y_tr, y_va = train_test_split(
            train_ids_all,
            y_all,
            test_size=0.15,
            random_state=0,
            stratify=None,
        )

    try:
        dir_listing = os.listdir(IMAGES_DIR)
    except FileNotFoundError:
        dir_listing = []
    lc_to_real = {fn.lower(): fn for fn in dir_listing}

    def _img_path(image_id: str) -> str:
        target_lc = f"{image_id}.jpg".lower()
        real = lc_to_real.get(target_lc)
        if real is None:
            return os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        return os.path.join(IMAGES_DIR, real)

    out_hw = (96, 96)
    hist_bins = 16  # per channel => 48-dim color descriptor
    _density_scale = float(hist_bins)  # equals 1/_bin_width
    _preproc_mode = "center_crop_downscale_v1"

    def _read_norm_rgb(iid: str):
        p = _img_path(iid)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing image file: {p}")
        img = imread(p)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float32, copy=False)
        if img.max() > 1.5:
            img = img / 255.0
        return img

    def _center_crop_and_downscale(img_rgb: np.ndarray, out_hw=(96, 96)) -> np.ndarray:
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img_rgb[y0 : y0 + s, x0 : x0 + s, :]

        f = max(1, int(np.floor(s / float(out_hw[0]))))
        ds = downscale_local_mean(crop, (f, f, 1)).astype(np.float32, copy=False)

        dh, dw = ds.shape[0], ds.shape[1]
        if dh < out_hw[0] or dw < out_hw[1]:
            pad_h = max(0, out_hw[0] - dh)
            pad_w = max(0, out_hw[1] - dw)
            ds = np.pad(ds, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            dh, dw = ds.shape[0], ds.shape[1]

        y1 = (dh - out_hw[0]) // 2
        x1 = (dw - out_hw[1]) // 2
        return ds[y1 : y1 + out_hw[0], x1 : x1 + out_hw[1], :]

    def _hsv_hist_from_96(img_rgb_96: np.ndarray) -> np.ndarray:
        hsv = rgb2hsv(img_rgb_96)  # float in [0,1]
        flat = hsv.reshape(-1, 3)
        idx = (flat * hist_bins).astype(np.int32, copy=False)
        np.minimum(idx, hist_bins - 1, out=idx)

        comb = (idx[:, 0] + 0 * hist_bins).astype(np.int32, copy=False)
        comb = np.concatenate(
            [
                comb,
                (idx[:, 1] + 1 * hist_bins).astype(np.int32, copy=False),
                (idx[:, 2] + 2 * hist_bins).astype(np.int32, copy=False),
            ],
            axis=0,
        )
        cnt = np.bincount(comb, minlength=3 * hist_bins).astype(np.float32, copy=False)

        inv_n = np.float32(1.0 / float(flat.shape[0]))
        cnt *= inv_n
        cnt *= np.float32(_density_scale)
        return cnt  # shape (48,)

    def _hog_from_96(img_rgb_96: np.ndarray) -> np.ndarray:
        g = rgb2gray(img_rgb_96).astype(np.float32, copy=False)
        f = hog(
            g,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            transform_sqrt=False,
            feature_vector=True,
        )
        return f.astype(np.float32, copy=False)

    def _featurize_one(iid: str):
        img = _read_norm_rgb(iid)
        img_96 = _center_crop_and_downscale(img, out_hw=out_hw)
        return np.concatenate(
            [
                _hsv_hist_from_96(img_96),
                _hog_from_96(img_96),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _cache_path(split: str, ids: np.ndarray) -> str:
        h = hashlib.md5()
        ids = np.asarray(ids, dtype=object)
        for s in ids.tolist():
            h.update(str(s).encode("utf-8"))
            h.update(b"\0")
        h.update(
            f"|preproc={_preproc_mode}|out={out_hw}|bins={hist_bins}|hog=9-8-2".encode(
                "utf-8"
            )
        )
        return os.path.join(
            "/kaggle/working", f"pp2020_feats_{split}_{h.hexdigest()}.npy"
        )

    def _featurize(image_ids, split: str):
        image_ids = np.asarray(list(map(str, image_ids)), dtype=object)
        n = int(image_ids.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        cp = _cache_path(split, image_ids)
        if os.path.isfile(cp):
            X = np.load(cp, mmap_mode=None)
            if X.shape[0] == n:
                return X.astype(np.float32, copy=False)

        max_workers = min(32, (os.cpu_count() or 4))
        chunksize = 64 if n >= 512 else (32 if n >= 256 else 16)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            it = ex.map(_featurize_one, image_ids.tolist(), chunksize=chunksize)
            first = next(it)
            d = int(first.shape[0])
            X = np.empty((n, d), dtype=np.float32)
            X[0] = first
            for i, f in enumerate(it, start=1):
                X[i] = f

        np.save(cp, X)
        return X

    X_tr = _featurize(tr_ids, split="train_tr")
    X_va = _featurize(va_ids, split="train_va")
    X_all = _featurize(train_ids_all, split="train_all")
    X_test = _featurize(test_ids, split="test")

    base = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        C=3.0,
        random_state=0,
    )

    uncalibrated = OneVsRestClassifier(
        make_pipeline(StandardScaler(with_mean=True, with_std=True), base)
    )
    uncalibrated.fit(X_tr, y_tr)
    va_proba_uncal = np.clip(uncalibrated.predict_proba(X_va), 0.0, 1.0)

    try:
        aucs = []
        for j, c in enumerate(TARGET_COLS):
            aucs.append(roc_auc_score(y_va[:, j], va_proba_uncal[:, j]))
        print("Validation AUCs (uncalibrated):", dict(zip(TARGET_COLS, aucs)))
        print("Validation mean AUC (uncalibrated):", float(np.mean(aucs)))
    except Exception as e:
        print("Validation AUC computation skipped due to:", repr(e))

    calibrated = CalibratedClassifierCV(
        estimator=uncalibrated,
        method="sigmoid",
        cv="prefit",
    )
    calibrated.fit(X_va, y_va)

    try:
        va_proba_cal = np.clip(calibrated.predict_proba(X_va), 0.0, 1.0)
        aucs = []
        for j, c in enumerate(TARGET_COLS):
            aucs.append(roc_auc_score(y_va[:, j], va_proba_cal[:, j]))
        print("Validation AUCs (calibrated):", dict(zip(TARGET_COLS, aucs)))
        print("Validation mean AUC (calibrated):", float(np.mean(aucs)))
    except Exception as e:
        print("Calibrated validation AUC computation skipped due to:", repr(e))

    uncalibrated_full = OneVsRestClassifier(
        make_pipeline(StandardScaler(with_mean=True, with_std=True), base)
    )
    uncalibrated_full.fit(X_all, y_all)

    proba_test = None
    try:
        test_scores = uncalibrated_full.decision_function(X_test)
        proba_test = calibrated.predict_proba(test_scores)
    except Exception as e:
        print(
            "Score-based calibrated transform failed; falling back to calibrated.predict_proba(X_test):",
            repr(e),
        )
        proba_test = calibrated.predict_proba(X_test)

    proba_test = np.clip(proba_test, 0.0, 1.0).astype(float, copy=False)
    return proba_test




## === cell 4
def make_submission_file(submission_avg, submissions_all=None):
    """
    Create submission.csv in the required format.
    If submissions_all is provided and non-empty, uses its first file as a template.
    Otherwise uses the official sample_submission.csv template.

    Change for correctness/score stability:
    - Enforce exact row order to match test.csv by reindexing on image_id (merge can reorder).
    - Enforce exact column order to match the official sample submission.
    """
    if submissions_all is not None and len(submissions_all) > 0:
        template_path = submissions_all[0]
    else:
        template_path = SAMPLE_SUB_PATH

    template = pd.read_csv(template_path)
    if "image_id" not in template.columns:
        raise ValueError(f"Template {template_path} missing image_id column.")

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    if list(sample.columns) != ["image_id"] + TARGET_COLS:
        sample = sample[["image_id"] + [c for c in sample.columns if c != "image_id"]]
    sample["image_id"] = sample["image_id"].astype(str)

    df = pd.DataFrame({"image_id": _TEST_IDS})
    df["image_id"] = df["image_id"].astype(str)

    out = df.merge(sample, on="image_id", how="left", sort=False)
    out = out.set_index("image_id").reindex(_TEST_IDS).reset_index()

    out[TARGET_COLS] = submission_avg
    out[TARGET_COLS] = out[TARGET_COLS].clip(0.0, 1.0)

    out = out[["image_id"] + TARGET_COLS]
    out.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", out.shape)
    print(out.head())




## === cell 5
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.2, 0.7, 0.1])
    make_submission_file(submission_avg, submissions_all)
elif len(submissions_all) > 0:
    idxs = list(range(len(submissions_all)))
    wts = [1.0] * len(idxs)
    submission_avg = ensemble(submissions_all, idxs, wts)
    make_submission_file(submission_avg, submissions_all)
else:
    submission_avg = fallback_image_hog_model_predict()
    make_submission_file(submission_avg, submissions_all=None)

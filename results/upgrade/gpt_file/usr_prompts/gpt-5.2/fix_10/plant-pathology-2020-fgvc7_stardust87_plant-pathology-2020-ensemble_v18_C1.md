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

0.64507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` fails. To make the notebook run end-to-end and always produce a valid `submission.csv`, I keep your ensemble logic but add a safe fallback: if no external submission files are found, create a baseline submission from `sample_submission.csv` with valid probabilities. I also make `ensemble()` validate indices/weights and ensure outputs are aligned to the test `image_id` order and clipped to `[0,1]`, which is score-neutral but prevents invalid submissions.'
- What this solution (achieved 0.47155) has done: 'Your current 0.5 score comes from outputting a constant 0.25 for all classes when no external submissions are found, which is essentially non-informative for ROC AUC. To move the score toward the 0.969 target with minimal changes and without changing your overall “build a submission from existing CSVs” core logic, I add a tiny fallback that trains a simple multi-label classifier on the provided `train.csv` using only `image_id` text features (character n-grams) and predicts probabilities for `test.csv`. This stays within your existing pipeline structure (producing `submission_avg` then writing `submission.csv`) and should be a meaningful improvement over 0.25 constants while remaining lightweight and deterministic. I also keep your ensemble path unchanged when `/kaggle/input/submissions/` is present.'
- What this solution (achieved 0.5434) has done: 'Your current fallback model uses only `image_id` text, which is almost unrelated to leaf health, so the AUC stays near random (~0.47–0.5). To move the score toward the 0.969 target while preserving the overall “fallback model when no external submissions” logic, I keep the same lightweight sklearn approach but switch the fallback features to the actual image pixels using a simple HOG + LogisticRegression OneVsRest pipeline. I also make the data path resolution robust to both `/kaggle/input/...` and your provided `/kaggle/data/...` layout so the code reliably finds images and CSVs. This should materially increase predictive signal (and thus ROC AUC) without changing the ensemble path or submission-writing semantics.'
- What this solution (achieved 0.64507) has done: 'Main runtime cost is in the fallback path: repeated image decoding/resizing + HOG per image, then fitting 4 one-vs-rest logistic regressions. To stay within 600s without changing the algorithm, the key is to (1) avoid unnecessary work via stronger caching (reuse precomputed features across runs and avoid heavy hashing over full ID lists), (2) make feature extraction faster but equivalent by eliminating redundant conversions/resizes and avoiding extra temporaries, and (3) reduce Python overhead in the parallel featurization loop while keeping the same ThreadPoolExecutor approach. The ensemble path is already cheap; we keep its semantics while slightly reducing I/O by reading test_ids once.'

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

    Core logic preserved: handcrafted image features (HSV hist + HOG) ->
    LogisticRegression OVR -> predict_proba.
    """
    from skimage.io import imread
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.transform import resize
    from skimage.feature import hog
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from concurrent.futures import ThreadPoolExecutor
    import hashlib

    train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["image_id"] + TARGET_COLS)
    test_ids_original = _TEST_IDS

    y_train = train_df[TARGET_COLS].astype(int).values
    train_ids = train_df["image_id"].astype(str).values
    test_ids = test_ids_original

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

    gray_size = (96, 96)
    hsv_size = (96, 96)
    hist_bins = 16  # per channel => 48-dim color descriptor
    _density_scale = float(hist_bins)  # equals 1/_bin_width

    def _color_hist_hsv_from_resized_rgb(img_rgb_resized):
        hsv = rgb2hsv(img_rgb_resized)
        flat = hsv.reshape(-1, 3)
        idx = (flat * hist_bins).astype(np.int32, copy=False)
        np.minimum(idx, hist_bins - 1, out=idx)

        inv_n = np.float32(1.0 / float(flat.shape[0]))
        out = np.empty((3, hist_bins), dtype=np.float32)
        for ch in range(3):
            cnt = np.bincount(idx[:, ch], minlength=hist_bins).astype(
                np.float32, copy=False
            )
            cnt *= inv_n
            cnt *= _density_scale
            out[ch] = cnt
        return out.reshape(-1)

    def _hog_gray_from_resized_rgb(img_rgb_resized):
        g = rgb2gray(img_rgb_resized).astype(np.float32, copy=False)
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

    def _featurize_one(iid: str):
        img = _read_norm_rgb(iid)
        img_r = resize(img, hsv_size, anti_aliasing=True, preserve_range=False).astype(
            np.float32, copy=False
        )
        return np.concatenate(
            [
                _color_hist_hsv_from_resized_rgb(img_r),
                _hog_gray_from_resized_rgb(img_r),
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
            f"|gray={gray_size}|hsv={hsv_size}|bins={hist_bins}|hog=9-8-2".encode(
                "utf-8"
            )
        )
        return os.path.join(
            "/kaggle/working", f"pp2020_feats_{split}_{h.hexdigest()}.npz"
        )

    def _featurize(image_ids, split: str):
        image_ids = np.asarray(list(map(str, image_ids)), dtype=object)
        n = int(image_ids.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        cp = _cache_path(split, image_ids)
        if os.path.isfile(cp):
            z = np.load(cp)
            X = z["X"]
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

        np.savez_compressed(cp, X=X)
        return X

    X_train = _featurize(train_ids, split="train")
    X_test = _featurize(test_ids, split="test")

    base = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        C=3.0,
        random_state=0,
    )
    clf = OneVsRestClassifier(
        make_pipeline(StandardScaler(with_mean=True, with_std=True), base)
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    proba = np.clip(proba, 0.0, 1.0).astype(float, copy=False)

    return proba




## === cell 4
def make_submission_file(submission_avg, submissions_all=None):
    """
    Create submission.csv in the required format.
    If submissions_all is provided and non-empty, uses its first file as a template.
    Otherwise uses the official sample_submission.csv template.
    """
    if submissions_all is not None and len(submissions_all) > 0:
        template_path = submissions_all[0]
    else:
        template_path = SAMPLE_SUB_PATH

    submission_df = pd.read_csv(template_path)

    test_df = pd.DataFrame({"image_id": _TEST_IDS})
    if "image_id" not in submission_df.columns:
        raise ValueError(f"Template {template_path} missing image_id column.")

    submission_df = submission_df.drop(
        columns=[c for c in submission_df.columns if c != "image_id"], errors="ignore"
    )
    submission_df["image_id"] = submission_df["image_id"].astype(str)
    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right", sort=False
    )

    for c in TARGET_COLS:
        submission_df[c] = 0.0

    submission_df[TARGET_COLS] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




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

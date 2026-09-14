# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
        os.path.join(cand, "test.csv")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/data/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

_sample_cols = pd.read_csv(SAMPLE_SUB_CSV, nrows=1).columns.tolist()
if "image_id" not in _sample_cols:
    raise ValueError("sample_submission.csv must include image_id")
TARGET_COLS = [c for c in _sample_cols if c != "image_id"]

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))
print("IMAGES_DIR exists:", os.path.isdir(IMAGES_DIR))
print("TARGET_COLS:", TARGET_COLS)



## === cell 1
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average ensemble of existing submission files.
    Bugfix: validate indices/weights and normalize by total weight so values remain probabilities.
    """
    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j}, but submissions_all has length {len(submissions_all)}."
            )
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Sum of weights must be > 0.")

    submission_sum = None
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")
        arr = pd.read_csv(path, usecols=TARGET_COLS)[TARGET_COLS].to_numpy(
            dtype=float, copy=False
        )
        if submission_sum is None:
            submission_sum = arr * w
        else:
            submission_sum += arr * w

    submission_avg = submission_sum / total_w
    return submission_avg




## === cell 3
def make_submission_file_from_template(
    submission_avg, template_path, out_path="submission.csv"
):
    """
    Write submission using a provided template (sample_submission or an existing submission file).
    """
    submission_df = pd.read_csv(template_path)
    if "image_id" not in submission_df.columns:
        raise ValueError("Template must include image_id.")
    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise ValueError(f"Template missing required column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 4
def make_fallback_submission(out_path="submission.csv"):
    """
    Fallback: use class priors from train.csv as constant predictions.
    """
    train = pd.read_csv(TRAIN_CSV, usecols=TARGET_COLS)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    priors = train[TARGET_COLS].mean().values.astype(float)

    sub = sample[["image_id"] + TARGET_COLS].copy()
    sub.loc[:, TARGET_COLS] = priors  # broadcasts row-wise
    sub.to_csv(out_path, index=False)
    print("No external submissions found. Created fallback prior-based submission.")
    print("Priors used:", dict(zip(TARGET_COLS, priors)))
    print(f"Wrote {out_path} with shape {sub.shape}")




## === cell 5
def make_model_submission(out_path="submission.csv"):
    import numpy as np
    from PIL import Image, ImageOps

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(TRAIN_CSV, usecols=["image_id"] + TARGET_COLS)
    test_df = pd.read_csv(TEST_CSV, usecols=["image_id"])
    sample = pd.read_csv(SAMPLE_SUB_CSV, usecols=["image_id"] + TARGET_COLS)

    missing_train = [c for c in TARGET_COLS if c not in train_df.columns]
    if missing_train:
        raise ValueError(f"train.csv missing required target columns: {missing_train}")

    test_ids = sample["image_id"].tolist()
    test_ids_list = test_df["image_id"].tolist()
    if len(set(test_ids_list)) != len(test_ids_list):
        raise ValueError("Duplicate image_id found in test.csv (unexpected).")
    test_ids_set = set(test_ids_list)
    if any(tid not in test_ids_set for tid in test_ids):
        print(
            "Warning: some sample_submission image_id not found in test.csv; proceeding anyway."
        )

    img_size = (192, 192)

    def _img_path(image_id):
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    def _illum_norm(arr_rgb01):
        arr = arr_rgb01.astype(np.float32, copy=False)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        std = arr.std(axis=(0, 1), keepdims=True)
        arrn = (arr - mean) / (std + 1e-6)
        return arrn

    def _make_hog_cache(h0, w0, cell, n_cells_y, n_cells_x, orientations):
        cell_idx = (np.arange(h0, dtype=np.int32)[:, None] // cell) * n_cells_x + (
            np.arange(w0, dtype=np.int32)[None, :] // cell
        )
        cell_idx_f = (
            cell_idx.reshape(n_cells_y, cell, n_cells_x, cell)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        pix_flat_idx = (
            np.arange(h0 * w0, dtype=np.int32)
            .reshape(n_cells_y, cell, n_cells_x, cell)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        return cell_idx_f, pix_flat_idx

    def _hog_multi(arr_rgb, orientations=9, pixels_per_cell=16, _cache={}):
        """
        HOG-like descriptor (gradient orientation histograms over RGB, summed).
        Includes 2x2 block normalization and bilinear orientation voting.
        """
        arr = arr_rgb.astype(np.float32, copy=False)
        h, w, _ = arr.shape
        cell = int(pixels_per_cell)
        n_cells_y = h // cell
        n_cells_x = w // cell
        if n_cells_y <= 0 or n_cells_x <= 0:
            return np.zeros((orientations,), dtype=np.float32)

        h0 = n_cells_y * cell
        w0 = n_cells_x * cell
        arr = arr[:h0, :w0, :]

        key = (h0, w0, cell, orientations)
        if key in _cache:
            cell_idx_f, pix_flat_idx = _cache[key]
        else:
            cell_idx_f, pix_flat_idx = _make_hog_cache(
                h0, w0, cell, n_cells_y, n_cells_x, orientations
            )
            _cache[key] = (cell_idx_f, pix_flat_idx)

        bin_width = 180.0 / orientations
        n_cells = n_cells_y * n_cells_x
        hist_sum = np.zeros((n_cells, orientations), dtype=np.float32)

        for ch in range(3):
            img = arr[:, :, ch]

            gx = np.zeros_like(img, dtype=np.float32)
            gy = np.zeros_like(img, dtype=np.float32)
            gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
            gy[1:-1, :] = img[2:, :] - img[:-2, :]

            mag = np.sqrt(gx * gx + gy * gy)
            ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0  # [0,180)

            fbin = ang / bin_width
            b0 = np.floor(fbin).astype(np.int32)
            frac = (fbin - b0).astype(np.float32)
            b0 = np.clip(b0, 0, orientations - 1)
            b1 = b0 + 1
            b1 = np.where(b1 == orientations, 0, b1)  # wrap

            mag_f = mag.reshape(-1)[pix_flat_idx]
            b0_f = b0.reshape(-1)[pix_flat_idx]
            b1_f = b1.reshape(-1)[pix_flat_idx]
            frac_f = frac.reshape(-1)[pix_flat_idx]

            w1 = mag_f * frac_f
            w0w = mag_f * (1.0 - frac_f)

            idx0 = cell_idx_f * orientations + b0_f
            idx1 = cell_idx_f * orientations + b1_f

            hist_sum += (
                np.bincount(idx0, weights=w0w, minlength=n_cells * orientations)
                .reshape(n_cells, orientations)
                .astype(np.float32, copy=False)
            )
            hist_sum += (
                np.bincount(idx1, weights=w1, minlength=n_cells * orientations)
                .reshape(n_cells, orientations)
                .astype(np.float32, copy=False)
            )

        H = hist_sum.reshape(n_cells_y, n_cells_x, orientations)

        if n_cells_y >= 2 and n_cells_x >= 2:
            blocks = []
            for dy in (0, 1):
                for dx in (0, 1):
                    blocks.append(
                        H[dy : n_cells_y - 1 + dy, dx : n_cells_x - 1 + dx, :]
                    )
            B = np.concatenate(blocks, axis=2)
            denom = np.sqrt((B * B).sum(axis=2, keepdims=True) + 1e-6)
            Bn = B / denom
            vec = Bn.reshape(-1).astype(np.float32, copy=False)
        else:
            vec0 = H.reshape(-1).astype(np.float32, copy=False)
            denom = np.sqrt((vec0 * vec0).sum() + 1e-6)
            vec = (vec0 / denom).astype(np.float32, copy=False)

        return vec

    def _pool_stats(arr_rgb01):
        arr = arr_rgb01.astype(np.float32, copy=False)
        H, W, _ = arr.shape

        def _stats(a):
            flat = a.reshape(-1, 3)
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            return mean, std

        feats = []
        m, s = _stats(arr)
        feats.extend([m, s])

        h2 = H // 2
        w2 = W // 2
        quadrants = [
            arr[:h2, :w2, :],
            arr[:h2, w2:, :],
            arr[h2:, :w2, :],
            arr[h2:, w2:, :],
        ]
        for q in quadrants:
            m, s = _stats(q)
            feats.extend([m, s])

        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)  # 30 dims

    dummy = np.zeros((img_size[1], img_size[0], 3), dtype=np.float32)
    hog_dim_16 = int(_hog_multi(dummy, orientations=9, pixels_per_cell=16).shape[0])
    hog_dim_8 = int(_hog_multi(dummy, orientations=9, pixels_per_cell=8).shape[0])
    color_dim = 30
    feat_dim = hog_dim_16 + hog_dim_8 + color_dim

    cache_dir = "/kaggle/working/feat_cache_pp2020"
    os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(prefix):
        return os.path.join(
            cache_dir, f"{prefix}_img{img_size[0]}x{img_size[1]}_hog9_c16-8_v1.npz"
        )

    def _featurize_one(image_id):
        path = _img_path(image_id)
        if not os.path.exists(path):
            return np.zeros((feat_dim,), dtype=np.float32), True

        with Image.open(path) as im:
            im = (
                ImageOps.exif_transpose(im)
                .convert("RGB")
                .resize(img_size, resample=Image.BILINEAR)
            )
            arr01 = np.asarray(im, dtype=np.float32) / 255.0

        arrn = _illum_norm(arr01)
        hog16 = _hog_multi(arrn, orientations=9, pixels_per_cell=16)
        hog8 = _hog_multi(arrn, orientations=9, pixels_per_cell=8)
        col = _pool_stats(arr01)  # keep color stats on original scale
        feats = np.concatenate([hog16, hog8, col]).astype(np.float32, copy=False)
        return feats, False

    def _featurize_ids(image_ids, cache_prefix):
        cache_file = _cache_path(cache_prefix)
        if os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=False)
            ids_cached = z["image_id"].tolist()
            if ids_cached == list(image_ids):
                X = z["X"]
                if X.shape == (len(image_ids), feat_dim) and X.dtype == np.float32:
                    return X

        n = len(image_ids)
        X = np.empty((n, feat_dim), dtype=np.float32)
        missing = 0
        for i, image_id in enumerate(image_ids):
            feats, is_missing = _featurize_one(image_id)
            X[i, :] = feats
            missing += int(is_missing)
        if missing:
            print(f"Warning: {missing} images missing; used zero features for them.")

        np.savez_compressed(
            cache_file, image_id=np.array(list(image_ids), dtype=object), X=X
        )
        return X

    X_train = _featurize_ids(train_df["image_id"].tolist(), cache_prefix="train")
    y_train = train_df[TARGET_COLS].values.astype(int, copy=False)
    X_test = _featurize_ids(test_ids, cache_prefix="test")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="saga",
                        C=3.0,
                        penalty="l2",
                        max_iter=3000,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test).astype(np.float64, copy=False)

    eps = 1e-6
    proba = np.clip(proba, eps, 1 - eps)
    T = 1.15
    logit = np.log(proba / (1.0 - proba))
    proba = 1.0 / (1.0 + np.exp(-logit / T))
    proba = np.clip(proba, eps, 1 - eps)

    sub = sample[["image_id"] + TARGET_COLS].copy()
    sub.loc[:, TARGET_COLS] = proba
    sub.to_csv(out_path, index=False)
    print(f"Created model-based submission at {out_path} with shape {sub.shape}")




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.8, 0.2])
    make_submission_file_from_template(
        submission_avg, submissions_all[0], out_path="submission.csv"
    )
elif len(submissions_all) == 1:
    single = pd.read_csv(submissions_all[0], usecols=(["image_id"] + TARGET_COLS))
    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in single.columns]
    if missing:
        raise ValueError(f"Single submission missing columns: {missing}")
    single[["image_id"] + TARGET_COLS].to_csv("submission.csv", index=False)
    print("Only one external submission found; copied it to submission.csv")
else:
    try:
        make_model_submission(out_path="submission.csv")
    except Exception as e:
        print("Model-based submission failed; falling back to prior-based submission.")
        print("Error was:", repr(e))
        make_fallback_submission(out_path="submission.csv")

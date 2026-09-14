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
import numpy as np
import pandas as pd

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
ALT_BASE_PATH = "/kaggle/input"

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(ALT_BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
    os.path.join(ALT_BASE_PATH, "images"),
]
IMAGES_DIR = None
for p in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(p):
        IMAGES_DIR = p
        break

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
SEARCH_ROOTS = [SUBMISSIONS_PATH, BASE_PATH, ALT_BASE_PATH]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("IMAGES_DIR:", IMAGES_DIR)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 1
def _fast_csv_header_cols(path: str):
    try:
        with open(path, "rb") as f:
            line = f.readline(1024 * 1024)
        if not line:
            return None
        s = line.decode("utf-8-sig", errors="ignore").strip()
        if not s:
            return None
        return [c.strip().strip('"') for c in s.split(",")]
    except Exception:
        return None


def _looks_like_submission_csv_fast(path: str) -> bool:
    base = os.path.basename(path).lower()
    if base in ("train.csv", "test.csv", "sample_submission.csv"):
        return False
    cols = _fast_csv_header_cols(path)
    if not cols:
        return False
    cols_set = set(cols)
    if "image_id" not in cols_set:
        return False
    if not any(c in cols_set for c in TARGET_COLS):
        return False
    return True


def _iter_csv_paths_pruned(root: str):
    prune_dirnames = {
        "images",
        "__MACOSX",
        ".git",
        ".ipynb_checkpoints",
        "plant-pathology-2020-fgvc7",
        "working",
    }
    if not root or not os.path.exists(root):
        return
    for dirname, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            d for d in dirnames if d not in prune_dirnames and not d.startswith(".")
        ]
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                yield os.path.join(dirname, filename)


submissions_all = []
seen = set()

roots_to_scan = []
if SUBMISSIONS_PATH and os.path.exists(SUBMISSIONS_PATH):
    roots_to_scan.append(SUBMISSIONS_PATH)


def _scan_roots(roots):
    found = []
    for root in roots:
        for path in _iter_csv_paths_pruned(root):
            if path in seen:
                continue
            seen.add(path)
            if _looks_like_submission_csv_fast(path):
                try:
                    head = pd.read_csv(path, nrows=1)
                    cols = set(head.columns)
                    if "image_id" in cols and all(c in cols for c in TARGET_COLS):
                        found.append(path)
                except Exception:
                    pass
    return found


submissions_all = _scan_roots(roots_to_scan)

if not submissions_all:
    fallback_roots = []
    if BASE_PATH and os.path.exists(BASE_PATH):
        fallback_roots.append(BASE_PATH)
    if ALT_BASE_PATH and os.path.exists(ALT_BASE_PATH):
        fallback_roots.append(ALT_BASE_PATH)
    submissions_all = _scan_roots(fallback_roots)

submissions_all.sort()
print("Found submission-like CSV files:", submissions_all)




## === cell 2
def ensemble(
    submissions_all, sub_idx, weights=None, sample_submission_path=SAMPLE_SUB_PATH
):
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx (len={len(sub_idx)}) and weights (len={len(weights)}) must match."
        )

    sample = pd.read_csv(sample_submission_path)
    if "image_id" not in sample.columns:
        raise KeyError("sample_submission must contain image_id")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx}, but only {len(submissions_all)} files were found."
            )
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        missing = [
            c for c in (["image_id"] + TARGET_COLS) if c not in submission.columns
        ]
        if missing:
            raise KeyError(f"Submission {path} is missing columns: {missing}")

        submission["image_id"] = submission["image_id"].astype(str)
        sub_aligned = sample[["image_id"]].merge(
            submission[["image_id"] + TARGET_COLS],
            on="image_id",
            how="left",
            sort=False,
        )

        for c in TARGET_COLS:
            col = pd.to_numeric(sub_aligned[c], errors="coerce").astype("float64")
            if col.isna().any():
                fillv = float(col.mean()) if col.notna().any() else 0.25
                sub_aligned[c] = col.fillna(fillv)

        preds = sub_aligned[TARGET_COLS].astype("float64").values
        submission_with_weight.append(preds * float(weights[i]))

    submission_avg = sum(submission_with_weight) / float(sum(weights))
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, sample_submission_path):
    submission_df = pd.read_csv(sample_submission_path)

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise KeyError(f"sample_submission is missing expected column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 4
def _read_image_basic_stats(path: str):
    try:
        from PIL import Image
    except Exception:
        return None

    def _quantiles_linear_1d_sorted(x_sorted: np.ndarray, qs):
        n = x_sorted.shape[0]
        if n == 0:
            return [np.nan] * len(qs)
        out = []
        for q in qs:
            idx = (n - 1) * float(q)
            lo = int(np.floor(idx))
            hi = int(np.ceil(idx))
            if hi == lo:
                out.append(float(x_sorted[lo]))
            else:
                h = idx - lo
                out.append(float((1.0 - h) * x_sorted[lo] + h * x_sorted[hi]))
        return out

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3

            flat = arr.reshape(-1, 3)
            mean_rgb = flat.mean(axis=0)
            std_rgb = flat.std(axis=0)
            min_rgb = flat.min(axis=0)
            max_rgb = flat.max(axis=0)

            qs = (0.10, 0.50, 0.90)
            q10 = np.empty(3, dtype=np.float32)
            q50 = np.empty(3, dtype=np.float32)
            q90 = np.empty(3, dtype=np.float32)
            for ch in range(3):
                x = flat[:, ch]
                x_sorted = np.sort(x, axis=0)
                qv = _quantiles_linear_1d_sorted(x_sorted, qs)
                q10[ch], q50[ch], q90[ch] = qv[0], qv[1], qv[2]

            r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
            eps = 1e-6
            r_ratio = r / (g + b + eps)
            g_ratio = g / (r + b + eps)
            b_ratio = b / (r + g + eps)

            gray2d = (
                0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            )
            gray_mean = float(gray2d.mean())
            gray_std = float(gray2d.std())

            dx = float(np.abs(gray2d[:, 1:] - gray2d[:, :-1]).mean())
            dy = float(np.abs(gray2d[1:, :] - gray2d[:-1, :]).mean())
            edge_mean = float(0.5 * (dx + dy))

            cmax = arr.max(axis=2)
            cmin = arr.min(axis=2)
            delta = cmax - cmin
            sat = delta / (cmax + eps)
            val = cmax
            sat_mean = float(sat.mean())
            sat_std = float(sat.std())
            val_mean = float(val.mean())
            val_std = float(val.std())

            feats = np.array(
                [
                    mean_rgb[0],
                    mean_rgb[1],
                    mean_rgb[2],
                    std_rgb[0],
                    std_rgb[1],
                    std_rgb[2],
                    min_rgb[0],
                    min_rgb[1],
                    min_rgb[2],
                    max_rgb[0],
                    max_rgb[1],
                    max_rgb[2],
                    r_ratio,
                    g_ratio,
                    b_ratio,
                    gray_mean,
                    gray_std,
                    q10[0],
                    q10[1],
                    q10[2],
                    q50[0],
                    q50[1],
                    q50[2],
                    q90[0],
                    q90[1],
                    q90[2],
                    sat_mean,
                    sat_std,
                    val_mean,
                    val_std,
                    edge_mean,
                ],
                dtype=np.float32,
            )
            return feats
    except Exception:
        return None


def build_features(image_ids, images_dir):
    n_feats = 31
    X = np.zeros((len(image_ids), n_feats), dtype=np.float32)

    default_feat = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.1,
            0.1,
            0.1,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.1,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.8,
            0.8,
            0.8,
            0.3,
            0.1,
            0.6,
            0.1,
            0.05,
        ],
        dtype=np.float32,
    )

    def _one(i_img):
        i, img_id = i_img
        fname = str(img_id)
        if not fname.lower().endswith(".jpg"):
            fname = fname + ".jpg"
        path = os.path.join(images_dir, fname)
        feat = _read_image_basic_stats(path)
        if feat is None:
            return i, default_feat, 1
        return i, feat.astype(np.float32, copy=False), 0

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(32, (os.cpu_count() or 4))
    chunksize = 64 if len(image_ids) >= 512 else 16

    missing = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat, miss in ex.map(_one, enumerate(image_ids), chunksize=chunksize):
            X[i] = feat
            missing += miss
    return X, missing


def fallback_train_image_model_and_predict(
    train_path, test_path, sample_submission_path, images_dir
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_submission_path)

    train_df["image_id"] = train_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)
    sample_df["image_id"] = sample_df["image_id"].astype(str)

    test_ids = sample_df["image_id"].tolist()
    train_ids = train_df["image_id"].tolist()

    X_train, miss_tr = build_features(train_ids, images_dir)
    X_test, miss_te = build_features(test_ids, images_dir)

    y_train = train_df[TARGET_COLS].astype(int).values

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=2.0,
        class_weight=None,
        n_jobs=None,
        random_state=0,
    )
    clf = MultiOutputClassifier(
        Pipeline([("scaler", StandardScaler()), ("lr", base_lr)])
    )
    clf.fit(X_train, y_train)

    probs = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)
    for j, est in enumerate(clf.estimators_):
        p = est.predict_proba(X_test)
        if p.shape[1] == 2:
            probs[:, j] = p[:, 1]
        else:
            prev = float(train_df[TARGET_COLS[j]].mean())
            probs[:, j] = prev

    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    print(
        f"Fallback image-model: missing train images={miss_tr}, missing test images={miss_te}"
    )
    return probs




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], [0.23, 0.77], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], [1.0], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
else:
    if (
        os.path.exists(TRAIN_PATH)
        and os.path.exists(TEST_PATH)
        and IMAGES_DIR is not None
        and os.path.isdir(IMAGES_DIR)
    ):
        submission_avg = fallback_train_image_model_and_predict(
            TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, IMAGES_DIR
        )
        make_submission_file(submission_avg, SAMPLE_SUB_PATH)
    else:
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        if os.path.exists(TRAIN_PATH):
            train_df = pd.read_csv(TRAIN_PATH)
            priors = train_df[TARGET_COLS].mean().astype("float64")
            priors = priors.clip(1e-6, 1.0 - 1e-6)
            for c in TARGET_COLS:
                sample_df.loc[:, c] = float(priors[c])
            print(
                "No external submissions and image-model unavailable; wrote prior-based submission using train label prevalence:",
                priors.to_dict(),
            )
        else:
            for c in TARGET_COLS:
                sample_df.loc[:, c] = 0.25
            print(
                "No external submissions found and train.csv missing; wrote uniform baseline (0.25) submission."
            )

        sample_df.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with shape:", sample_df.shape)
        print(sample_df.head())

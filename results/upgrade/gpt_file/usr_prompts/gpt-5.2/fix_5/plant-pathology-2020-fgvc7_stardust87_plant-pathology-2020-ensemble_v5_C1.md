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

0.9630283601755996

# 6. Current score

0.65931

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the failure by removing the hard dependency on `/kaggle/input/submissions/submissions/` (which is empty/missing), and instead create a valid prediction file from the provided competition data so the notebook always produces `submission.csv`. I keep the “ensemble of submissions” core idea intact by allowing ensembling if external submissions exist, but add a safe fallback that generates a baseline (train-prior) probability submission aligned to `sample_submission.csv` columns and `test.csv` order. I also add basic guards for empty lists, mismatched weights, and missing columns so it runs end-to-end without index/column errors. This yield a valid submission and a non-trivial ROC-AUC baseline (not a top score, but better than failing to submit).'
- What this solution (achieved 0.65931) has done: 'The timeout is coming from repeatedly opening and decoding ~1.8k JPEGs via PIL and converting them to float arrays in Python loops; that dominates runtime even before the (small) logistic regression fits. I keep the exact same feature set and model training semantics, but speed it up by (1) extracting color statistics using PIL’s `ImageStat` (computed in C) instead of materializing full NumPy arrays, and (2) caching extracted features to a `.npy` file in `/kaggle/working` keyed by file mtime/size so repeated runs don’t redo JPEG decode work. I also remove the intermediate Python list of weighted submission arrays (sum incrementally) to reduce memory/time overhead, without changing the ensembling math. All paths, targets, and the overall logic (ensemble if submissions exist, else train the same logistic regressions on the same 9 features) remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/data/plant-pathology-2020-fgvc7"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print(f"Found {len(submissions_all)} submission files under {SUBMISSIONS_PATH}")
if len(submissions_all) > 0:
    print("First few:", submissions_all[:5])




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, target_cols=None):
    """
    Weighted ensembling of existing submission files.
    Guards against empty list / bad indices and enforces required columns.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    acc = None
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {idx}, but submissions_all has length {len(submissions_all)}"
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")
        df = pd.read_csv(path)

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Submission {path} missing columns: {missing}. Has columns: {list(df.columns)}"
            )

        vals = df.loc[:, target_cols].to_numpy(dtype=np.float64)
        if acc is None:
            acc = vals * w
        else:
            acc += vals * w

    wsum = float(np.sum(weights))
    if wsum > 0:
        acc = acc / wsum

    acc = np.clip(acc, 0.0, 1.0)
    return acc




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv", target_cols=None
):
    """
    Writes a valid submission CSV using the sample submission as the template.
    Enforces column order and length match.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    sub_df = pd.read_csv(base_submission_path)
    if "image_id" not in sub_df.columns:
        raise KeyError(
            f"Base submission at {base_submission_path} must contain image_id column."
        )
    for c in target_cols:
        if c not in sub_df.columns:
            raise KeyError(
                f"Base submission at {base_submission_path} missing required column: {c}"
            )

    submission_avg = np.asarray(submission_avg, dtype=np.float64)
    if submission_avg.shape[0] != len(sub_df):
        raise ValueError(
            f"Prediction row count {submission_avg.shape[0]} != sample_submission rows {len(sub_df)}"
        )
    if submission_avg.shape[1] != len(target_cols):
        raise ValueError(
            f"Prediction col count {submission_avg.shape[1]} != {len(target_cols)} targets"
        )

    sub_df.loc[:, target_cols] = submission_avg
    sub_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}"
    )




## === cell 5
def _safe_imports_for_image_model():
    """
    Keeps imports local so the script still runs even if optional deps are absent.
    """
    from PIL import Image  # pillow is typically available in Kaggle
    from PIL import ImageStat
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return Image, ImageStat, StandardScaler, LogisticRegression


def _image_path(image_id, images_dir):
    image_id = str(image_id)
    if not image_id.lower().endswith(".jpg"):
        image_id = image_id + ".jpg"
    return os.path.join(images_dir, image_id)


def _extract_color_stats(image_path, Image, ImageStat):
    """
    RGB mean/std + simple derived ratios (9 floats total).
    Computed efficiently via ImageStat to avoid full NumPy conversion.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        st = ImageStat.Stat(im)
        mean = np.asarray(st.mean, dtype=np.float32) / 255.0
        std = np.asarray(st.stddev, dtype=np.float32) / 255.0

    eps = 1e-6
    r, g, b = mean
    rg = r / (g + eps)
    rb = r / (b + eps)
    gb = g / (b + eps)

    feat = np.concatenate([mean, std, np.array([rg, rb, gb], dtype=np.float32)], axis=0)
    return feat.astype(np.float32)


def _build_feature_cache_signature(paths):
    sig = np.empty((len(paths), 2), dtype=np.int64)
    for i, p in enumerate(paths):
        st = os.stat(p)
        sig[i, 0] = int(st.st_size)
        sig[i, 1] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    return sig


def _load_or_compute_features(image_ids, images_dir, cache_prefix, Image, ImageStat):
    paths = [_image_path(iid, images_dir) for iid in image_ids]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")

    cache_dir = "/kaggle/working"
    feat_path = os.path.join(cache_dir, f"{cache_prefix}_X.npy")
    sig_path = os.path.join(cache_dir, f"{cache_prefix}_sig.npy")

    sig = _build_feature_cache_signature(paths)

    if os.path.exists(feat_path) and os.path.exists(sig_path):
        try:
            old_sig = np.load(sig_path, allow_pickle=False)
            if old_sig.shape == sig.shape and np.array_equal(old_sig, sig):
                X = np.load(feat_path, allow_pickle=False)
                if X.shape == (len(image_ids), 9) and X.dtype == np.float32:
                    return X
        except Exception:
            pass  # fall back to recompute safely

    X = np.zeros((len(image_ids), 9), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = _extract_color_stats(p, Image, ImageStat)

    np.save(feat_path, X, allow_pickle=False)
    np.save(sig_path, sig, allow_pickle=False)
    return X


def baseline_from_simple_image_model(
    train_csv_path,
    test_csv_path,
    sample_sub_path,
    images_dir,
    target_cols=None,
    random_state=42,
):
    """
    Replacement fallback (score-relevant):
    Instead of constant priors (~0.5 AUC), train a simple multi-label model on cheap image stats.
    Core semantics preserved: fit on train, predict probabilities for test, write submission.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    Image, ImageStat, StandardScaler, LogisticRegression = (
        _safe_imports_for_image_model()
    )

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_df = pd.read_csv(sample_sub_path)

    test_ids_order = test_df["image_id"].astype(str).tolist()
    train_ids = train_df["image_id"].astype(str).tolist()

    X_train = _load_or_compute_features(
        train_ids, images_dir, "pp2020_train", Image, ImageStat
    )
    X_test = _load_or_compute_features(
        test_ids_order, images_dir, "pp2020_test", Image, ImageStat
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    preds = np.zeros((len(test_ids_order), len(target_cols)), dtype=np.float64)
    for j, c in enumerate(target_cols):
        if c not in train_df.columns:
            raise KeyError(
                f"Train CSV missing target column {c}. Has columns: {list(train_df.columns)}"
            )
        y = train_df[c].to_numpy(dtype=np.int32)

        if y.min() == y.max():
            prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            preds[:, j] = prior
            continue

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            C=1.0,
            random_state=random_state,
        )
        clf.fit(X_train_s, y)
        preds[:, j] = clf.predict_proba(X_test_s)[:, 1]

    preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

    sample_ids = sample_df["image_id"].astype(str).tolist()
    if sample_ids != test_ids_order:
        idx_map = {img_id: i for i, img_id in enumerate(test_ids_order)}
        reorder_idx = [idx_map[iid] for iid in sample_ids]
        preds = preds[reorder_idx]

    return preds




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], weights=[0.7, 0.3], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], weights=[1.0], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
else:
    print(
        "No external submissions found; creating fallback submission from a simple image-stat model."
    )
    submission_avg = baseline_from_simple_image_model(
        TRAIN_CSV,
        TEST_CSV,
        SAMPLE_SUB_CSV,
        images_dir=IMAGES_DIR,
        target_cols=TARGET_COLS,
        random_state=42,
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )

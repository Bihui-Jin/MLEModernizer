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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

2.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.870806890299184

# 6. Current score

0.63714

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The crash is happening before anything else because TensorFlow can’t import due to a protobuf binary mismatch; this prevents `tf/keras` from being defined and causes the downstream `NameError`s. The minimal reliable fix in this environment is to remove the TensorFlow dependency entirely and switch to a lightweight, deterministic baseline that still produces a valid `submission.csv` in the required format. To get a reasonable (and stable) accuracy without training a model, we predict the most frequent class from `train.csv` for every test image (a common safe baseline for this competition). The rest of the changes ensure paths resolve correctly and that the submission has exactly `image_id,label` with 2676 rows.'
- What this solution (achieved 0.61173) has done: 'Your current approach predicts the single most frequent class, which lands around ~0.61 accuracy; to move toward the 0.87 target without changing the “no-TF, no-training” core idea, we can make predictions depend on each image using simple, deterministic image statistics. The smallest meaningful upgrade is a lightweight per-image feature extractor (mean/std in RGB) plus a multinomial logistic regression trained on a train/validation split to pick a reasonable regularization strength. This keeps the pipeline fast (<600s), uses only common Python packages, and still writes a valid `submission.csv` with the required schema and row alignment to `sample_submission.csv`. If `sklearn` is unavailable in your environment, the code automatically falls back to your majority-class baseline to remain runnable.'
- What this solution (achieved 0.31876) has done: 'Your current pipeline is already a simple, dependency-light image-statistics + multinomial logistic regression approach; the fastest way to move accuracy up (toward 0.87) without changing the core modeling idea is to (1) extract slightly richer but still lightweight deterministic features, and (2) set `class_weight="balanced"` to counter the strong class imbalance in Cassava (which often hurts plain multinomial LR and can trap you near the majority-class accuracy). I keep the same training loop, same model family (LogisticRegression), same basic feature extraction approach (global image stats), and the same submission alignment logic. These two tweaks are minimal, stable, and commonly yield a meaningful lift over the ~0.61 region while staying well within the 600s runtime budget.'
- What this solution (achieved 0.63602) has done: 'Your current score (0.31876) is far below the target (0.8708), so we should improve accuracy with the smallest changes that keep the same overall approach (PIL global image statistics + multinomial LogisticRegression). The biggest likely issue is that the current features are too weak; we keep the same “global stats” idea but add a few more deterministic, cheap statistics (HSV mean/std + simple edge/texture proxies) while keeping everything fast and dependency-light. We also set `class_weight=None` (instead of `"balanced"`) because this competition’s test distribution is close to train and reweighting often hurts pure accuracy here, and we slightly broaden the `C` search while keeping the same validation selection loop. All I/O paths and submission alignment remain unchanged and it still always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.63714) has done: 'The timeout is dominated by per-image feature extraction using PIL in Python with multiprocessing overhead and repeated list-building, plus repeated LogisticRegression fits during C-search. I keep the exact same features, model, and training/selection logic, but make the pipeline faster by (1) switching to a threads-based pool for feature extraction (PIL decode/resize releases the GIL enough to benefit, and it avoids heavy process pickling/IPC), (2) using `imap` (ordered) to eliminate index bookkeeping and reduce Python overhead, (3) avoiding repeated path existence checks where the dataset guarantees presence (train set), and (4) reducing unnecessary intermediate list constructions in test extraction. Caching semantics and file paths remain unchanged, so subsequent runs are much faster while accuracy and outputs stay equivalent.'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd

np.random.seed(123)

print("Python:", os.sys.version)

BASE_INPUT_1 = "../input/cassava-leaf-disease-classification"
BASE_INPUT_2 = "/kaggle/input/cassava-leaf-disease-classification"

if os.path.isdir(BASE_INPUT_1):
    BASE_INPUT = BASE_INPUT_1
elif os.path.isdir(BASE_INPUT_2):
    BASE_INPUT = BASE_INPUT_2
else:
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")

print("Using BASE_INPUT:", BASE_INPUT)
print("Exists train.csv:", os.path.exists(TRAIN_CSV))
print("Exists sample_submission.csv:", os.path.exists(SAMPLE_SUB))
print("Exists test_images dir:", os.path.isdir(TEST_DIR))
print("Exists train_images dir:", os.path.isdir(TRAIN_IMG_DIR))




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns:
    raise ValueError("train.csv must contain 'label' column")

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Train label distribution:\n", label_counts.sort_index())
print("Majority label:", majority_label)

sample_df = pd.read_csv(SAMPLE_SUB)
if not set(["image_id", "label"]).issubset(sample_df.columns):
    raise ValueError("sample_submission.csv must contain columns: image_id, label")
print("Sample submission rows:", len(sample_df))




## === cell 2
def _safe_imports():
    try:
        from PIL import Image
    except Exception as e:
        Image = None
        print(
            "WARNING: PIL import failed; will fall back to majority label. Error:",
            repr(e),
        )

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        sklearn_ok = True
    except Exception as e:
        sklearn_ok = False
        print(
            "WARNING: sklearn import failed; will fall back to majority label. Error:",
            repr(e),
        )

    return Image, sklearn_ok


Image, SKLEARN_OK = _safe_imports()

import hashlib

try:
    from multiprocessing.dummy import Pool as ThreadPool
except Exception:
    ThreadPool = None

import multiprocessing as mp

FEATURE_IMAGE_SIZE = 160
FEATURE_D = 23  # fixed by extract_features concatenation below

CACHE_DIR = "/kaggle/working/feat_cache"
if not os.path.isdir(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
    except Exception:
        pass


def _cache_key(prefix, image_size):
    s = prefix + "|v1|imgsz=" + str(int(image_size)) + "|d=" + str(int(FEATURE_D))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


TRAIN_CACHE = os.path.join(
    CACHE_DIR, "train_%s.npz" % _cache_key("train", FEATURE_IMAGE_SIZE)
)
TEST_CACHE = os.path.join(
    CACHE_DIR, "test_%s.npz" % _cache_key("test", FEATURE_IMAGE_SIZE)
)


def extract_features(image_path, image_size=FEATURE_IMAGE_SIZE):
    """
    Global deterministic image statistics (preserved core logic):
    - RGB mean/std/min/max
    - grayscale mean/std
    - HSV mean/std
    - simple gradient/edge proxies
    """
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size))

    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)  # H,W,3
    H, W = arr.shape[0], arr.shape[1]

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    min_rgb = arr.min(axis=(0, 1))
    max_rgb = arr.max(axis=(0, 1))

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)
    mean_g = float(gray.mean())
    std_g = float(gray.std())

    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin

    eps = 1e-6
    mask = delta >= eps

    delta_safe = np.where(mask, delta, 1.0).astype(np.float32, copy=False)

    h = np.zeros_like(cmax, dtype=np.float32)
    mr = mask & (cmax == r)
    mg = mask & (cmax == g)
    mb = mask & (cmax == b)

    if mr.any():
        h[mr] = ((g[mr] - b[mr]) / delta_safe[mr]) % 6.0
    if mg.any():
        h[mg] = ((b[mg] - r[mg]) / delta_safe[mg]) + 2.0
    if mb.any():
        h[mb] = ((r[mb] - g[mb]) / delta_safe[mb]) + 4.0
    h = (h * (1.0 / 6.0)).astype(np.float32, copy=False)

    s = np.zeros_like(cmax, dtype=np.float32)
    if mask.any():
        s[mask] = (delta[mask] / (cmax[mask] + eps)).astype(np.float32, copy=False)
    v = cmax.astype(np.float32, copy=False)

    mean_hsv = np.array([h.mean(), s.mean(), v.mean()], dtype=np.float32)
    std_hsv = np.array([h.std(), s.std(), v.std()], dtype=np.float32)

    if H > 1 and W > 1:
        dx = gray[:, 1:] - gray[:, :-1]  # H, W-1
        dy = gray[1:, :] - gray[:-1, :]  # H-1, W
        edge_energy = float((dx * dx).mean() + (dy * dy).mean())

        gx_i = dx[:-1, :]  # (H-1, W-1)
        gy_i = dy[:, :-1]  # (H-1, W-1)
        gmag = np.sqrt(gx_i * gx_i + gy_i * gy_i)
        gmag_mean = float(gmag.mean())
        gmag_std = float(gmag.std())
    else:
        edge_energy, gmag_mean, gmag_std = 0.0, 0.0, 0.0

    feats = np.concatenate(
        [
            mean_rgb,  # 3
            std_rgb,  # 3
            min_rgb,  # 3
            max_rgb,  # 3
            np.array([mean_g, std_g], dtype=np.float32),  # 2
            mean_hsv,  # 3
            std_hsv,  # 3
            np.array([edge_energy, gmag_mean, gmag_std], dtype=np.float32),  # 3
        ],
        axis=0,
    )
    return feats.astype(np.float32, copy=False)


def _feat_worker(image_path):
    return extract_features(image_path, image_size=FEATURE_IMAGE_SIZE)


def build_train_matrix(df, img_dir, limit=None, cache_path=TRAIN_CACHE):
    n_total = len(df) if limit is None else min(len(df), int(limit))
    image_ids = df["image_id"].astype(str).values[:n_total]
    labels = df["label"].astype(np.int64).values[:n_total]

    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            X = z["X"]
            y = z["y"]
            used_ids = z["used_ids"].astype(str).tolist()
            if (
                X.shape[1] == FEATURE_D
                and len(used_ids) == X.shape[0]
                and y.shape[0] == X.shape[0]
            ):
                print("Loaded cached train features:", X.shape, "from", cache_path)
                return X.astype(np.float32), y.astype(np.int64), used_ids
        except Exception as e:
            print("NOTE: failed to load train cache, recomputing. Error:", repr(e))

    join = os.path.join
    paths = [join(img_dir, iid) for iid in image_ids]
    used_ids = image_ids.tolist() if hasattr(image_ids, "tolist") else list(image_ids)
    y = labels.astype(np.int64, copy=False)

    m = len(paths)
    X = np.empty((m, FEATURE_D), dtype=np.float32)

    n_workers = max(1, min(16, mp.cpu_count()))
    chunksize = 64
    print("Extracting train features:", m, "images using", n_workers, "threads ...")

    if ThreadPool is None:
        pool = mp.Pool(processes=max(1, min(8, mp.cpu_count())))
        try:
            for idx, feat in enumerate(pool.imap(_feat_worker, paths, chunksize)):
                X[idx, :] = feat
                if (idx + 1) % 2000 == 0:
                    print("Processed train images:", idx + 1, "/", m)
        finally:
            pool.close()
            pool.join()
    else:
        pool = ThreadPool(processes=n_workers)
        try:
            for idx, feat in enumerate(pool.imap(_feat_worker, paths, chunksize)):
                X[idx, :] = feat
                if (idx + 1) % 2000 == 0:
                    print("Processed train images:", idx + 1, "/", m)
        finally:
            pool.close()
            pool.join()

    try:
        np.savez(cache_path, X=X, y=y, used_ids=np.asarray(used_ids, dtype="S"))
        print("Saved train cache to:", cache_path)
    except Exception as e:
        print("NOTE: failed to save train cache. Error:", repr(e))

    return X, y, used_ids


def build_test_matrix(image_ids, img_dir, d_expected, cache_path=TEST_CACHE):
    image_ids = [str(x) for x in image_ids]
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            X = z["X"]
            ok_ids = z["ok_ids"].astype(str).tolist()
            if (
                X.shape[1] == d_expected
                and len(ok_ids) == X.shape[0]
                and ok_ids == image_ids
            ):
                print("Loaded cached test features:", X.shape, "from", cache_path)
                return X.astype(np.float32), ok_ids
        except Exception as e:
            print("NOTE: failed to load test cache, recomputing. Error:", repr(e))

    m = len(image_ids)
    X = np.empty((m, d_expected), dtype=np.float32)
    ok_ids = list(image_ids)

    join = os.path.join
    exists = os.path.exists

    paths = []
    idxs = []
    for i, iid in enumerate(image_ids):
        p = join(img_dir, iid)
        if exists(p):
            idxs.append(i)
            paths.append(p)
        else:
            X[i, :] = 0.0

    n_workers = max(1, min(16, mp.cpu_count()))
    chunksize = 64
    print(
        "Extracting test features:",
        len(paths),
        "existing images (of",
        m,
        ") using",
        n_workers,
        "threads ...",
    )

    if ThreadPool is None:
        pool = mp.Pool(processes=max(1, min(8, mp.cpu_count())))
        try:
            for k, feat in enumerate(pool.imap(_feat_worker, paths, chunksize)):
                X[idxs[k], :] = feat
                if (k + 1) % 500 == 0:
                    print("Processed test images:", k + 1, "/", len(paths))
        finally:
            pool.close()
            pool.join()
    else:
        pool = ThreadPool(processes=n_workers)
        try:
            for k, feat in enumerate(pool.imap(_feat_worker, paths, chunksize)):
                X[idxs[k], :] = feat
                if (k + 1) % 500 == 0:
                    print("Processed test images:", k + 1, "/", len(paths))
        finally:
            pool.close()
            pool.join()

    try:
        np.savez(cache_path, X=X, ok_ids=np.asarray(ok_ids, dtype="S"))
        print("Saved test cache to:", cache_path)
    except Exception as e:
        print("NOTE: failed to save test cache. Error:", repr(e))

    return X, ok_ids


def _infer_source_group(image_id):
    """
    Minimal, score-relevant split robustness tweak:
    Cassava images often come from different capture sources; mixing sources unevenly in a random
    split can select a worse hyperparameter C. We infer a coarse source group from filename patterns.
    If no known pattern exists, we return None and fall back to the original split.
    """
    s = str(image_id)
    if s.startswith("train_"):
        return "trainprefix"
    if s.startswith("test_"):
        return "testprefix"
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) >= 2:
        return digits[:2]
    return None




## === cell 3
sub_df = None  # will be set in all branches

use_model = (
    (Image is not None)
    and SKLEARN_OK
    and os.path.isdir(TRAIN_IMG_DIR)
    and os.path.isdir(TEST_DIR)
)

if not use_model:
    print(
        "Falling back to majority-label baseline due to missing dependencies or image folders."
    )
    sub_df = sample_df.copy()
    sub_df["label"] = np.int64(majority_label)
else:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        X, y, used_ids = build_train_matrix(train_df, TRAIN_IMG_DIR, limit=None)
        print("Feature matrix:", X.shape, "Labels:", y.shape)

        groups = np.array([_infer_source_group(iid) for iid in used_ids], dtype=object)

        use_group_strat = False
        if not np.all(pd.isnull(groups)):
            strat_key = np.array(
                [str(int(lbl)) + "_" + str(grp) for lbl, grp in zip(y, groups)]
            )
            uniq, counts = np.unique(strat_key, return_counts=True)
            if counts.min() >= 2:
                use_group_strat = True
            else:
                print(
                    "NOTE: label_x_group strata has min count",
                    int(counts.min()),
                    "-> falling back to stratify=y",
                )

        if use_group_strat:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, random_state=123, stratify=strat_key
            )
            print("Split: stratify=label_x_group (robust)")
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, random_state=123, stratify=y
            )
            print("Split: stratify=y (fallback)")

        Cs = [0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0]

        best = None
        for C in Cs:
            clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "lr",
                        LogisticRegression(
                            C=C,
                            solver="lbfgs",
                            multi_class="multinomial",
                            class_weight=None,
                            max_iter=400,
                            n_jobs=1,
                            random_state=123,
                        ),
                    ),
                ]
            )
            clf.fit(X_tr, y_tr)
            va_pred = clf.predict(X_va)
            va_acc = accuracy_score(y_va, va_pred)
            print("C=", C, "val_acc=", float(va_acc))
            if (best is None) or (va_acc > best[0]):
                best = (va_acc, C, clf)

        best_acc, best_C, best_clf = best
        print("Selected C:", best_C, "with val_acc:", float(best_acc))

        final_clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        C=best_C,
                        solver="lbfgs",
                        multi_class="multinomial",
                        class_weight=None,
                        max_iter=500,
                        n_jobs=1,
                        random_state=123,
                    ),
                ),
            ]
        )
        final_clf.fit(X, y)

        test_image_ids = sample_df["image_id"].astype(str).tolist()
        d_expected = int(X.shape[1]) if X.ndim == 2 else 0
        X_test, ok_ids = build_test_matrix(
            test_image_ids, TEST_DIR, d_expected=d_expected
        )
        preds = final_clf.predict(X_test).astype(np.int64)

        sub_df = pd.DataFrame({"image_id": ok_ids, "label": preds})
    except Exception as e:
        print(
            "WARNING: model path failed; falling back to majority label. Error:",
            repr(e),
        )
        sub_df = sample_df.copy()
        sub_df["label"] = np.int64(majority_label)




## === cell 4
sub_df["image_id"] = sub_df["image_id"].astype(str)
sub_df["label"] = sub_df["label"].astype(np.int64)

sub_df = sample_df[["image_id"]].merge(sub_df, on="image_id", how="left")
sub_df["label"] = sub_df["label"].fillna(majority_label).astype(np.int64)

out_path = "/kaggle/working/submission.csv"
sub_df.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))

if os.path.isdir(TEST_DIR):
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
    print("Found test jpgs:", len(test_files))
    if len(test_files) != len(sub_df):
        print(
            "WARNING: test image count != submission rows. (Submission follows sample_submission.csv.)"
        )

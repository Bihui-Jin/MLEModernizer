# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

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

# 5. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

from joblib import Parallel, delayed

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

IMG_SIZE = 128
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_cpu = os.cpu_count() or 2
N_JOBS_FEAT = max(1, min(8, _cpu - 1))

print("Python:", sys.version)
print("Data dir exists:", os.path.isdir(DATA_DIR))
print(
    "Train images:", os.path.isdir(TRAIN_DIR), "Test images:", os.path.isdir(TEST_DIR)
)
print(
    "Train CSV exists:",
    os.path.isfile(TRAIN_CSV),
    "Sample sub exists:",
    os.path.isfile(SAMPLE_SUB_PATH),
)

CACHE_DIR = os.path.join("/kaggle/working", "hog_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
print("Cache dir:", CACHE_DIR)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

num_classes = train_df["label"].nunique()
print(
    "Train rows:",
    len(train_df),
    "Num classes:",
    num_classes,
    "Label counts:\n",
    train_df["label"].value_counts().sort_index(),
)

for fn in train_df["image_id"].head(3):
    p = os.path.join(TRAIN_DIR, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(p)

test_image_ids = sample_sub["image_id"].tolist()
test_paths = (TEST_DIR + "/" + sample_sub["image_id"].values).tolist()

missing = [p for p in test_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

print("Test rows:", len(test_image_ids))
print("Feature extraction jobs:", N_JOBS_FEAT)



## === cell 2
import hashlib


def _dataset_cache_key(paths, data_root_hint: str) -> str:
    h = hashlib.md5()
    h.update(
        f"IMG_SIZE={IMG_SIZE};PPC={HOG_PIXELS_PER_CELL};CPB={HOG_CELLS_PER_BLOCK};ORI={HOG_ORIENTATIONS};BN=L2-Hys;RGBHOG_v1".encode(
            "utf-8"
        )
    )

    try:
        st_dir = os.stat(data_root_hint)
        h.update(str(int(st_dir.st_mtime_ns)).encode("ascii"))
        h.update(str(int(st_dir.st_size)).encode("ascii"))
    except FileNotFoundError:
        pass

    for p in paths:
        h.update(os.path.basename(p).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _hog_2d(gray_2d: np.ndarray) -> np.ndarray:
    return hog(
        gray_2d,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    ).astype(np.float32, copy=False)


def extract_hog_feature(image_path: str) -> np.ndarray:
    """
    Read image -> resize -> concatenate HOG features from each RGB channel.
    Core logic preserved.
    """
    img = imread(image_path)

    if img.ndim == 2:
        gray = img.astype(np.float32, copy=False)
        if gray.max() > 1.5:
            gray = gray / 255.0
        gray = resize(
            gray,
            (IMG_SIZE, IMG_SIZE),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32, copy=False)
        return _hog_2d(gray)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img.astype(np.float32, copy=False)
    if img.max() > 1.5:
        img = img / 255.0

    img = resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32, copy=False)

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    feat = np.concatenate([_hog_2d(r), _hog_2d(g), _hog_2d(b)], axis=0).astype(
        np.float32, copy=False
    )
    return feat


def _compute_one_row(idx_path):
    idx, p = idx_path
    return idx, extract_hog_feature(p)


def build_features_from_paths(
    paths, n_jobs: int, cache_prefix: str, data_root_hint: str
) -> np.ndarray:
    key = _dataset_cache_key(paths, data_root_hint=data_root_hint)
    meta_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.meta.npz")
    data_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.dat")

    if os.path.exists(meta_path) and os.path.exists(data_path):
        meta = np.load(meta_path, allow_pickle=False)
        shape = tuple(meta["shape"])
        X_mm = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
        return np.asarray(X_mm)

    f0 = extract_hog_feature(paths[0])
    d = int(f0.shape[0])
    n = len(paths)

    X_mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=(n, d))

    X_mm[0] = f0
    X_mm.flush()

    items = [(i, paths[i]) for i in range(1, n)]

    gen = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        batch_size=64,
        pre_dispatch="4*n_jobs",
        return_as="generator",
    )(delayed(_compute_one_row)(it) for it in items)

    for idx, feat in gen:
        X_mm[idx] = feat

    X_mm.flush()
    np.savez_compressed(meta_path, shape=np.array([n, d], dtype=np.int64))
    return np.asarray(X_mm)


train_paths = (TRAIN_DIR + "/" + train_df["image_id"].values).tolist()
y = train_df["label"].values

print("Extracting HOG features for train set (parallel CPU)...")
X = build_features_from_paths(
    train_paths, n_jobs=N_JOBS_FEAT, cache_prefix="train", data_root_hint=TRAIN_DIR
)
print("Train feature matrix:", X.shape, "dtype:", X.dtype)



## === cell 3
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                max_iter=1200,
                multi_class="multinomial",
                solver="lbfgs",
                n_jobs=1,
                class_weight=None,
                C=3.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print("Fitting classifier (train split for sanity-check)...")
clf.fit(X_tr, y_tr)
va_acc = clf.score(X_va, y_va)
print("Validation accuracy (sanity check):", va_acc)

print("Refitting classifier on full training data for final test inference...")
clf.fit(X, y)



## === cell 4
print("Extracting HOG features for test set (parallel CPU)...")
X_test = build_features_from_paths(
    test_paths, n_jobs=N_JOBS_FEAT, cache_prefix="test", data_root_hint=TEST_DIR
)
print("Test feature matrix:", X_test.shape, "dtype:", X_test.dtype)

pred_labels = clf.predict(X_test).astype(int)
pred_labels = np.clip(pred_labels, 0, 4)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
print("Label distribution:\n", submission_df["label"].value_counts().sort_index())

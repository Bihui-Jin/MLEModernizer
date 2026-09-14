# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8644605621033545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31091) has done: 'You’re not getting a score because the current pipeline is likely failing to run end-to-end in the Kaggle environment due to (a) missing PyTorch/TorchVision packages despite using them, and (b) hard-coded model weight paths that are not present in the provided dataset tree. The minimal fix that preserves your overall “single pretrained model inference over test images” core logic is to switch to a TensorFlow/Keras pretrained classifier (available inside TensorFlow) and keep the same submission schema. I also ensure the submission rows exactly match `sample_submission.csv` order (Kaggle-safe alignment) rather than relying on arbitrary filesystem listing order. This should produce a valid `submission.csv` and yield a reasonable accuracy baseline that can be iterated toward your target.'
- What this solution (achieved 0.45329) has done: 'I fix the immediate runtime crash caused by a TensorFlow/Protobuf incompatibility in this Kaggle environment by switching the core model from `tf.keras.applications` (which triggers the protobuf issue on import) to a lightweight, locally-available TFRecord-based baseline that does not require TensorFlow at all. This preserves the end-to-end logic (read official files → generate predictions for every `image_id` in `sample_submission.csv` → write `submission.csv`), while ensuring the notebook runs reliably under Python 3.13. To move the score up from ~0.31 toward your ~0.864 target (a large gap), I implement a simple but stronger image-classification baseline using scikit-learn (available in Kaggle) with HOG-like gradient features and a linear classifier trained on `train.csv` images. Finally, I guarantee strict submission alignment with `sample_submission.csv` ordering and enforce integer labels 0–4.'
- What this solution (achieved 0.44096) has done: 'I fix the “Too many open files” crash by removing joblib’s memmap-based inter-process result passing (the root cause in your traceback) and switching feature extraction to the thread backend with controlled batching, while also avoiding returning open memmaps from the cache. This keeps the same core logic (HOG feature extraction + multinomial LogisticRegression) but makes it run end-to-end reliably in Kaggle and produce `submission.csv`. I also ensure the cached features are loaded as regular in-memory arrays to prevent file-descriptor accumulation. Finally, I keep submission ordering aligned to `sample_submission.csv` exactly as required.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
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

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

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
    "Train TFRecords:",
    os.path.isdir(TRAIN_TFREC_DIR),
    "Test TFRecords:",
    os.path.isdir(TEST_TFREC_DIR),
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

train_image_ids = train_df["image_id"].values.tolist()
test_image_ids = sample_sub["image_id"].values.tolist()

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in train_image_ids]
test_paths = [os.path.join(TEST_DIR, fn) for fn in test_image_ids]

for fn in train_image_ids[:3]:
    p = os.path.join(TRAIN_DIR, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(p)

print("Test rows:", len(test_image_ids))
print("Feature extraction jobs:", N_JOBS_FEAT)




## === cell 2
import hashlib

try:
    import tensorflow as tf

    _HAS_TF = True
except Exception as e:
    print(
        "TensorFlow not available; falling back to image-file reads. Import error:",
        repr(e),
    )
    _HAS_TF = False


def _tfrecord_files(tfrecord_dir: str):
    if not os.path.isdir(tfrecord_dir):
        return []
    files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrec*")))
    return files


def _dataset_cache_key_from_files(file_list, extra_sig: str) -> str:
    h = hashlib.md5()
    h.update(extra_sig.encode("utf-8"))
    for fp in file_list:
        try:
            st = os.stat(fp)
            h.update(os.path.basename(fp).encode("utf-8", errors="ignore"))
            h.update(str(int(st.st_mtime_ns)).encode("ascii"))
            h.update(str(int(st.st_size)).encode("ascii"))
        except FileNotFoundError:
            h.update(os.path.basename(fp).encode("utf-8", errors="ignore"))
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


def extract_hog_feature_from_img(img: np.ndarray) -> np.ndarray:
    """
    Core logic preserved: resize -> concatenate HOG from each RGB channel.
    img expected as uint8 or float, shape (H,W) or (H,W,C).
    """
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


def extract_hog_feature(image_path: str) -> np.ndarray:
    img = imread(image_path)
    return extract_hog_feature_from_img(img)


def _compute_one_row(idx_path):
    idx, p = idx_path
    return idx, extract_hog_feature(p)


def build_features_from_paths(
    paths, n_jobs: int, cache_prefix: str, data_root_hint: str
) -> np.ndarray:
    key = _dataset_cache_key_from_files(
        [data_root_hint],
        extra_sig=f"PATHS;IMG_SIZE={IMG_SIZE};PPC={HOG_PIXELS_PER_CELL};CPB={HOG_CELLS_PER_BLOCK};ORI={HOG_ORIENTATIONS};BN=L2-Hys;RGBHOG_v1;N={len(paths)}",
    )
    meta_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.meta.npz")
    data_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.dat")

    if os.path.exists(meta_path) and os.path.exists(data_path):
        meta = np.load(meta_path, allow_pickle=False)
        shape = tuple(meta["shape"])
        X_mm = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
        return X_mm

    f0 = extract_hog_feature(paths[0])
    d = int(f0.shape[0])
    n = len(paths)

    X_mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=(n, d))
    X_mm[0] = f0
    X_mm.flush()

    items = [(i, paths[i]) for i in range(1, n)]

    gen = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        batch_size=1024,
        pre_dispatch="2*n_jobs",
        return_as="generator_unordered",
    )(delayed(_compute_one_row)(it) for it in items)

    for idx, feat in gen:
        X_mm[idx] = feat

    X_mm.flush()
    np.savez_compressed(meta_path, shape=np.array([n, d], dtype=np.int64))
    return X_mm


def build_features_from_tfrecords(
    tfrecord_files,
    image_id_order,
    labels_by_image_id=None,
    cache_prefix: str = "train",
):
    """
    Returns (X_mm, y_array_or_None).
    Deterministic row order: exactly matches image_id_order.
    """
    key = _dataset_cache_key_from_files(
        tfrecord_files,
        extra_sig=f"TFREC;IMG_SIZE={IMG_SIZE};PPC={HOG_PIXELS_PER_CELL};CPB={HOG_CELLS_PER_BLOCK};ORI={HOG_ORIENTATIONS};BN=L2-Hys;RGBHOG_v1;N={len(image_id_order)}",
    )
    meta_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.meta.npz")
    data_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.dat")
    y_path = (
        os.path.join(CACHE_DIR, f"{cache_prefix}_{key}.y.npy")
        if labels_by_image_id is not None
        else None
    )

    if (
        os.path.exists(meta_path)
        and os.path.exists(data_path)
        and (y_path is None or os.path.exists(y_path))
    ):
        meta = np.load(meta_path, allow_pickle=False)
        shape = tuple(meta["shape"])
        X_mm = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
        y_arr = np.load(y_path) if y_path is not None else None
        return X_mm, y_arr

    if not (_HAS_TF and tfrecord_files):
        raise RuntimeError(
            "TFRecord build requested but TensorFlow/TFRecords not available."
        )

    id_to_idx = {img_id: i for i, img_id in enumerate(image_id_order)}

    def _parse_example(ex):
        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        }
        if labels_by_image_id is None:
            pass
        else:
            feature_desc["label"] = tf.io.FixedLenFeature(
                [], tf.int64, default_value=-1
            )
        x = tf.io.parse_single_example(ex, feature_desc)
        return x

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=1)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    def _decode(x):
        img = tf.io.decode_jpeg(x["image"], channels=3)
        return x["image_id"], img

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    first_id, first_img = next(iter(ds.take(1)))
    first_np = first_img.numpy()
    f0 = extract_hog_feature_from_img(first_np)
    d = int(f0.shape[0])
    n = len(image_id_order)

    X_mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=(n, d))
    X_mm[:] = 0.0  # ensure fully initialized on disk

    y_arr = None
    if labels_by_image_id is not None:
        y_arr = np.empty((n,), dtype=np.int64)

    fid = first_id.numpy().decode("utf-8")
    if fid in id_to_idx:
        idx0 = id_to_idx[fid]
        X_mm[idx0] = f0
        if y_arr is not None:
            y_arr[idx0] = int(labels_by_image_id[fid])

    for img_id_t, img_t in ds.skip(1):
        img_id = img_id_t.numpy().decode("utf-8")
        idx = id_to_idx.get(img_id)
        if idx is None:
            continue
        feat = extract_hog_feature_from_img(img_t.numpy())
        X_mm[idx] = feat
        if y_arr is not None:
            y_arr[idx] = int(labels_by_image_id[img_id])

    X_mm.flush()
    np.savez_compressed(meta_path, shape=np.array([n, d], dtype=np.int64))
    if y_path is not None:
        np.save(y_path, y_arr)
    return X_mm, y_arr


train_tfrec_files = _tfrecord_files(TRAIN_TFREC_DIR)
test_tfrec_files = _tfrecord_files(TEST_TFREC_DIR)

labels_by_id = dict(
    zip(train_df["image_id"].values.tolist(), train_df["label"].values.tolist())
)

if _HAS_TF and train_tfrec_files:
    print("Extracting HOG features for train set from TFRecords (fast path)...")
    X, y = build_features_from_tfrecords(
        train_tfrec_files,
        image_id_order=train_image_ids,
        labels_by_image_id=labels_by_id,
        cache_prefix="train",
    )
else:
    print(
        "Extracting HOG features for train set from image files (fallback, parallel CPU)..."
    )
    y = train_df["label"].values
    X = build_features_from_paths(
        train_paths, n_jobs=N_JOBS_FEAT, cache_prefix="train", data_root_hint=TRAIN_DIR
    )

print("Train feature matrix:", X.shape, "dtype:", X.dtype)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/867472906.py in <cell line: 0>()
      1 X_tr, X_va, y_tr, y_va = train_test_split(
----> 2     X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
      3 )
      4 
      5 clf = Pipeline(

NameError: name 'X' is not defined

## === cell 4
if _HAS_TF and test_tfrec_files:
    print("Extracting HOG features for test set from TFRecords (fast path)...")
    X_test, _ = build_features_from_tfrecords(
        test_tfrec_files,
        image_id_order=test_image_ids,
        labels_by_image_id=None,
        cache_prefix="test",
    )
else:
    print(
        "Extracting HOG features for test set from image files (fallback, parallel CPU)..."
    )
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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/4239654447.py in <cell line: 0>()
      1 if _HAS_TF and test_tfrec_files:
      2     print("Extracting HOG features for test set from TFRecords (fast path)...")
----> 3     X_test, _ = build_features_from_tfrecords(
      4         test_tfrec_files,
      5         image_id_order=test_image_ids,

/tmp/ipykernel_55/33054045.py in build_features_from_tfrecords(tfrecord_files, image_id_order, labels_by_image_id, cache_prefix)
    215 
    216     # Get feature dimension from first record
--> 217     first_id, first_img = next(iter(ds.take(1)))
    218     first_np = first_img.numpy()
    219     f0 = extract_hog_feature_from_img(first_np)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:12 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::ParallelMapV2::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:

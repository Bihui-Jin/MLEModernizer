# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

CACHE_DIR = "/kaggle/working/preprocessed_cache_224"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    _cpu = os.cpu_count() or 4
    cv2.setNumThreads(max(1, min(4, _cpu // 2)))
except Exception:
    pass


def crop_image_from_gray(img, tol=7):
    if img is None:
        return img

    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y : y + h, x : x + w]

    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y : y + h, x : x + w, :]

    return img


def preprocessing_rgb_uint8_to_float(image_rgb_uint8, sigmaX=10):
    image = crop_image_from_gray(image_rgb_uint8).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def _cache_path_for_filepath(filepath: str) -> str:
    base = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(CACHE_DIR, f"{base}_{IMG_SIZE}.tensor")


def _preprocess_one_to_cache(filepath: str) -> str:
    out_path = _cache_path_for_filepath(filepath)
    if os.path.exists(out_path):
        return out_path

    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = preprocessing_rgb_uint8_to_float(img_rgb)  # float32 [0,1], (H,W,3)

    tensor_bytes = tf.io.serialize_tensor(
        tf.convert_to_tensor(arr, dtype=tf.float32)
    ).numpy()

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(tensor_bytes)
    os.replace(tmp_path, out_path)
    return out_path


def build_cache_for_filepaths(filepaths, workers=None):
    filepaths = list(map(str, filepaths))
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) // 2))

    cache_paths = [_cache_path_for_filepath(fp) for fp in filepaths]
    missing = [fp for fp, cp in zip(filepaths, cache_paths) if not os.path.exists(cp)]
    if not missing:
        return

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_preprocess_one_to_cache, missing, chunksize=16):
            pass


def build_memmap_from_cached_npy(cache_paths, mmap_prefix: str):
    cache_paths = np.asarray(cache_paths, dtype=object)
    idx_path = os.path.join(CACHE_DIR, f"{mmap_prefix}_{IMG_SIZE}_paths.npy")
    tmp_idx_path = idx_path + ".tmp"
    with open(tmp_idx_path, "wb") as f:
        np.save(f, cache_paths, allow_pickle=True)
    os.replace(tmp_idx_path, idx_path)
    return None, idx_path




## === cell 2
base_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model = tf.keras.models.Sequential(
    [
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["id_code"].astype(str) + ".png"
exists_mask = train_df["filepath"].map(os.path.exists)
train_df = train_df[exists_mask].reset_index(drop=True)

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

build_cache_for_filepaths(tr_df["filepath"].values)
build_cache_for_filepaths(va_df["filepath"].values)

tr_cache_paths = tr_df["filepath"].map(_cache_path_for_filepath).values
va_cache_paths = va_df["filepath"].map(_cache_path_for_filepath).values
tr_dat, tr_idx = build_memmap_from_cached_npy(tr_cache_paths, "train")
va_dat, va_idx = build_memmap_from_cached_npy(va_cache_paths, "val")


def make_dataset_from_cached_tensors(cache_paths, labels, shuffle=False):
    cache_paths = np.asarray(cache_paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((cache_paths, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    if shuffle:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)

    def _load_one(path, y):
        b = tf.io.read_file(path)
        img = tf.io.parse_tensor(b, out_type=tf.float32)
        img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
        return img, y

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset_from_cached_tensors(
    tr_cache_paths, tr_df["diagnosis"].values, shuffle=True
)
val_ds = make_dataset_from_cached_tensors(
    va_cache_paths, va_df["diagnosis"].values, shuffle=False
)

EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

gc.collect()




## === cell 4
test_df = pd.read_csv(TEST_CSV)

test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"
assert test_df.shape[0] > 0

build_cache_for_filepaths(test_df["filepath"].values)

te_cache_paths = test_df["filepath"].map(_cache_path_for_filepath).values
te_dat, te_idx = build_memmap_from_cached_npy(te_cache_paths, "test")


def make_test_dataset_from_cached_tensors(cache_paths):
    cache_paths = np.asarray(cache_paths, dtype=str)
    ds = tf.data.Dataset.from_tensor_slices(cache_paths)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    def _load_one(path):
        b = tf.io.read_file(path)
        img = tf.io.parse_tensor(b, out_type=tf.float32)
        img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
        return img

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset_from_cached_tensors(te_cache_paths)
probs = model.predict(test_ds, verbose=1)
test_pred = np.argmax(probs, axis=1).astype("int64")

sub = pd.read_csv(SAMPLE_SUB)
sub = sub.merge(test_df[["id_code"]], on="id_code", how="right")  # keep correct ids
sub["diagnosis"] = test_pred
sub = sub[["id_code", "diagnosis"]]
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_pred, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

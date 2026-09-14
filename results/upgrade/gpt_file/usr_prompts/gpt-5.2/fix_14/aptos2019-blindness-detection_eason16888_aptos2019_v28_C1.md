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


def _mmap_paths(prefix: str):
    data_path = os.path.join(CACHE_DIR, f"{prefix}_{IMG_SIZE}.mmap")
    meta_path = os.path.join(CACHE_DIR, f"{prefix}_{IMG_SIZE}_meta.npz")
    paths_path = os.path.join(CACHE_DIR, f"{prefix}_{IMG_SIZE}_paths.npy")
    return data_path, meta_path, paths_path


def _process_index_and_path_top(args):
    i, fp, img_size, sigmaX = args
    img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {fp}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = crop_image_from_gray(img_rgb).astype("uint8")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    arr = img.astype("float32") / 255.0
    return i, arr


def build_memmap_cache_for_filepaths(filepaths, mmap_prefix: str, workers=None):
    filepaths = list(map(str, filepaths))
    data_path, meta_path, paths_path = _mmap_paths(mmap_prefix)

    if (
        os.path.exists(data_path)
        and os.path.exists(meta_path)
        and os.path.exists(paths_path)
    ):
        try:
            meta = np.load(meta_path)
            if (
                int(meta["n"]) == len(filepaths)
                and int(meta["img_size"]) == IMG_SIZE
                and str(meta["dtype"]) == "float32"
            ):
                old_paths = np.load(paths_path, allow_pickle=True)
                if len(old_paths) == len(filepaths) and np.all(
                    old_paths == np.asarray(filepaths, dtype=object)
                ):
                    return data_path, meta_path, paths_path
        except Exception:
            pass  # fall through to rebuild

    if workers is None:
        workers = min(12, max(1, (os.cpu_count() or 2)))

    filepaths_arr = np.asarray(filepaths, dtype=object)

    n = len(filepaths_arr)
    shape = (n, IMG_SIZE, IMG_SIZE, 3)
    tmp_data_path = data_path + ".tmp"
    if os.path.exists(tmp_data_path):
        try:
            os.remove(tmp_data_path)
        except Exception:
            pass

    mm = np.memmap(tmp_data_path, dtype="float32", mode="w+", shape=shape)

    from concurrent.futures import ThreadPoolExecutor

    idx_and_fp = list(enumerate(filepaths_arr.tolist()))
    packed = [(i, fp, IMG_SIZE, 10) for i, fp in idx_and_fp]
    chunksize = 64 if n >= 1024 else 16

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, arr in ex.map(_process_index_and_path_top, packed, chunksize=chunksize):
            mm[i] = arr

    mm.flush()
    del mm
    os.replace(tmp_data_path, data_path)

    tmp_meta_base = meta_path + ".tmp_base"
    for p in (tmp_meta_base, tmp_meta_base + ".npz"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    np.savez(tmp_meta_base, n=n, img_size=IMG_SIZE, dtype=np.dtype("float32").name)
    os.replace(tmp_meta_base + ".npz", meta_path)

    tmp_paths_path = paths_path + ".tmp"
    with open(tmp_paths_path, "wb") as f:
        np.save(f, filepaths_arr, allow_pickle=True)
    os.replace(tmp_paths_path, paths_path)

    return data_path, meta_path, paths_path


def make_dataset_from_memmap(
    data_path, n_items, labels=None, shuffle=False, indices=None
):
    mm = np.memmap(
        data_path, dtype="float32", mode="r", shape=(n_items, IMG_SIZE, IMG_SIZE, 3)
    )

    if indices is None:
        indices = np.arange(n_items, dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(indices)
    else:
        labels = np.asarray(labels, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((indices, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    if shuffle:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)

    def _get_np(i):
        return mm[int(i)]

    if labels is None:

        def _load_one(i):
            img = tf.numpy_function(_get_np, [i], Tout=tf.float32)
            img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
            return img

        ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    else:

        def _load_one(i, y):
            img = tf.numpy_function(_get_np, [i], Tout=tf.float32)
            img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
            return img, y

        ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




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

filepaths = train_df["filepath"].to_numpy()
exists = np.fromiter(
    (os.path.exists(p) for p in filepaths), dtype=bool, count=len(filepaths)
)
train_df = train_df[exists].reset_index(drop=True)

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

full_tr_data_path, full_tr_meta_path, full_tr_paths_path = (
    build_memmap_cache_for_filepaths(
        train_df["filepath"].values, mmap_prefix="train_full", workers=None
    )
)

fp_to_idx = {fp: i for i, fp in enumerate(train_df["filepath"].values.tolist())}
tr_indices = np.fromiter(
    (fp_to_idx[fp] for fp in tr_df["filepath"].values.tolist()),
    dtype=np.int32,
    count=len(tr_df),
)
va_indices = np.fromiter(
    (fp_to_idx[fp] for fp in va_df["filepath"].values.tolist()),
    dtype=np.int32,
    count=len(va_df),
)

train_ds = make_dataset_from_memmap(
    full_tr_data_path,
    n_items=len(train_df),
    labels=tr_df["diagnosis"].values,
    shuffle=True,
    indices=tr_indices,
)
val_ds = make_dataset_from_memmap(
    full_tr_data_path,
    n_items=len(train_df),
    labels=va_df["diagnosis"].values,
    shuffle=False,
    indices=va_indices,
)

EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

gc.collect()



## === cell 4
test_df = pd.read_csv(TEST_CSV)

test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"
assert test_df.shape[0] > 0

te_data_path, te_meta_path, te_paths_path = build_memmap_cache_for_filepaths(
    test_df["filepath"].values, mmap_prefix="test", workers=None
)

test_ds = make_dataset_from_memmap(
    te_data_path, n_items=len(test_df), labels=None, shuffle=False
)
probs = model.predict(test_ds, verbose=1)
test_pred = np.argmax(probs, axis=1).astype("int64")

sub = pd.read_csv(SAMPLE_SUB)
sub = sub[["id_code"]].merge(test_df[["id_code"]], on="id_code", how="right")
sub["diagnosis"] = test_pred
sub = sub[["id_code", "diagnosis"]]
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_pred, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

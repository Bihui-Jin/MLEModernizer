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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import gc
import random
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 2)))
except Exception:
    pass




## === cell 1
"""
Config + image preprocessing (kept core logic: Ben Graham style preprocessing + cropping)

Speed fix (equivalent semantics):
- Cache stores uint8 RGB bytes instead of float32 to cut disk I/O by 4x and write/read faster.
- At load time we cast to float32 and divide by 255.0, yielding the same tensor values
  as the previous float32 cache (negligible FP differences only from dtype conversion order).
"""

IMG_SIZE = 224
BATCH_SIZE = 16

DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

CACHE_DIR = "/kaggle/working/preprocessed_cache_224"
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, "train")
TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test")
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)

CACHE_EXT = ".bin"
CACHE_NBYTES = IMG_SIZE * IMG_SIZE * 3  # uint8 bytes on disk


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_color_bgr(image_bgr, sigmaX=10):
    if image_bgr is None:
        return None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocess_path_uint8(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = load_ben_color_bgr(img)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        img = np.asarray(img, dtype=np.uint8).reshape((IMG_SIZE, IMG_SIZE, 3))
    return img


def preprocess_path_cached_raw(path: str, cache_dir: str):
    base = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, base + CACHE_EXT)

    try:
        st = os.stat(cache_path)
        if st.st_size == CACHE_NBYTES:
            return cache_path
    except FileNotFoundError:
        pass
    except Exception:
        pass

    arr = preprocess_path_uint8(path)
    if arr is None:
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(arr.tobytes(order="C"))
    os.replace(tmp_path, cache_path)
    return cache_path




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert set(["id_code", "diagnosis"]).issubset(train_df.columns)
assert set(["id_code"]).issubset(test_df.columns)
assert set(["id_code", "diagnosis"]).issubset(sub_df.columns)

print("Train:", train_df.shape, " Test:", test_df.shape, " Sample:", sub_df.shape)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train split:", tr_df.shape, "Val split:", va_df.shape)

cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=tr_df["diagnosis"].values,
)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Class weights:", class_weights)




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _cache_worker(args):
    pth, cache_dir = args
    preprocess_path_cached_raw(pth, cache_dir)
    return 1


def _ensure_cache_for_paths(paths, cache_dir, workers=None):
    """
    Speed fix (equivalent semantics):
    - Avoid O(#files-in-cache-dir) scans; instead check existence/size only for needed ids.
      This is provably correct because cache file name is a pure function of id_code.
    - Keep multiprocessing but reduce Python overhead with larger chunksize.
    """
    from multiprocessing import get_context

    os.makedirs(cache_dir, exist_ok=True)

    tasks = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        cp = os.path.join(cache_dir, base + CACHE_EXT)
        try:
            if os.path.getsize(cp) == CACHE_NBYTES:
                continue
            try:
                os.remove(cp)
            except Exception:
                pass
        except FileNotFoundError:
            pass
        except Exception:
            pass
        tasks.append(p)

    if not tasks:
        return

    if workers is None:
        cpu = os.cpu_count() or 2
        workers = max(1, min(8, cpu - 1))

    if workers <= 1 or len(tasks) < 64:
        for p in tasks:
            preprocess_path_cached_raw(p, cache_dir)
        return

    try:
        ctx = get_context("fork")
    except Exception:
        ctx = get_context("spawn")

    arg_iter = ((p, cache_dir) for p in tasks)
    with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
        for _ in pool.imap_unordered(_cache_worker, arg_iter, chunksize=512):
            pass


tr_paths = np.char.add(
    np.char.add(TRAIN_IMG_DIR + "/", tr_df["id_code"].values.astype(str)), ".png"
).astype(str)
va_paths = np.char.add(
    np.char.add(TRAIN_IMG_DIR + "/", va_df["id_code"].values.astype(str)), ".png"
).astype(str)
te_paths = np.char.add(
    np.char.add(TEST_IMG_DIR + "/", test_df["id_code"].values.astype(str)), ".png"
).astype(str)

_ensure_cache_for_paths(tr_paths, TRAIN_CACHE_DIR)
_ensure_cache_for_paths(va_paths, TRAIN_CACHE_DIR)
_ensure_cache_for_paths(te_paths, TEST_CACHE_DIR)


def make_dataset_from_cache(paths, labels, training, cache_dir, with_labels=True):
    """
    Speed fix (equivalent semantics):
    - Replace per-element FixedLengthRecordDataset construction with tf.io.read_file.
      This avoids creating a Dataset inside map(), which is a major bottleneck.
    - Decode cached uint8 bytes and convert to float32/255.0 (same values as prior pipeline).
    """
    paths_arr = np.asarray(paths, dtype=str)
    bases = np.char.rpartition(paths_arr, "/")[:, 2]
    bases = np.char.replace(bases, ".png", "")
    cache_paths = np.char.add(np.char.add(cache_dir + "/", bases), CACHE_EXT).astype(
        str
    )

    for cp in cache_paths[: min(64, len(cache_paths))]:
        try:
            if os.path.getsize(cp) != CACHE_NBYTES:
                _ensure_cache_for_paths(paths_arr, cache_dir)
                break
        except Exception:
            _ensure_cache_for_paths(paths_arr, cache_dir)
            break

    options = tf.data.Options()
    options.deterministic = True

    path_ds = tf.data.Dataset.from_tensor_slices(cache_paths)

    @tf.function
    def _read_cached_uint8(path):
        raw = tf.io.read_file(path)  # bytes length = CACHE_NBYTES
        flat = tf.io.decode_raw(raw, out_type=tf.uint8)
        img = tf.reshape(flat, [IMG_SIZE, IMG_SIZE, 3])
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        return img

    if with_labels:
        label_ds = tf.data.Dataset.from_tensor_slices(
            np.asarray(labels, dtype=np.int32)
        )
        ds = tf.data.Dataset.zip((path_ds, label_ds))

        def _load(path, label):
            img = _read_cached_uint8(path)
            return img, tf.one_hot(label, 5)

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    else:
        ds = path_ds.map(_read_cached_uint8, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


train_ds = make_dataset_from_cache(
    tr_paths,
    tr_df["diagnosis"].values,
    training=True,
    cache_dir=TRAIN_CACHE_DIR,
    with_labels=True,
)
val_ds = make_dataset_from_cache(
    va_paths,
    va_df["diagnosis"].values,
    training=False,
    cache_dir=TRAIN_CACHE_DIR,
    with_labels=True,
)
test_ds = make_dataset_from_cache(
    te_paths,
    labels=None,
    training=False,
    cache_dir=TEST_CACHE_DIR,
    with_labels=False,
)




## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base = tf.keras.applications.DenseNet121(
    include_top=False, weights="imagenet", input_tensor=inputs
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 5
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)

for layer in base.layers[-60:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)




## === cell 6
pred_proba = model.predict(test_ds, verbose=1)
test_pred = np.argmax(pred_proba, axis=1).astype(np.int64)

submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": test_pred}
)
submission = sub_df[["id_code"]].merge(submission, on="id_code", how="left")
assert submission["diagnosis"].isna().sum() == 0
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())

unique, counts = np.unique(test_pred, return_counts=True)
print("Pred label distribution:", dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv ->", os.path.abspath("submission.csv"))

gc.collect()

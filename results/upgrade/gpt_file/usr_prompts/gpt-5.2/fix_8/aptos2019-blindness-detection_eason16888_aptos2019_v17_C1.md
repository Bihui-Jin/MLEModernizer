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

import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.data.experimental.enable_debug_mode = False

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

print("TF:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5  # keep identical training schedule


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img

    _, mask = cv2.threshold(gray, tol, 255, cv2.THRESH_BINARY)
    nz = cv2.findNonZero(mask)
    if nz is None:
        return img
    x, y, w, h = cv2.boundingRect(nz)

    if img.ndim == 2:
        return img[y : y + h, x : x + w]
    else:
        return img[y : y + h, x : x + w, :]


def load_ben_color(image_bgr, sigmaX=10):
    if image_bgr is None:
        return None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

assert "id_code" in train_df.columns and "diagnosis" in train_df.columns
assert "id_code" in test_df.columns
assert list(sample_sub.columns) == ["id_code", "diagnosis"]

trn_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

print("Train:", trn_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)



## === cell 3
import multiprocessing as mp

CACHE_DIR = "/kaggle/working/preprocessed_cache_224"
os.makedirs(CACHE_DIR, exist_ok=True)


def _preprocess_one_for_pool(args):
    idx, id_code, img_dir, img_size = args
    img_path = os.path.join(img_dir, f"{id_code}.png")
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        arr_f = np.zeros((img_size, img_size, 3), dtype=np.float32)
    else:
        arr_f = load_ben_color(img_bgr, sigmaX=10)
        if arr_f is None:
            arr_f = np.zeros((img_size, img_size, 3), dtype=np.float32)
    arr_u8 = np.clip(arr_f * 255.0 + 0.5, 0, 255).astype(np.uint8, copy=False)
    return idx, arr_u8


def _split_paths(split_name):
    x_path = os.path.join(CACHE_DIR, f"{split_name}_X_uint8.dat")
    ids_path = os.path.join(CACHE_DIR, f"{split_name}_ids.npy")
    return x_path, ids_path


def _load_or_build_split_memmap(split_name, df, img_dir, with_labels):
    x_path, ids_path = _split_paths(split_name)
    y = None
    if with_labels:
        y = df["diagnosis"].values.astype(np.int32, copy=False)
    ids = df["id_code"].values.astype(object)

    if os.path.exists(x_path) and os.path.exists(ids_path):
        cached_ids = np.load(ids_path, allow_pickle=True)
        if len(cached_ids) == len(ids) and np.all(cached_ids == ids):
            print(f"[cache] {split_name}: using existing memmap cache")
            return x_path, ids, y
        else:
            try:
                os.remove(x_path)
            except Exception:
                pass
            try:
                os.remove(ids_path)
            except Exception:
                pass

    np.save(ids_path, ids, allow_pickle=True)

    n = len(ids)
    shape = (n, IMG_SIZE, IMG_SIZE, 3)
    mm = np.memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)

    cpu = mp.cpu_count()
    nworkers = min(8, max(2, cpu // 2))
    chunksize = 16  # reduces IPC overhead; deterministic ordering is restored by writing via idx.
    print(
        f"[cache] {split_name}: building memmap cache with {nworkers} processes, n={n}"
    )

    args_iter = ((i, id_code, img_dir, IMG_SIZE) for i, id_code in enumerate(ids))
    done = 0
    with mp.get_context("fork").Pool(processes=nworkers) as pool:
        for idx, arr_u8 in pool.imap_unordered(
            _preprocess_one_for_pool, args_iter, chunksize=chunksize
        ):
            mm[idx] = arr_u8
            done += 1
            if done % 500 == 0 or done == n:
                print(f"[cache] {split_name}: {done}/{n}")

    mm.flush()
    del mm
    return x_path, ids, y


trn_x_path, trn_ids, trn_labels = _load_or_build_split_memmap(
    "train", trn_df, TRAIN_IMG_DIR, with_labels=True
)
val_x_path, val_ids, val_labels = _load_or_build_split_memmap(
    "val", val_df, TRAIN_IMG_DIR, with_labels=True
)
tst_x_path, tst_ids, _ = _load_or_build_split_memmap(
    "test", test_df, TEST_IMG_DIR, with_labels=False
)

print("Cached memmaps:", trn_x_path, val_x_path, tst_x_path)




## === cell 4
def make_memmap_ds(x_path, n, labels=None, training=False):
    shape = (n, IMG_SIZE, IMG_SIZE, 3)
    mm = np.memmap(x_path, mode="r", dtype=np.uint8, shape=shape)

    idxs = tf.data.Dataset.range(n)
    if training:
        idxs = idxs.shuffle(min(n, 2048), seed=SEED, reshuffle_each_iteration=True)

    idxs = idxs.batch(BATCH_SIZE, drop_remainder=False)

    def _load_batch(batch_idxs):
        def _np_get_batch(ii):
            ii = np.asarray(ii, dtype=np.int64)
            arr = np.array(mm[ii], copy=True)  # (B,H,W,3) uint8
            return arr.astype(np.float32) / 255.0

        imgs = tf.py_function(_np_get_batch, [batch_idxs], Tout=tf.float32)
        imgs.set_shape((None, IMG_SIZE, IMG_SIZE, 3))
        return imgs

    options = tf.data.Options()
    options.experimental_deterministic = True

    if labels is None:
        ds = idxs.map(_load_batch, num_parallel_calls=2, deterministic=True)
        ds = ds.with_options(options)
        ds = ds.prefetch(2)
        return ds

    labels_tf = tf.constant(labels, dtype=tf.int32)

    def _map(batch_idxs):
        imgs = _load_batch(batch_idxs)
        y = tf.one_hot(tf.gather(labels_tf, batch_idxs), depth=5)
        y.set_shape((None, 5))
        return imgs, y

    ds = idxs.map(_map, num_parallel_calls=2, deterministic=True)
    ds = ds.with_options(options)
    ds = ds.prefetch(2)
    return ds


train_ds = make_memmap_ds(trn_x_path, len(trn_ids), labels=trn_labels, training=True)
val_ds = make_memmap_ds(val_x_path, len(val_ids), labels=val_labels, training=False)
test_ds = make_memmap_ds(tst_x_path, len(tst_ids), labels=None, training=False)



## === cell 5
base = DenseNet121(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # keep identical training approach

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(5, activation="softmax")(x)
model = Model(inp, out)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=trn_df["diagnosis"].values,
)
cw = {i: float(w) for i, w in enumerate(cw)}
print("class_weight:", cw)

ckpt_path = "/kaggle/working/best_model.keras"
cbs = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=cw,
    callbacks=cbs,
    verbose=2,
)

model = keras.models.load_model(ckpt_path)



## === cell 6
probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(np.int64)

sub = test_df.copy()
sub["diagnosis"] = preds

sub = sub[["id_code", "diagnosis"]]
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["id_code", "diagnosis"]

sub_path = "/kaggle/working/submission.csv"
sub.to_csv(sub_path, index=False)

unique, counts = np.unique(preds, return_counts=True)
print("Prediction distribution:", dict(zip(unique.tolist(), counts.tolist())))
print("Wrote:", sub_path)

gc.collect()

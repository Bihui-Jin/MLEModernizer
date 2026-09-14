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
import gc
import numpy as np
import pandas as pd
import cv2

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)




## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3  # keep small to stay within Kaggle time; still trains end-to-end
N_CLASSES = 5

DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

CACHE_DIR = "/kaggle/working/preprocessed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None

    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        y0, y1 = ys[0], ys[-1] + 1
        x0, x1 = xs[0], xs[-1] + 1
        return img[y0:y1, x0:x1]

    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        y0, y1 = ys[0], ys[-1] + 1
        x0, x1 = xs[0], xs[-1] + 1
        return img[y0:y1, x0:x1, :]

    return img


def load_ben_color_bgr_to_rgb(path, sigmaX=10):
    """
    Reads via cv2 (BGR), converts to RGB, applies Ben Graham-like preprocessing,
    returns float32 in [0,1] with shape (IMG_SIZE, IMG_SIZE, 3).
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    img = img.astype("float32") / 255.0
    return img


def preprocess_one_path(p: str):
    img = load_ben_color_bgr_to_rgb(p)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    return img




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

assert "id_code" in train_df.columns and "diagnosis" in train_df.columns
assert "id_code" in test_df.columns

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print("Train class counts:\n", tr_df["diagnosis"].value_counts().sort_index())




## === cell 3
def _cache_paths(prefix: str):
    x_path = os.path.join(CACHE_DIR, f"{prefix}_x.npy")
    y_path = os.path.join(CACHE_DIR, f"{prefix}_y.npy")
    return x_path, y_path


def _preprocess_and_save(df: pd.DataFrame, img_dir: str, prefix: str, has_labels: bool):
    x_path, y_path = _cache_paths(prefix)
    if os.path.exists(x_path) and ((not has_labels) or os.path.exists(y_path)):
        return x_path, (y_path if has_labels else None)

    ids = df["id_code"].astype(str).values
    paths = [os.path.join(img_dir, f"{i}.png") for i in ids]

    x_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(len(paths), IMG_SIZE, IMG_SIZE, 3)
    )

    import multiprocessing as mp

    nproc = max(1, min(8, (os.cpu_count() or 4) - 1))

    chunksize = 32 if len(paths) >= 1024 else 16

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

    with ctx.Pool(processes=nproc) as pool:
        for idx, img in enumerate(
            pool.imap(preprocess_one_path, paths, chunksize=chunksize)
        ):
            x_mm[idx] = img

    del x_mm

    if has_labels:
        y = df["diagnosis"].values.astype(np.int32)
        np.save(y_path, y, allow_pickle=False)
        return x_path, y_path
    return x_path, None


tr_x_path, tr_y_path = _preprocess_and_save(
    tr_df, TRAIN_IMG_DIR, "train", has_labels=True
)
va_x_path, va_y_path = _preprocess_and_save(
    va_df, TRAIN_IMG_DIR, "val", has_labels=True
)
te_x_path, _ = _preprocess_and_save(test_df, TEST_IMG_DIR, "test", has_labels=False)

tr_x = np.load(tr_x_path, mmap_mode="r")
tr_y = np.load(tr_y_path, mmap_mode="r")
va_x = np.load(va_x_path, mmap_mode="r")
va_y = np.load(va_y_path, mmap_mode="r")
te_x = np.load(te_x_path, mmap_mode="r")

print(
    "Cached arrays shapes:",
    "tr_x",
    tr_x.shape,
    "tr_y",
    tr_y.shape,
    "va_x",
    va_x.shape,
    "va_y",
    va_y.shape,
    "te_x",
    te_x.shape,
)




## === cell 4
tr_y_oh = np.eye(N_CLASSES, dtype=np.float32)[np.asarray(tr_y, dtype=np.int32)]
va_y_oh = np.eye(N_CLASSES, dtype=np.float32)[np.asarray(va_y, dtype=np.int32)]


def make_dataset_from_arrays(x, y=None, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(
                min(int(x.shape[0]), 2048), seed=SEED, reshuffle_each_iteration=True
            )

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset_from_arrays(tr_x, tr_y_oh, training=True)
val_ds = make_dataset_from_arrays(va_x, va_y_oh, training=False)
test_ds = make_dataset_from_arrays(te_x, y=None, training=False)




## === cell 5
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
backbone = DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
x = layers.GlobalAveragePooling2D()(backbone.output)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 6
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

gc.collect()




## === cell 7
pred_proba = model.predict(test_ds, verbose=1)
test_prediction = np.argmax(pred_proba, axis=1).astype(np.int64)

submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission["diagnosis"] = test_prediction
submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Done!")

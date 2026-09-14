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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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
    tf.config.optimizer.set_jit(False)
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
AUTOTUNE = tf.data.AUTOTUNE
TOL = 7.0
SIGMA_X = 10.0

try:
    tf.data.experimental.disable_debug_mode()
except Exception:
    pass

_sigma = tf.constant(SIGMA_X, dtype=tf.float32)
_radius = tf.cast(tf.math.ceil(3.0 * _sigma), tf.int32)
_x = tf.cast(tf.range(-_radius, _radius + 1), tf.float32)
_gauss_1d = tf.exp(-(_x * _x) / (2.0 * _sigma * _sigma))
_gauss_1d = _gauss_1d / tf.reduce_sum(_gauss_1d)
_gauss_2d = tf.tensordot(_gauss_1d, _gauss_1d, axes=0)  # [k,k]
_gauss_2d = _gauss_2d[:, :, tf.newaxis, tf.newaxis]  # [k,k,1,1]
GAUSS_KERNEL_DW = tf.tile(_gauss_2d, [1, 1, 3, 1])  # [k,k,3,1]


@tf.function(jit_compile=False)
def _ben_preprocess_tf(img_uint8):
    img = tf.cast(img_uint8, tf.float32)

    gray = tf.image.rgb_to_grayscale(img)  # float32
    mask2 = tf.squeeze(gray > TOL, axis=-1)  # [H,W] bool

    def do_crop_fast():
        rows = tf.reduce_any(mask2, axis=1)  # [H]
        cols = tf.reduce_any(mask2, axis=0)  # [W]

        y0 = tf.argmax(rows, output_type=tf.int32)
        y1 = tf.shape(rows, out_type=tf.int32)[0] - tf.argmax(
            tf.reverse(rows, axis=[0]), output_type=tf.int32
        )
        x0 = tf.argmax(cols, output_type=tf.int32)
        x1 = tf.shape(cols, out_type=tf.int32)[0] - tf.argmax(
            tf.reverse(cols, axis=[0]), output_type=tf.int32
        )

        return img[y0:y1, x0:x1, :]

    img = tf.cond(tf.reduce_any(mask2), do_crop_fast, lambda: img)

    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )

    img4 = img[tf.newaxis, ...]  # [1,H,W,3]
    blurred = tf.nn.depthwise_conv2d(
        img4,
        GAUSS_KERNEL_DW,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
    )[0]

    img = 4.0 * img + (-4.0) * blurred + 128.0
    img = img / 255.0
    return img


def _load_png_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)  # uint8 RGB
    img = _ben_preprocess_tf(img)
    return img


def make_dataset_from_paths(paths, y=None, training=False, cache=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        if training:
            ds = ds.shuffle(
                min(len(paths), 2048), seed=SEED, reshuffle_each_iteration=True
            )

    options = tf.data.Options()
    options.experimental_deterministic = True

    try:
        if hasattr(options, "experimental_optimization") and hasattr(
            options.experimental_optimization, "map_vectorization"
        ):
            options.experimental_optimization.map_vectorization.enabled = True
    except Exception:
        pass

    ds = ds.with_options(options)

    if y is None:
        ds = ds.map(_load_png_and_preprocess, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(
            lambda p, yy: (_load_png_and_preprocess(p), yy), num_parallel_calls=AUTOTUNE
        )

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


tr_paths = (TRAIN_IMG_DIR + "/" + tr_df["id_code"].astype(str).values + ".png").astype(
    str
)
va_paths = (TRAIN_IMG_DIR + "/" + va_df["id_code"].astype(str).values + ".png").astype(
    str
)
te_paths = (TEST_IMG_DIR + "/" + test_df["id_code"].astype(str).values + ".png").astype(
    str
)

tr_y = tr_df["diagnosis"].values.astype(np.int32)
va_y = va_df["diagnosis"].values.astype(np.int32)

tr_y_oh = np.eye(N_CLASSES, dtype=np.float32)[tr_y]
va_y_oh = np.eye(N_CLASSES, dtype=np.float32)[va_y]

train_ds = make_dataset_from_paths(tr_paths, tr_y_oh, training=True, cache=True)
val_ds = make_dataset_from_paths(va_paths, va_y_oh, training=False, cache=True)
test_ds = make_dataset_from_paths(te_paths, y=None, training=False, cache=True)

print(
    "Datasets ready:",
    "train",
    len(tr_paths),
    "val",
    len(va_paths),
    "test",
    len(te_paths),
)



## === cell 4
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



## === cell 5
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

gc.collect()



## === cell 6
pred_proba = model.predict(test_ds, verbose=1)
test_prediction = np.argmax(pred_proba, axis=1).astype(np.int64)

submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
assert submission.shape[0] == test_df.shape[0], "sample_submission/test size mismatch"
submission["diagnosis"] = test_prediction
submission = submission[["id_code", "diagnosis"]]

submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Done!")

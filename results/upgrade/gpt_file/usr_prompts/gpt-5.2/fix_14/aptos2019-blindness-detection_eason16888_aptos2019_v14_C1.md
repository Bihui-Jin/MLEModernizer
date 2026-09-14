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

# 5. Target score

0.0313057905882835

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import gc
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    _CPU = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _CPU))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None

    if img.ndim == 2:
        mask = (img > tol).astype(np.uint8)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return img
        x, y, w, h = cv2.boundingRect(pts)
        return img[y : y + h, x : x + w]

    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = (gray_img > tol).astype(np.uint8)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return img
        x, y, w, h = cv2.boundingRect(pts)
        return img[y : y + h, x : x + w, :]

    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


_TOL_INT = 7
_SIGMAX = 10.0


def _tf_crop_from_gray(img_u8, tol=_TOL_INT):
    gray = tf.image.rgb_to_grayscale(img_u8)  # uint8 -> uint8
    gray2 = tf.squeeze(gray, axis=-1)  # [H,W]
    mask = gray2 > tf.cast(tol, gray2.dtype)

    def _no_crop():
        return img_u8

    def _do_crop():
        ys = tf.where(tf.reduce_any(mask, axis=1))[:, 0]
        xs = tf.where(tf.reduce_any(mask, axis=0))[:, 0]
        y0 = tf.reduce_min(ys)
        y1 = tf.reduce_max(ys) + 1
        x0 = tf.reduce_min(xs)
        x1 = tf.reduce_max(xs) + 1
        return img_u8[y0:y1, x0:x1, :]

    return tf.cond(tf.reduce_any(mask), _do_crop, _no_crop)


def _gaussian_kernel2d(ksize: int, sigma: float, dtype=tf.float32):
    k = tf.cast(tf.range(-(ksize // 2), ksize // 2 + 1), dtype)
    xx, yy = tf.meshgrid(k, k)
    kernel = tf.exp(-(xx * xx + yy * yy) / (2.0 * tf.cast(sigma, dtype) ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel  # [ksize, ksize]


def _tf_gaussian_blur(img_f32, sigma: float):
    ksize = tf.cast(tf.maximum(3.0, 2.0 * tf.math.ceil(3.0 * sigma) + 1.0), tf.int32)
    ksize = tf.minimum(ksize, 101)  # safety cap
    kernel2d = _gaussian_kernel2d(
        int(ksize.numpy()) if tf.executing_eagerly() else 61, sigma, dtype=img_f32.dtype
    )
    if not tf.executing_eagerly():
        kernel2d = _gaussian_kernel2d(61, sigma, dtype=img_f32.dtype)

    kernel4d = kernel2d[:, :, tf.newaxis, tf.newaxis]  # [K,K,1,1]
    kernel4d = tf.tile(kernel4d, [1, 1, 3, 1])  # [K,K,3,1] depthwise per-channel
    img4 = img_f32[tf.newaxis, ...]  # [1,H,W,3]
    blurred = tf.nn.depthwise_conv2d(
        img4, kernel4d, strides=[1, 1, 1, 1], padding="SAME"
    )
    return blurred[0]


def _tf_load_ben_preprocess(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_png(bytes_, channels=3)  # uint8 RGB
    img = _tf_crop_from_gray(img, tol=_TOL_INT)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img_f = tf.cast(img, tf.float32)

    blur = _tf_gaussian_blur(img_f, sigma=_SIGMAX)
    img_f = 4.0 * img_f + (-4.0) * blur + 128.0
    img_f = tf.clip_by_value(img_f, 0.0, 255.0) * (1.0 / 255.0)
    img_f.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img_f




## === cell 2
EFF_MODEL_PATH = "../input/eff-b0-model/eff_b0_model"


def build_fallback_model():
    base = DenseNet121(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_input_base_dir():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.basename(c) == "data" and os.path.exists(
                os.path.join(c, "aptos2019-blindness-detection")
            ):
                return os.path.join(c, "aptos2019-blindness-detection")
            return c
    return "/kaggle/input/aptos2019-blindness-detection"


input_dir = get_input_base_dir()
train_csv_path = os.path.join(input_dir, "train.csv")
test_csv_path = os.path.join(input_dir, "test.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
train_img_dir = os.path.join(input_dir, "train_images")
test_img_dir = os.path.join(input_dir, "test_images")

_MAP_PARALLEL = tf.data.AUTOTUNE

if os.path.exists(EFF_MODEL_PATH):
    model = keras.models.load_model(EFF_MODEL_PATH, compile=False)
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception:
        pass
else:
    train_df = pd.read_csv(train_csv_path)
    trn_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
    )

    def make_dataset(df, images_dir, training, cache=False, cache_path=None):
        ids = df["id_code"].astype(str).values
        full_paths = np.char.add(
            np.char.add(np.char.add(images_dir, os.sep), ids), ".png"
        )
        labels_np = df["diagnosis"].values.astype("int64")

        ds = tf.data.Dataset.from_tensor_slices((full_paths, labels_np))

        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)

        if training:
            ds = ds.shuffle(
                min(int(df.shape[0]), 2048), seed=SEED, reshuffle_each_iteration=True
            )

        def _map(path, label):
            img = _tf_load_ben_preprocess(path)
            label.set_shape(())
            return img, label

        ds = ds.map(_map, num_parallel_calls=_MAP_PARALLEL, deterministic=True)

        if cache:
            ds = ds.cache()

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(
        trn_df,
        train_img_dir,
        training=True,
        cache=True,
        cache_path="train_cache.tfdata",
    )
    val_ds = make_dataset(
        val_df, train_img_dir, training=False, cache=True, cache_path="val_cache.tfdata"
    )

    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=trn_df["diagnosis"].values,
    )
    cw = {i: float(w) for i, w in enumerate(cw)}

    model = build_fallback_model()
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
        ),
        ModelCheckpoint(
            "fallback_best.keras", monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,  # unchanged core logic
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )
    if os.path.exists("fallback_best.keras"):
        model = keras.models.load_model("fallback_best.keras")

    del train_ds, val_ds, train_df, trn_df, val_df
    gc.collect()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/561361703.py in <cell line: 0>()
     94         return ds
     95 
---> 96     train_ds = make_dataset(
     97         trn_df,
     98         train_img_dir,

/tmp/ipykernel_11/561361703.py in make_dataset(df, images_dir, training, cache, cache_path)
     65         # Fix: robust string concatenation for numpy arrays
     66         full_paths = np.char.add(
---> 67             np.char.add(np.char.add(images_dir, os.sep), ids), ".png"
     68         )
     69         labels_np = df["diagnosis"].values.astype("int64")

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U52' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 3
test_csv = pd.read_csv(test_csv_path)
id_code = test_csv["id_code"].astype(str).values


def make_test_dataset(ids, images_dir):
    full_paths = np.char.add(
        np.char.add(np.char.add(images_dir, os.sep), ids.astype(str)), ".png"
    )

    ds = tf.data.Dataset.from_tensor_slices(full_paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    def _map(path):
        img = _tf_load_ben_preprocess(path)
        return img

    ds = ds.map(_map, num_parallel_calls=_MAP_PARALLEL, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(id_code, test_img_dir)

pred = model.predict(test_ds, verbose=0)
test_prediction = np.asarray(np.argmax(pred, axis=1), dtype="int64")

del test_ds, pred
gc.collect()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3985963326.py in <cell line: 0>()
     26 test_ds = make_test_dataset(id_code, test_img_dir)
     27 
---> 28 pred = model.predict(test_ds, verbose=0)
     29 test_prediction = np.asarray(np.argmax(pred, axis=1), dtype="int64")
     30 

NameError: name 'model' is not defined

## === cell 4
sub = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction.astype("int64")})
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print(tmp)
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3766998422.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction.astype("int64")})
      2 sub.to_csv("submission.csv", index=False)
      3 
      4 unique, counts = np.unique(test_prediction, return_counts=True)
      5 tmp = dict(zip(unique.tolist(), counts.tolist()))

NameError: name 'test_prediction' is not defined

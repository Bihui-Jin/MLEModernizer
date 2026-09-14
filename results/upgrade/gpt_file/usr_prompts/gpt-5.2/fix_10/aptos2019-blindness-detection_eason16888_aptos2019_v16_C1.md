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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    cv2.setNumThreads(min(4, max(1, os.cpu_count() or 1)))
except Exception:
    pass

BASE_DIR_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
BASE_DIR = None
for p in BASE_DIR_CANDIDATES:
    if os.path.exists(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory in expected locations."
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV:", TEST_CSV)
print("Train images exist:", os.path.exists(TRAIN_IMG_DIR), "->", TRAIN_IMG_DIR)
print("Test images exist:", os.path.exists(TEST_IMG_DIR), "->", TEST_IMG_DIR)

"""
Config + preprocessing
"""
IMG_SIZE = 224
BATCH_SIZE = 16
N_CLASSES = 5
EPOCHS = 6  # unchanged


def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
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
        return np.stack([img1, img2, img3], axis=-1)
    return img


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def _preprocess_path_numpy(path_bytes):
    path = path_bytes.decode("utf-8")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img)  # returns float32 in [0,1], shape (IMG_SIZE, IMG_SIZE, 3)
    return img


def tf_load_and_preprocess(path):
    img = tf.numpy_function(_preprocess_path_numpy, [path], Tout=tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


_rand_rotate = tf.keras.layers.RandomRotation(
    factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
)


def tf_augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED + 1)

    img = tf.image.random_brightness(img, max_delta=0.2, seed=SEED + 2)
    img = tf.clip_by_value(img, 0.0, 1.0)

    img = _rand_rotate(img, training=True)
    img = tf.clip_by_value(img, 0.0, 1.0)

    zoom = tf.random.uniform([], minval=0.9, maxval=1.0, seed=SEED + 4)
    new_size = tf.cast(tf.round(zoom * IMG_SIZE), tf.int32)
    new_size = tf.maximum(new_size, 1)
    img = tf.image.random_crop(img, size=[new_size, new_size, 3], seed=SEED + 5)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def make_dataset(
    paths, labels_onehot=None, training=False, cache=False, cache_path=None
):
    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve deterministic behavior

    if labels_onehot is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(tf_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))
        ds = ds.map(
            lambda p, y: (tf_load_and_preprocess(p), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.with_options(options)
    ds = ds.ignore_errors()

    if cache:
        if cache_path is not None:
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 2048), seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.map(
            lambda x, y: (tf_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 1
"""
Load CSVs and build input pipelines.
Core training objective unchanged (DenseNet121 transfer learning classifier).
"""
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert "id_code" in train_df.columns and "diagnosis" in train_df.columns
assert "id_code" in test_df.columns
assert list(sample_sub.columns) == ["id_code", "diagnosis"]

train_df = train_df.copy()
train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df = test_df.copy()
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_split, val_split = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"],
)

train_paths = (TRAIN_IMG_DIR + "/" + train_split["filename"].values).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + val_split["filename"].values).astype(str)

test_df = test_df.reset_index(drop=True)
test_paths = (TEST_IMG_DIR + "/" + test_df["filename"].values).astype(str)

y_train = tf.keras.utils.to_categorical(
    train_split["diagnosis"].values, num_classes=N_CLASSES
)
y_val = tf.keras.utils.to_categorical(
    val_split["diagnosis"].values, num_classes=N_CLASSES
)

cache_dir = "./tfdata_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, "train_preproc.cache")
val_cache_path = os.path.join(cache_dir, "val_preproc.cache")
test_cache_path = os.path.join(cache_dir, "test_preproc.cache")

train_ds = make_dataset(
    train_paths, y_train, training=True, cache=True, cache_path=train_cache_path
)
val_ds = make_dataset(
    val_paths, y_val, training=False, cache=True, cache_path=val_cache_path
)
test_ds = make_dataset(
    test_paths,
    labels_onehot=None,
    training=False,
    cache=True,
    cache_path=test_cache_path,
)

classes_sorted = np.array(sorted(train_df["diagnosis"].unique()))
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=classes_sorted,
    y=train_split["diagnosis"].values,
)
class_weights = {int(cls): float(w) for cls, w in zip(classes_sorted, cw)}
print("class_weights:", class_weights)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))



## === cell 2
"""
Model definition: DenseNet121 backbone + GAP + Dropout + Dense softmax (5 classes).
"""
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.5)(x)
out = Dense(N_CLASSES, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "best_model.keras"
callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
    ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
]

model.summary()



## === cell 3
"""
Train
"""
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

model = keras.models.load_model(ckpt_path, compile=False)

gc.collect()



## === cell 4
"""
Predict on test and write submission.csv with required columns.
"""
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

submission = sample_sub.copy()
submission["id_code"] = test_df["id_code"].values
submission["diagnosis"] = pred_labels.astype(int)

assert submission.shape[0] == test_df.shape[0], (submission.shape, test_df.shape)
assert list(submission.columns) == ["id_code", "diagnosis"]
assert submission["diagnosis"].between(0, 4).all()

submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(pred_labels, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))

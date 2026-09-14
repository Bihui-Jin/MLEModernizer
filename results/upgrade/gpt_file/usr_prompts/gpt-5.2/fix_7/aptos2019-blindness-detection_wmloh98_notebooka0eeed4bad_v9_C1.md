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

3.9

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

0.0577323120784064

# 6. Current score

-0.02392

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.02392) has done: 'The timeout is overwhelmingly driven by slow image input: Keras’ `ImageDataGenerator.flow_from_dataframe` uses Python/PIL loading and single-process augmentation, which becomes the bottleneck even for 2 epochs. I keep the exact same model, loss, epochs, and augmentation semantics, but switch data feeding to a `tf.data` pipeline that performs the same decode/resize/rescale and the same augmentations (rotation + horizontal flip) inside TensorFlow with parallelism, prefetching, and deterministic seeding. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting (it forces a slower pure-Python protobuf path) to speed up TF startup/runtime without affecting results. Finally, I compute `steps_per_epoch` explicitly to avoid any generator bookkeeping overhead and ensure the pipeline runs efficiently end-to-end.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train_images")
TEST_IMG_DIR = os.path.join(DATA_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_files = set(os.listdir(TRAIN_IMG_DIR))
test_files = set(os.listdir(TEST_IMG_DIR))

train_df = train_df[train_df["filename"].isin(train_files)].reset_index(drop=True)
test_df = test_df[test_df["filename"].isin(test_files)].reset_index(drop=True)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
train_df.head()




## === cell 1
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
val_frac = 0.15

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:n_val].copy()
tr_df = train_df_shuf.iloc[n_val:].copy()

num_classes = 5
print("num_classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE


def _build_paths_and_labels(df, img_dir, has_labels: bool):
    paths = (img_dir + "/" + df["filename"].values).astype(str)
    if has_labels:
        labels = df["diagnosis"].values.astype(np.int32)
        return paths, labels
    return paths, None


def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _one_hot(label):
    return tf.one_hot(label, depth=num_classes, dtype=tf.float32)


_MAX_DEG = 10.0
_PI = tf.constant(np.pi, dtype=tf.float32)


def _augment(img, seed):
    seed1 = tf.random.experimental.stateless_split(seed, 1)[0]
    angle = tf.random.stateless_uniform(
        [], seed=seed1, minval=-_MAX_DEG, maxval=_MAX_DEG, dtype=tf.float32
    ) * (_PI / 180.0)
    img = tf.image.rotate(img, angle, interpolation="BILINEAR")
    seed2 = tf.random.experimental.stateless_split(seed1, 1)[0]
    do_flip = (
        tf.random.stateless_uniform(
            [], seed=seed2, minval=0.0, maxval=1.0, dtype=tf.float32
        )
        < 0.5
    )
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    return img


def make_train_ds(df):
    paths, labels = _build_paths_and_labels(df, TRAIN_IMG_DIR, has_labels=True)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_resize_rescale(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(h, tf.int64)])
        img = _augment(img, seed)
        return img, _one_hot(label)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df):
    paths, labels = _build_paths_and_labels(df, TRAIN_IMG_DIR, has_labels=True)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_resize_rescale(path)
        return img, _one_hot(label)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(tr_df)
val_ds = make_val_ds(val_df)

steps_per_epoch = int(np.ceil(len(tr_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1974298318.py in <cell line: 0>()
     25 # Speed fix: feed tf.data pipeline (parallel + prefetch) and pass explicit steps to reduce overhead.
     26 history = model.fit(
---> 27     train_ds,
     28     validation_data=val_ds,
     29     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 3
def make_test_ds(df):
    paths, _ = _build_paths_and_labels(df, TEST_IMG_DIR, has_labels=False)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_resize_rescale(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_df)
test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

pred_proba = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
pred = np.argmax(pred_proba, axis=1).astype(int)

print(pred.shape, pred[:10])




## === cell 4
submission_df = pd.read_csv(TEST_CSV)
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

pred_map = dict(zip(test_df["filename"].tolist(), pred.tolist()))
submission_df["diagnosis"] = (
    submission_df["filename"].map(pred_map).fillna(0).astype(int)
)

submission_df = submission_df[["id_code", "diagnosis"]]
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(submission_df.head())
print("Saved:", submission_path, "rows:", len(submission_df))




## === cell 5
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["id_code", "diagnosis"]
assert len(sub_check) == len(pd.read_csv(TEST_CSV))
print(sub_check["diagnosis"].value_counts().sort_index())




## === cell 6
from collections import Counter

cnt = Counter(sub_check["diagnosis"].tolist())
cnt

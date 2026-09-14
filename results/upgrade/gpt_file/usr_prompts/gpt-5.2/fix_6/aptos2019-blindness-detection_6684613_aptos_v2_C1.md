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

0.8684391833613058

# 6. Current score

0.04908

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.04908) has done: 'The timeout is dominated by slow Python-side image loading/augmentation via `ImageDataGenerator.flow_from_dataframe` and a non-parallel `Sequence` for test inference. I keep the exact same model, loss, epochs, and augmentation semantics, but move the input pipeline to `tf.data` with parallel decode/resize, prefetch, and (for training) stateless per-example augmentation that matches the same transforms. I also avoid OpenCV in the test path and use the same `tf.io` decode pipeline for faster, parallelized inference. These changes are computationally equivalent (same images, same preprocess function, same augmentation types/ranges) but remove a major single-threaded bottleneck so training+inference can finish under 600 seconds.'

# 9. Code solution

## === cell 0
import os

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"

BS = 16
IMG_SIZE = 300
SIZE = (IMG_SIZE, IMG_SIZE)

NUM_CLASSES = 5

print("Using BASE_INPUT:", BASE_INPUT)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)

df["name"] = df["id_code"].astype(str) + ".png"
df["diagnosis_int"] = df["diagnosis"].astype(int)
df["diagnosis_str"] = df["diagnosis_int"].astype(str)
df["name"] = df["name"].astype(str)

print(df.head())
print(df.dtypes)




## === cell 2
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=SEED,
    shuffle=True,
    stratify=df["diagnosis_int"],
)



def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    return img


def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


def _get_transform_matrix(angle_rad, tx, ty, zx, zy):
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    a0 = cos_a / zx
    a1 = -sin_a / zx
    a2 = tx
    b0 = sin_a / zy
    b1 = cos_a / zy
    b2 = ty
    return tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])


def _apply_affine(img, transform):
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="NEAREST",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(img, 0)


def _augment(img, seed_pair):
    s0, s1 = seed_pair[0], seed_pair[1]

    angle = tf.random.stateless_uniform(
        [], seed=[s0, s1], minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    s0 += 1
    tx = tf.random.stateless_uniform(
        [], seed=[s0, s1], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)
    s0 += 1
    ty = tf.random.stateless_uniform(
        [], seed=[s0, s1], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)
    s0 += 1
    zoom = tf.random.stateless_uniform([], seed=[s0, s1], minval=0.9, maxval=1.1)
    s0 += 1
    transform = _get_transform_matrix(angle, tx, ty, zoom, zoom)
    img = _apply_affine(img, transform)

    do_flip = (
        tf.random.stateless_uniform([], seed=[s0, s1], minval=0.0, maxval=1.0) < 0.5
    )
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    return img


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(idx, data):
        path, y = data
        img = _read_decode_resize(path)
        img = _augment(
            img,
            seed_pair=tf.stack(
                [tf.cast(idx, tf.int32) + SEED, tf.cast(SEED, tf.int32)]
            ),
        )
        img = _preprocess(img)
        return img, y

    ds = ds.enumerate()
    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, y):
        img = _read_decode_resize(path)
        img = _preprocess(img)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths = (TRAIN_DIR + os.sep + train_df["name"].values.astype(str)).tolist()
val_paths = (TRAIN_DIR + os.sep + val_df["name"].values.astype(str)).tolist()
train_labels = train_df["diagnosis_int"].values.astype(np.int32)
val_labels = val_df["diagnosis_int"].values.astype(np.int32)

train_ds = make_train_ds(train_paths, train_labels)
val_ds = make_val_ds(val_paths, val_labels)

train_steps = int(np.ceil(len(train_df) / BS))
val_steps = int(np.ceil(len(val_df) / BS))
print("train steps:", train_steps, "val steps:", val_steps)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2686536567.py in <cell line: 0>()
    137 
    138 
--> 139 train_paths = (TRAIN_DIR + os.sep + train_df["name"].values.astype(str)).tolist()
    140 val_paths = (TRAIN_DIR + os.sep + val_df["name"].values.astype(str)).tolist()
    141 train_labels = train_df["diagnosis_int"].values.astype(np.int32)

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U52'), dtype('<U16')) -> None

## === cell 3
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    "bestmodel.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    mode="min",
    verbose=1,
)




## === cell 4
base = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 8

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[checkpoint, reduce_lr],
    verbose=1,
)

model = tf.keras.models.load_model("bestmodel.h5", compile=False)
assert model is not None




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427471490.py in <cell line: 0>()
     27 # Speed: using tf.data eliminates Python generator overhead; keep epochs/callbacks identical.
     28 history = model.fit(
---> 29     train_ds,
     30     validation_data=val_ds,
     31     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 5
sub_test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

sub_test = sub_test.copy()
sample_sub = sample_sub.copy()
assert len(sub_test) == len(
    sample_sub
), "test.csv and sample_submission.csv row count mismatch"

test_ids = sub_test["id_code"].astype(str).values
test_paths = (TEST_DIR + os.sep + pd.Series(test_ids).astype(str) + ".png").tolist()


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _read_decode_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths)

preds = model.predict(test_ds, verbose=0)
results = preds.argmax(axis=1).astype(int).tolist()

sample_sub["id_code"] = sub_test["id_code"].values
sample_sub["diagnosis"] = results

assert sample_sub.shape[0] == len(
    sub_test
), f"Unexpected submission rows: {sample_sub.shape[0]}"
assert list(sample_sub.columns) == ["id_code", "diagnosis"]

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())

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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
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

DS_OPTIONS = tf.data.Options()
DS_OPTIONS.deterministic = True
try:
    DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
    DS_OPTIONS.experimental_optimization.map_parallelization = True
    DS_OPTIONS.experimental_optimization.autotune_buffers = True
    DS_OPTIONS.experimental_slack = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HAS_GPU = bool(tf.config.list_physical_devices("GPU"))
print("GPU available:", HAS_GPU)




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


@tf.function(reduce_retracing=True)
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _decode_map(path, y):
    img = _read_decode_resize(path)
    return img, y


@tf.function(reduce_retracing=True)
def _val_map_fn(path, y):
    img = _read_decode_resize(path)
    return img, y


def _maybe_prefetch_to_device(ds):
    if HAS_GPU:
        try:
            return ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            return ds
    return ds


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    shuffle_buf = min(int(len(paths)), 2048)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)


    ds = ds.with_options(DS_OPTIONS)
    return ds


def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(_val_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DS_OPTIONS)
    return ds


train_paths = (TRAIN_DIR + os.sep + train_df["name"].astype(str).to_numpy()).astype(str)
val_paths = (TRAIN_DIR + os.sep + val_df["name"].astype(str).to_numpy()).astype(str)
train_labels = train_df["diagnosis_int"].to_numpy(dtype=np.int32)
val_labels = val_df["diagnosis_int"].to_numpy(dtype=np.int32)

train_ds = make_train_ds(train_paths, train_labels)
val_ds = make_val_ds(val_paths, val_labels)

train_steps = int(np.ceil(len(train_df) / BS))
val_steps = int(np.ceil(len(val_df) / BS))
print("train steps:", train_steps, "val steps:", val_steps)




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

x = tf.keras.layers.RandomRotation(factor=10.0 / 180.0, fill_mode="reflect", seed=SEED)(
    inputs
)
x = tf.keras.layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
)(x)
x = tf.keras.layers.RandomZoom(
    height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", seed=SEED
)(x)
x = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED)(x)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base(x, training=False)
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
    jit_compile=True,
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




## === cell 5
sub_test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

sub_test = sub_test.copy()
sample_sub = sample_sub.copy()
assert len(sub_test) == len(
    sample_sub
), "test.csv and sample_submission.csv row count mismatch"

test_ids = sub_test["id_code"].astype(str).to_numpy()
test_paths = (TEST_DIR + os.sep + test_ids + ".png").astype(str)


@tf.function(reduce_retracing=True)
def _test_map_fn(path):
    img = _read_decode_resize(path)
    return img


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.map(_test_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BS, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DS_OPTIONS)
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

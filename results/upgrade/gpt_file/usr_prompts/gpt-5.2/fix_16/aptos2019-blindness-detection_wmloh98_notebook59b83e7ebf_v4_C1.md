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

0.5341472935256788

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We fix the environment-breaking import error by avoiding the standalone `keras` backend (which triggers the `MessageFactory` protobuf issue) and using `tf.keras.backend` consistently. Since the referenced pre-trained model file doesn’t exist in your input folders, we replace the missing `load_model()` step with a minimal training/inference pipeline that uses the same image preprocessing and produces valid 5-class predictions for the required submission format. We also fix deprecated/removed APIs (`predict_generator`) and ensure paths point to the provided dataset directory so the generator can find images. Finally, we always write `submission.csv` with columns `id_code,diagnosis` and the correct row order.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe`, plus OpenCV resizing repeated for every epoch. To keep the exact same model and training loop semantics, I replace the Keras generator with an equivalent `tf.data` pipeline that performs the same preprocessing (decode → resize to 256×256 → rescale) and the same augmentations (rotation/flip/zoom ranges) but runs in the TensorFlow graph with parallelism, caching, and prefetch. I also cache the *decoded+resized* images (before random augmentation) so epochs 2–3 don’t repeatedly decode/resize ~3k PNGs, while keeping augmentations random each epoch. Predictions are similarly sped up with `tf.data` + prefetch.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_PATH = "/kaggle/input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32

assert os.path.exists(
    os.path.join(DATA_PATH, "train.csv")
), "train.csv not found at DATA_PATH"
assert os.path.exists(
    os.path.join(DATA_PATH, "test.csv")
), "test.csv not found at DATA_PATH"
assert os.path.isdir(
    os.path.join(DATA_PATH, "train_images")
), "train_images dir not found"
assert os.path.isdir(
    os.path.join(DATA_PATH, "test_images")
), "test_images dir not found"

print("TensorFlow version:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def create_kappa_loss(bsize, eps=1e-10, N=5):
    repeat_op = tf.cast(
        tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype=tf.float32
    )
    weights_const = tf.square(repeat_op - tf.transpose(repeat_op)) / tf.cast(
        (N - 1) ** 2, dtype=tf.float32
    )

    @tf.function
    def kappa_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        pred_ = tf.square(y_pred)
        pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, axis=1), [-1, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, axis=0)
        hist_rater_b = tf.reduce_sum(y_true, axis=0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
        nom = tf.reduce_sum(weights_const * conf_mat)

        b = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        denom = tf.reduce_sum(
            weights_const
            * tf.matmul(
                tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])
            )
            / (b + eps)
        )
        return nom / (denom + eps)

    return kappa_loss


KAPPA_LOSS = create_kappa_loss(BATCH_SIZE)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_model(input_shape=(DIM_Y, DIM_X, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


model = build_model()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 2
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"
train_df["diagnosis"] = train_df["diagnosis"].astype(np.int32)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("Train/valid sizes:", len(tr_df), len(va_df))

AUTOTUNE = tf.data.AUTOTUNE

train_dir = os.path.join(DATA_PATH, "train_images")
test_dir = os.path.join(DATA_PATH, "test_images")

num_classes = 5

ROTATION_RANGE_DEG = 10.0
ZOOM_RANGE = 0.05
HFLIP = True
VFLIP = True

augmenter = tf.keras.Sequential(
    [
        layers.RandomRotation(
            factor=ROTATION_RANGE_DEG / 360.0, fill_mode="reflect", seed=SEED
        ),
        (
            layers.RandomFlip(
                mode=(
                    "horizontal_and_vertical"
                    if (HFLIP and VFLIP)
                    else "horizontal" if HFLIP else "vertical" if VFLIP else None
                ),
                seed=SEED,
            )
            if (HFLIP or VFLIP)
            else layers.Lambda(lambda x: x)
        ),
        layers.RandomZoom(
            height_factor=(-ZOOM_RANGE, ZOOM_RANGE),
            width_factor=(-ZOOM_RANGE, ZOOM_RANGE),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


@tf.function
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [DIM_Y, DIM_X], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _with_ds_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.autotune.enabled = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return ds.with_options(opts)


def make_train_ds(df, batch_size):
    paths = (train_dir + "/" + df["filename"].values.astype(str)).astype(np.str_)
    labels = df["diagnosis"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: (_read_decode_resize(p), y), num_parallel_calls=AUTOTUNE)

    ds = ds.cache(os.path.join("/kaggle/working", "cache_train_decode_resized.tf-data"))

    @tf.function
    def _aug_map(img, y):
        img = augmenter(img, training=True)
        y_oh = tf.one_hot(y, depth=num_classes, dtype=tf.float32)
        return img, y_oh

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_ds_options(ds)
    return ds


def make_valid_ds(df, batch_size):
    paths = (train_dir + "/" + df["filename"].values.astype(str)).astype(np.str_)
    labels = df["diagnosis"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    @tf.function
    def _val_map(p, y):
        return _read_decode_resize(p), tf.one_hot(
            y, depth=num_classes, dtype=tf.float32
        )

    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE)

    ds = ds.cache(os.path.join("/kaggle/working", "cache_valid_decode_resized.tf-data"))

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_ds_options(ds)
    return ds


train_ds = make_train_ds(tr_df, BATCH_SIZE)
valid_ds = make_valid_ds(va_df, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(tr_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_df) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1488813988.py in <cell line: 0>()
    130 
    131 
--> 132 train_ds = make_train_ds(tr_df, BATCH_SIZE)
    133 valid_ds = make_valid_ds(va_df, BATCH_SIZE)
    134 

/tmp/ipykernel_11/1488813988.py in make_train_ds(df, batch_size)
     81 def make_train_ds(df, batch_size):
     82     # Speed: avoid pandas .map(lambda ...) Python overhead; vectorized string concatenation.
---> 83     paths = (train_dir + "/" + df["filename"].values.astype(str)).astype(np.str_)
     84     labels = df["diagnosis"].values.astype(np.int32)
     85 

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U57'), dtype('<U16')) -> None

## === cell 3
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/277675732.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds,
      5     validation_data=valid_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
def make_test_ds(df, batch_size):
    paths = (test_dir + "/" + df["filename"].values.astype(str)).astype(np.str_)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join("/kaggle/working", "cache_test_decode_resized.tf-data"))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_ds_options(ds)
    return ds


test_ds = make_test_ds(test_df, BATCH_SIZE)
test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

pred_proba = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)

pred = np.argmax(pred_proba, axis=1).astype(int)

print(
    "Pred shape:",
    pred.shape,
    "Unique:",
    pd.Series(pred).value_counts().sort_index().to_dict(),
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1122805176.py in <cell line: 0>()
     12 
     13 
---> 14 test_ds = make_test_ds(test_df, BATCH_SIZE)
     15 test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))
     16 

/tmp/ipykernel_11/1122805176.py in make_test_ds(df, batch_size)
      1 def make_test_ds(df, batch_size):
      2     # Speed: avoid pandas .map(lambda ...) Python overhead; vectorized string concatenation.
----> 3     paths = (test_dir + "/" + df["filename"].values.astype(str)).astype(np.str_)
      4     ds = tf.data.Dataset.from_tensor_slices(paths)
      5     ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U56'), dtype('<U16')) -> None

## === cell 5
submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

pred_map = dict(zip(test_df["id_code"].values, pred))
submission["diagnosis"] = submission["id_code"].map(pred_map)

submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
submission = submission[["id_code", "diagnosis"]]

assert (
    submission.shape[0]
    == pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv")).shape[0]
)
assert list(submission.columns) == ["id_code", "diagnosis"]

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Unique predictions:", submission["diagnosis"].value_counts().sort_index().to_dict()
)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3080529078.py in <cell line: 0>()
      1 submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
      2 
----> 3 pred_map = dict(zip(test_df["id_code"].values, pred))
      4 submission["diagnosis"] = submission["id_code"].map(pred_map)
      5 

NameError: name 'pred' is not defined

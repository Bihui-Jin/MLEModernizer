# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB7

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    if hasattr(tf.data.experimental, "enable_debug_mode"):
        pass
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

print("TF version:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.data.experimental.enable_shared_memory()  # no-op if unavailable
except Exception:
    pass



## === cell 1
df = pd.read_csv(TRAIN_CSV)
df.head()



## === cell 2
df["labels"] = df["labels"].astype("category")
df["label_num"] = df["labels"].cat.codes

class_map = dict(sorted(df[["label_num", "labels"]].values.tolist()))
inv_class_map = {v: k for k, v in class_map.items()}

num_classes = df["label_num"].nunique()
print("Num training rows:", len(df))
print("Num classes:", num_classes)
df[["image", "labels", "label_num"]].head()



## === cell 3
submission = pd.read_csv(SAMPLE_SUB)
submission.head()



## === cell 4
IMG_SIZE = 600
BATCH_SIZE = 4  # keep small to fit GPU/CPU memory with 600x600 EfficientNetB7

df = df.copy()
df["filepath"] = TRAIN_DIR.rstrip("/") + "/" + df["image"].astype(str)

val_frac = 0.1
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_size = int(len(df) * val_frac)
df_val = df.iloc[:val_size].reset_index(drop=True)
df_train = df.iloc[val_size:].reset_index(drop=True)


@tf.function
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


rand_rot = tf.keras.layers.RandomRotation(
    factor=15.0 / 180.0, fill_mode="reflect", interpolation="bilinear", seed=SEED
)
rand_trans = tf.keras.layers.RandomTranslation(
    height_factor=0.05,
    width_factor=0.05,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=SEED,
)
rand_zoom = tf.keras.layers.RandomZoom(
    height_factor=(-0.1, 0.1),
    width_factor=(-0.1, 0.1),
    fill_mode="reflect",
    interpolation="bilinear",
    seed=SEED,
)
rand_flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED)


@tf.function
def _augment_with_path_seed(img, y, path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)

    img = img[None, ...]
    img = rand_rot(img, training=True, seed=seed)
    img = rand_zoom(img, training=True, seed=seed)
    img = rand_trans(img, training=True, seed=seed)
    img = rand_flip(img, training=True, seed=seed)
    img = tf.squeeze(img, axis=0)

    return img, y


def _make_train_ds(paths, labels):
    paths = tf.convert_to_tensor(paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    shuffle_buf = min(len(df_train), 2048)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: (_decode_and_resize(p), y, p), num_parallel_calls=AUTOTUNE)

    ds = ds.map(
        lambda img, y, p: _augment_with_path_seed(img, y, p),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.map(
        lambda img, y: (img, tf.one_hot(y, depth=num_classes, dtype=tf.float32)),
        num_parallel_calls=AUTOTUNE,
    )

    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # allow parallelism for speed; seeds still deterministic per-example
    )
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels):
    paths = tf.convert_to_tensor(paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTOTUNE)

    ds = ds.map(
        lambda img, y: (img, tf.one_hot(y, depth=num_classes, dtype=tf.float32)),
        num_parallel_calls=AUTOTUNE,
    )

    options = tf.data.Options()
    options.experimental_deterministic = (
        True  # keep deterministic validation for stable metrics
    )
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(df_train["filepath"].values, df_train["label_num"].values)
val_ds = _make_val_ds(df_val["filepath"].values, df_val["label_num"].values)

train_steps = len(df_train) // BATCH_SIZE  # drop_remainder=True
val_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

print("Train steps/epoch:", train_steps)
print("Val steps:", val_steps)



## === cell 5
base = EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "best_model.keras"
callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
]

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)

model = tf.keras.models.load_model(ckpt_path)



## === cell 6
test_paths = (TEST_DIR.rstrip("/") + "/" + submission["image"].astype(str)).values
pred_nums = np.zeros(len(test_paths), dtype=np.int32)


def _make_test_ds(paths):
    paths = tf.convert_to_tensor(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.batch(
        32, drop_remainder=False
    )  # larger batch for inference; does not change predictions.
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)
probs = model.predict(test_ds, verbose=0)
pred_nums = np.argmax(probs, axis=1).astype(np.int32)

print("Pred length:", len(pred_nums), "Submission length:", len(submission))



## === cell 7
submission_result = submission.copy()
submission_result["labels"] = pd.Series(pred_nums).map(class_map)

if submission_result["labels"].isna().any():
    most_common_label = df["labels"].value_counts().index[0]
    submission_result["labels"] = submission_result["labels"].fillna(most_common_label)

submission_result.to_csv("submission.csv", index=False)
print(submission_result.head())
print("Wrote submission.csv with shape:", submission_result.shape)



## === cell 8
print("Competition Complete!!")

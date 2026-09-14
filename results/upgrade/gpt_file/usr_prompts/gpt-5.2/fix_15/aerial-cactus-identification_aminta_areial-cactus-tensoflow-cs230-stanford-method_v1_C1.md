# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import json
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn.utils

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    DepthwiseConv2D,
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

print("Python/TensorFlow versions:", tf.__version__)



## === cell 1
BASE_INPUT_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input",
    "/kaggle/input",
]

DATA_ROOT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        if os.path.basename(p) in ["input"] and os.path.exists(
            os.path.join(p, "aerial-cactus-identification")
        ):
            DATA_ROOT = os.path.join(p, "aerial-cactus-identification")
            break
        if os.path.exists(os.path.join(p, "train.csv")) and (
            os.path.exists(os.path.join(p, "train"))
            or os.path.exists(os.path.join(p, "train.zip"))
        ):
            DATA_ROOT = p
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the aerial-cactus-identification dataset directory."
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("train_csv_path:", train_csv_path)
print("train_dir:", train_dir)
print("test_dir:", test_dir)
print("sample_sub_path:", sample_sub_path)

assert os.path.isfile(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.isdir(train_dir), f"Missing train/ directory at {train_dir}"
assert os.path.isdir(test_dir), f"Missing test/ directory at {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"



## === cell 2
SEED = 1372
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    _cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(4, _cpu_count))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))




## === cell 3
class Params:
    """Class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)




## === cell 4
df = pd.read_csv(train_csv_path)
df["filepath"] = (train_dir.rstrip("/") + "/" + df["id"].astype(str)).values
df = sklearn.utils.shuffle(df, random_state=SEED).reset_index(drop=True)

filenames = df["filepath"].values
labels = df["has_cactus"].astype(np.float32).values

print("sample filename:", filenames[0])
print("sample label:", labels[0], type(labels[0]))




## === cell 5
@tf.function
def _decode_and_resize_from_bytes(image_bytes, size):
    image_decoded = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return tf.image.resize(image, [size, size])


@tf.function
def train_preprocess_stateless(image, label, use_random_flip, seed2):
    if use_random_flip:
        image = tf.image.stateless_random_flip_left_right(image, seed=seed2)
    s_b = seed2 + tf.constant([1, 0], dtype=seed2.dtype)
    s_s = seed2 + tf.constant([0, 1], dtype=seed2.dtype)
    image = tf.image.stateless_random_brightness(
        image, max_delta=32.0 / 255.0, seed=s_b
    )
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=s_s)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


@tf.function
def _read_decode_resize(filename, label, image_size):
    img_bytes = tf.io.read_file(filename)
    img = _decode_and_resize_from_bytes(img_bytes, image_size)
    return img, label


@tf.function
def _read_decode_resize_only(filename, image_size):
    img_bytes = tf.io.read_file(filename)
    return _decode_and_resize_from_bytes(img_bytes, image_size)


@tf.function
def _seed_from_filename(filename):
    h = tf.strings.to_hash_bucket_fast(filename, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int64), tf.cast(h, tf.int64)], axis=0)


def input_fn(is_training, filenames, labels, params, cache_path=None):
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_slack = True
    except Exception:
        pass

    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

    image_size = tf.constant(params.image_size, dtype=tf.int32)
    use_random_flip = tf.constant(bool(params.use_random_flip))

    def _maybe_cache(ds):
        if cache_path:
            return ds.cache(cache_path)
        return ds.cache()

    if is_training:

        def _load_with_fname(fname, y):
            img_bytes = tf.io.read_file(fname)
            img = _decode_and_resize_from_bytes(img_bytes, image_size)
            return img, y, fname

        dataset = dataset.map(
            _load_with_fname,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        dataset = _maybe_cache(dataset)

        dataset = dataset.shuffle(num_samples, seed=SEED, reshuffle_each_iteration=True)

        def _augment(img, y, fname):
            seed2 = _seed_from_filename(fname)
            img, y = train_preprocess_stateless(img, y, use_random_flip, seed2)
            return img, y

        dataset = dataset.map(
            _augment,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        dataset = dataset.batch(params.batch_size, drop_remainder=True).repeat()
        if _HAS_GPU:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(
            lambda f, y: _read_decode_resize(f, y, image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset = _maybe_cache(dataset)
        dataset = dataset.batch(params.batch_size, drop_remainder=False)
        if _HAS_GPU:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset




## === cell 6
with open("params.json", "w") as text_file:
    text_file.write(
        "{\n"
        '"learning_rate": 1e-3,'
        '"batch_size": 32,'
        '"num_epochs": 10,'
        '"image_size": 32,'
        '"use_random_flip": true,'
        '"num_labels": 2,'
        '"num_parallel_calls": 4,'
        '"save_summary_steps": 1'
        "\n}"
    )

json_path = os.path.join("./", "params.json")
assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
params = Params(json_path)



## === cell 7
split = int(len(filenames) * 0.15)
train_filenames, valid_filenames = filenames[split:], filenames[:split]
train_labels, valid_labels = labels[split:], labels[:split]

train_cache_path = os.path.join(".", "tfdata_cache_train")
valid_cache_path = os.path.join(".", "tfdata_cache_valid")

for _p in (train_cache_path, valid_cache_path):
    try:
        if os.path.exists(_p):
            os.remove(_p)
    except Exception:
        pass

train_dataset = input_fn(
    True, train_filenames, train_labels, params, cache_path=train_cache_path
)
valid_dataset = input_fn(
    False, valid_filenames, valid_labels, params, cache_path=valid_cache_path
)

print("Train samples:", len(train_filenames), "Valid samples:", len(valid_filenames))



## === cell 8
model = Sequential()

model.add(Conv2D(3, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))

model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="Same", use_bias=True))
model.add(Conv2D(filters=32, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="Same", use_bias=True))
model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="Same", use_bias=True))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="Same", use_bias=True))
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="Same", use_bias=True))
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))



## === cell 9
opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

model.compile(
    optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"], jit_compile=False
)
model.summary()



## === cell 10
file_path = "weights-aerial-cactus.h5"

callbacks = [
    ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min", min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=1e-10,
        patience=15,
        verbose=1,
        restore_best_weights=True,
    ),
]



## === cell 11
steps_per_epoch = len(train_filenames) // params.batch_size  # drop_remainder=True
validation_steps = (len(valid_filenames) + params.batch_size - 1) // params.batch_size

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=int(params.num_epochs),
    verbose=True,
    steps_per_epoch=int(steps_per_epoch),
    validation_steps=int(validation_steps),
    callbacks=callbacks,
)



## === cell 12
if os.path.exists(file_path):
    model.load_weights(file_path)
else:
    model.save_weights(file_path)




## === cell 13
def plot_training_curves(history):
    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    acc = history.history.get(acc_key, [])
    val_acc = history.history.get(val_acc_key, [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Losses")
    plt.legend()

    plt.figure()
    plt.plot(epochs, acc, "r", label="Training acc")
    plt.plot(epochs, val_acc, "g", label="Validation acc")
    plt.title("Accuracies")
    plt.legend()

    plt.show()




## === cell 14
test_df = pd.read_csv(sample_sub_path)
images_test = test_df["id"].values

test_paths = (test_dir.rstrip("/") + "/" + test_df["id"].astype(str)).values

test_options = tf.data.Options()
test_options.experimental_deterministic = False
try:
    test_options.experimental_optimization.apply_default_optimizations = True
    test_options.experimental_optimization.map_parallelization = True
    test_options.experimental_optimization.parallel_batch = True
    test_options.experimental_slack = True
except Exception:
    pass

test_cache_path = os.path.join(".", "tfdata_cache_test")

try:
    if os.path.exists(test_cache_path):
        os.remove(test_cache_path)
except Exception:
    pass

image_size = tf.constant(params.image_size, dtype=tf.int32)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(test_options)
    .map(
        lambda f: _read_decode_resize_only(f, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    .cache(test_cache_path)
    .batch(params.batch_size, drop_remainder=False)
)

if _HAS_GPU:
    test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

y_test_pred = model.predict(test_ds, verbose=0).reshape(-1)
test_df["has_cactus"] = y_test_pred.astype(np.float32)

out_path = "submission.csv"
test_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", test_df.shape)
print(test_df.head())

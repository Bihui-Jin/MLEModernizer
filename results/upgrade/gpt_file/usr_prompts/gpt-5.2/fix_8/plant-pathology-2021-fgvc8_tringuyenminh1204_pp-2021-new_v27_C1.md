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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)



## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, test.shape, sub.shape)
print(train.head())



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not available:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not available:", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU config skipped:", e)



## === cell 3
print("Skipped matplotlib preview to save time.")



## === cell 4
import pathlib



## === cell 5
train_paths = [
    os.path.join(train_images_dir, img_id) for img_id in train["image"].tolist()
]
train_paths = sorted(train_paths)

test_paths = [
    os.path.join(test_images_dir, img_id) for img_id in test["image"].tolist()
]

print("n_train_paths:", len(train_paths))
print("n_test_paths:", len(test_paths))

sample_check = test_paths[:32]
missing_test = [p for p in sample_check if not tf.io.gfile.exists(p)]
print("missing_test_images_in_first_32:", len(missing_test))



## === cell 6
labs_series = train["labels"].astype(str).str.split()

flat_labels = (
    np.concatenate(
        [np.asarray(x, dtype=object) for x in labs_series.to_list() if len(x) > 0]
    )
    if len(labs_series)
    else np.array([], dtype=object)
)
classes = sorted(pd.unique(flat_labels).tolist()) if flat_labels.size else []
print("classes:", classes)
num_classes = len(classes)

class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}



## === cell 7
y = np.zeros((len(train), num_classes), dtype=np.float32)
if num_classes:
    labs_list = labs_series.to_list()
    for i, lst in enumerate(labs_list):
        if lst:
            y[i, [class_to_idx[lab] for lab in lst]] = 1.0

new_train = pd.concat(
    [train[["image", "labels"]], pd.DataFrame(y, columns=classes)], axis=1
)
new_train.head()



## === cell 8
new_train




## === cell 9
@tf.function
def decode_and_resize(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(
        bits, channels=3, try_recover_truncated=True, acceptable_fraction=0.75
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, image_size, method="bilinear")
    if label is None:
        return image
    return image, label




## === cell 10
test_paths[:5]



## === cell 11
BATCH_SIZE = 32  # keep as original
IMG_SIZE = (512, 512)



## === cell 12
test_options = tf.data.Options()
test_options.deterministic = True
try:
    test_options.experimental_optimization.map_parallelization = True
    test_options.experimental_optimization.map_and_batch_fusion = True
    test_options.experimental_optimization.map_and_filter_fusion = True
    test_options.experimental_optimization.parallel_batch = True
    test_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

test_paths_tf = tf.constant(test_paths)


def _map_test(p):
    return decode_and_resize(p, None, IMG_SIZE)


test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_tf)
    .with_options(test_options)
    .map(_map_test, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
from tensorflow import keras



## === cell 14
train_filepaths = [
    os.path.join(train_images_dir, img_id) for img_id in train["image"].tolist()
]

sample_check = train_filepaths[:64]
missing_train = [p for p in sample_check if not tf.io.gfile.exists(p)]
print("missing_train_images_in_first_64:", len(missing_train))



## === cell 15
idx = np.arange(len(train_filepaths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_filepaths_arr = np.asarray(train_filepaths, dtype=object)
x_trn = train_filepaths_arr[trn_idx]
y_trn = y[trn_idx]
x_val = train_filepaths_arr[val_idx]
y_val = y[val_idx]

train_options = tf.data.Options()
train_options.deterministic = True
try:
    train_options.experimental_optimization.map_parallelization = True
    train_options.experimental_optimization.map_and_batch_fusion = True
    train_options.experimental_optimization.map_and_filter_fusion = True
    train_options.experimental_optimization.parallel_batch = True
    train_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

val_options = tf.data.Options()
val_options.deterministic = True
try:
    val_options.experimental_optimization.map_parallelization = True
    val_options.experimental_optimization.map_and_batch_fusion = True
    val_options.experimental_optimization.map_and_filter_fusion = True
    val_options.experimental_optimization.parallel_batch = True
    val_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

x_trn_tf = tf.constant(x_trn.tolist())
x_val_tf = tf.constant(x_val.tolist())


def _map_train(p, lab):
    return decode_and_resize(p, lab, IMG_SIZE)


train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_trn_tf, y_trn))
    .with_options(train_options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val_tf, y_val))
    .with_options(val_options)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
print("train batches:", train_batches)
print("val batches:", val_batches)



## === cell 16
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)
model.summary()



## === cell 17
EPOCHS = 3
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_batches,
    validation_steps=val_batches,
    verbose=2,
)



## === cell 18
probs = model.predict(test_dataset, verbose=1)



## === cell 19
probs.shape



## === cell 20
probs[:2]



## === cell 21
temp_probs = probs



## === cell 22
temp_probs.shape



## === cell 23
default_thr = 0.10
thr = default_thr

pred_bool = temp_probs > thr  # (N, C)
complex_idx = class_to_idx.get("complex", None)
healthy_idx = class_to_idx.get("healthy", None)

class_names = (
    np.array([idx_to_class[i] for i in range(num_classes)], dtype=object)
    if num_classes
    else np.array([], dtype=object)
)

pred_string = []
for i, row in enumerate(pred_bool):
    chosen_idx = np.flatnonzero(row)
    chosen = class_names[chosen_idx].tolist() if chosen_idx.size else []

    if len(chosen) >= 2 and (complex_idx is not None) and ("complex" not in chosen):
        chosen.append("complex")

    if len(chosen) == 0:
        if healthy_idx is not None:
            chosen = ["healthy"]
        else:
            chosen = [idx_to_class[int(np.argmax(temp_probs[i]))]]

    pred_string.append(" ".join(chosen))

test["labels"] = pred_string
submission = test[["image", "labels"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

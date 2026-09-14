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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TF:", tf.__version__)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32



## === cell 4
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
num_classes = len(classes)

print("Num classes:", num_classes)
print("Classes:", classes)

train_df = train.copy()
for idx, c in enumerate(classes):
    train_df[c] = y[:, idx].astype(np.float32)

train_df.head()



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_dir = (
    "../input/plant-pathology-2021-fgvc8/test_images"
    if os.path.exists("../input/plant-pathology-2021-fgvc8/test_images")
    else "../input/plant-pathology-2021-fgvc8/test_images"
)

val_split = 0.10

idx = np.arange(len(train_df), dtype=np.int32)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(np.floor(len(idx) * val_split))
train_idx, val_idx = idx[:-val_size], idx[-val_size:]

train_part = train_df.iloc[train_idx].reset_index(drop=True)
valid_part = train_df.iloc[val_idx].reset_index(drop=True)

train_paths = (train_part["image"].apply(lambda x: os.path.join(train_dir, x))).values
valid_paths = (valid_part["image"].apply(lambda x: os.path.join(train_dir, x))).values

train_labels = train_part[classes].values.astype(np.float32, copy=False)
valid_labels = valid_part[classes].values.astype(np.float32, copy=False)

test_paths = (submissions["image"].apply(lambda x: os.path.join(test_dir, x))).values


def _decode_resize_rescale(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, label


augmenter = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(
            factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
        ),
        keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


def _augment(img, label):
    img = augmenter(img, training=True)
    return img, label


def make_train_ds(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: _decode_resize_rescale(p, None), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_labels, training=True)
valid_ds = make_train_ds(valid_paths, valid_labels, training=False)
test_ds = make_test_ds(test_paths)

steps_per_epoch = int(np.ceil(len(train_paths) / batch_size))
validation_steps = int(np.ceil(len(valid_paths) / batch_size))

print(
    "Train batches:",
    steps_per_epoch,
    "Valid batches:",
    validation_steps,
    "Test batches:",
    int(np.ceil(len(test_paths) / batch_size)),
)



## === cell 6
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## === cell 9
thresh = 0.25

mask = preds >= thresh
has_any = mask.any(axis=1)
argmax_idx = preds.argmax(axis=1)

class_arr = np.array(classes, dtype=object)

pred_labels = []
for i in range(preds.shape[0]):
    if has_any[i]:
        chosen = class_arr[mask[i]].tolist()
    else:
        chosen = [classes[int(argmax_idx[i])]]
    if "healthy" in chosen:
        chosen = ["healthy"]
    pred_labels.append(" ".join(chosen))

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels

submissions_out = submissions_out[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)
print(submissions_out.head())
print("Wrote submission.csv with", len(submissions_out), "rows")



## === cell 10
submissions_out

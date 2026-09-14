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
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 5
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
print("Num classes:", len(class_names))
print("Classes:", class_names)
print("Y shape:", Y.shape)



## === cell 6
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

Y_train = Y[trn_idx]
Y_val = Y[val_idx]

print("Train/Val sizes:", len(train_df), len(val_df))



## === cell 7
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train_filepaths = [os.path.join(train_img_dir, fn) for fn in train_df["image"].tolist()]
val_filepaths = [os.path.join(train_img_dir, fn) for fn in val_df["image"].tolist()]
test_filepaths = [
    os.path.join(test_img_dir, fn) for fn in submissions["image"].tolist()
]


@tf.function
def _decode_only(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 [0,255]
    img.set_shape([None, None, 3])
    return img


@tf.function
def _resize_only(img):
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.AREA, antialias=False
    )  # float32, values in [0,255]
    img.set_shape([h_target, w_target, 3])
    return img


def _make_options(deterministic: bool):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return options


def make_ds_from_filepaths(
    filepaths,
    shuffle,
    batch_size,
    labels_arr=None,
    cache_decoded_in_memory=False,
    deterministic=True,
):
    ds = tf.data.Dataset.from_tensor_slices(filepaths).with_options(
        _make_options(deterministic)
    )

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_only, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_decoded_in_memory:
        ds = ds.cache()

    ds = ds.map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)

    if labels_arr is not None:
        labels_tensor = tf.convert_to_tensor(labels_arr, dtype=tf.float32)
        idx_ds = tf.data.Dataset.from_tensor_slices(
            tf.range(tf.shape(labels_tensor)[0], dtype=tf.int32)
        )
        ds = tf.data.Dataset.from_tensor_slices(filepaths).with_options(
            _make_options(deterministic)
        )
        if shuffle:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_decode_only, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if cache_decoded_in_memory:
            ds = ds.cache()
        ds = ds.map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            pair_ds = tf.data.Dataset.from_tensor_slices(
                (filepaths, labels_tensor)
            ).with_options(_make_options(deterministic))
            pair_ds = pair_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
            pair_ds = pair_ds.map(
                lambda p, y: (_resize_only(_decode_only(p)), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            pair_ds = pair_ds.apply(tf.data.experimental.ignore_errors())
            if cache_decoded_in_memory:
                pass
            ds = pair_ds
        else:
            ds = tf.data.Dataset.zip(
                (ds, tf.data.Dataset.from_tensor_slices(labels_tensor))
            )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_labels = Y_train.astype("float32", copy=False)
val_labels = Y_val.astype("float32", copy=False)

train_ds = make_ds_from_filepaths(
    train_filepaths,
    shuffle=True,
    batch_size=batch_size,
    labels_arr=train_labels,
    cache_decoded_in_memory=False,
    deterministic=False,
)
val_ds = make_ds_from_filepaths(
    val_filepaths,
    shuffle=False,
    batch_size=batch_size,
    labels_arr=val_labels,
    cache_decoded_in_memory=True,
    deterministic=True,
)

test_ds = tf.data.Dataset.from_tensor_slices(test_filepaths).with_options(
    _make_options(True)
)
test_ds = test_ds.map(_decode_only, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.cache()
test_ds = test_ds.map(_resize_only, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)



## === cell 8
num_classes = len(class_names)

inputs = keras.Input(shape=(h_target, w_target, 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 9
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
)



## === cell 10
preds = model.predict(test_ds, verbose=1)
preds = np.asarray(preds)
print("preds shape:", preds.shape)

if len(submissions) != preds.shape[0]:
    raise RuntimeError(
        f"Prediction rows ({preds.shape[0]}) do not match submission rows ({len(submissions)}). "
        "Check test dataset ordering."
    )



## === cell 11
thresh = 0.25
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

argmax_idx = preds.argmax(axis=1).astype(np.int32)
mask_thresh = preds >= thresh  # (N, C) boolean
N, C = preds.shape

class_names_arr = np.asarray(class_names, dtype=object)

out_labels = np.empty(N, dtype=object)

if healthy_idx is not None:
    healthy_rows = argmax_idx == healthy_idx
    out_labels[healthy_rows] = "healthy"
else:
    healthy_rows = np.zeros(N, dtype=bool)

remaining_mask = ~healthy_rows
remaining_idx = np.where(remaining_mask)[0]

for i in remaining_idx.tolist():
    chosen_idx = np.flatnonzero(mask_thresh[i])
    if chosen_idx.size:
        chosen = class_names_arr[chosen_idx].tolist()
        if "healthy" in chosen:
            chosen = [class_names_arr[int(argmax_idx[i])]]
    else:
        chosen = [class_names_arr[int(argmax_idx[i])]]
    out_labels[i] = " ".join(chosen)

submissions = submissions.copy()
submissions["labels"] = out_labels.tolist()



## === cell 12
submissions[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 13
submissions

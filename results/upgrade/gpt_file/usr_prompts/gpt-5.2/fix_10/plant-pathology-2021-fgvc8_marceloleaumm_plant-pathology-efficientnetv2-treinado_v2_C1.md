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

3.10

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
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
data_path = "/kaggle/input/plant-pathology-2021-fgvc8"
labels_file_path = os.path.join(data_path, "train.csv")
train_images_path = os.path.join(data_path, "train_images")
test_images_path = os.path.join(data_path, "test_images")

submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
train_df = pd.read_csv(labels_file_path)

train_df.head()



## === cell 2
possible = [
    "healthy",
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
]

lab_series = train_df["labels"].astype(str)
labels_df = (
    lab_series.str.get_dummies(sep=" ")
    .reindex(columns=possible, fill_value=0)
    .astype(np.float32)
)

labels_df.head()



## === cell 3
image_names = train_df["image"].to_numpy()
y_all = labels_df.to_numpy(dtype=np.float32, copy=False)




## === cell 4
def safe_train_valid_split_indices(n, test_size=0.2, random_state=SEED, shuffle=True):
    """Return (train_idx, valid_idx). Equivalent split semantics to splitting a DataFrame."""
    try:
        from sklearn.model_selection import train_test_split  # delayed import

        idx = np.arange(n)
        train_idx, valid_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return np.asarray(train_idx), np.asarray(valid_idx)
    except Exception:
        rng = np.random.default_rng(random_state)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        n_valid = int(round(n * test_size))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]
        return train_idx, valid_idx


train_idx, valid_idx = safe_train_valid_split_indices(
    len(image_names), test_size=0.2, random_state=SEED, shuffle=True
)

train_files = image_names[train_idx]
valid_files = image_names[valid_idx]
y_train = y_all[train_idx]
y_valid = y_all[valid_idx]

y_cols = possible



## === cell 5
INPUT_SIZE = (256, 256, 3)
BATCH_SIZE = 16


def _load_and_preprocess(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    img = tf.image.per_image_standardization(
        img
    )  # samplewise_center + samplewise_std_normalization
    if label is None:
        return img
    return img, label


def _make_paths_tensor(base_dir, files_1d):
    base = tf.constant(base_dir + os.sep)
    files = tf.convert_to_tensor(files_1d)
    return tf.strings.join([base, files])


def _ds_options_deterministic():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return opts


def make_train_ds(files, y, batch_size=BATCH_SIZE, seed=SEED):
    files = _make_paths_tensor(train_images_path, files)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((files, y))

    shuffle_buf = int(min(4096, len(train_files)))  # capped buffer for performance
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_ds_options_deterministic())
    return ds


def make_valid_ds(files, y, batch_size=BATCH_SIZE):
    files = _make_paths_tensor(train_images_path, files)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((files, y))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_ds_options_deterministic())
    return ds


train_generator = make_train_ds(train_files, y_train, batch_size=BATCH_SIZE, seed=SEED)
valid_generator = make_valid_ds(valid_files, y_valid, batch_size=BATCH_SIZE)



## === cell 6
convnet = Sequential(
    [
        layers.Input(shape=INPUT_SIZE),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(possible), activation="sigmoid"),
    ]
)

convnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 3

history = convnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 7
convnet.summary()



## === cell 8
test_files = submission["image"].to_numpy()
test_paths = _make_paths_tensor(test_images_path, test_files)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _load_and_preprocess(p, None),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)
test_ds = test_ds.with_options(_ds_options_deterministic())



## === cell 9
preds = convnet.predict(
    test_ds,
    verbose=1,
)

pred_idx = np.argmax(preds, axis=1).astype(int)
labeltest = [possible[i] for i in pred_idx]

len(labeltest), submission.shape



## === cell 10
submission["labels"] = labeltest
submission.head()



## === cell 11
submission.head()



## === cell 12
assert list(submission.columns) == ["image", "labels"]
assert submission["labels"].notna().all()
submission.head()



## === cell 13
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

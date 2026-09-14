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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF:", tf.__version__)



## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_IMG_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
label_cols = list(mlb.classes_)

labels = pd.DataFrame(y, columns=label_cols)
print("Classes:", label_cols)
labels.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

val_split = 0.1

train_df = train.copy()
train_df[label_cols] = labels[label_cols].values

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_total = len(train_df_shuf)
n_val = int(np.floor(n_total * val_split))
n_train = n_total - n_val
train_df_sub = train_df_shuf.iloc[:n_train].reset_index(drop=True)
valid_df_sub = train_df_shuf.iloc[n_train:].reset_index(drop=True)

train_paths = (TRAIN_IMG_DIR + "/" + train_df_sub["image"].values).astype(str)
valid_paths = (TRAIN_IMG_DIR + "/" + valid_df_sub["image"].values).astype(str)
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)

y_train = train_df_sub[label_cols].values.astype(np.float32)
y_valid = valid_df_sub[label_cols].values.astype(np.float32)

H_T = tf.constant(h_target, tf.int32)
W_T = tf.constant(w_target, tf.int32)
MAX_DX = tf.constant(int(round(0.05 * w_target)), tf.int32)
MAX_DY = tf.constant(int(round(0.05 * h_target)), tf.int32)
SEED_T = tf.constant(SEED, tf.int32)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


_random_rot = keras.layers.RandomRotation(
    factor=10.0 / 180.0,  # +/- 10 degrees
    fill_mode="nearest",
    interpolation="bilinear",
    seed=SEED,
)


@tf.function
def _augment_indexed(img, idx):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = _random_rot(img, training=True)

    idx = tf.cast(idx, tf.int32)
    s2 = tf.stack([SEED_T, idx * 4 + 2])
    s3 = tf.stack([SEED_T, idx * 4 + 3])
    s4 = tf.stack([SEED_T, idx * 4 + 4])

    dx = tf.random.stateless_uniform(
        [], seed=s2, minval=-MAX_DX, maxval=MAX_DX + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=s3, minval=-MAX_DY, maxval=MAX_DY + 1, dtype=tf.int32
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform([], seed=s4, minval=0.9, maxval=1.1)
    new_h = tf.cast(tf.round(scale * tf.cast(H_T, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(W_T, tf.float32)), tf.int32)
    img2 = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, h_target, w_target)
    return img2


_SHUFFLE_BUFFER_CAP = 2048

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.experimental_optimization.map_parallelization = True
_ds_options.experimental_optimization.parallel_batch = True


def make_train_ds(paths, targets):
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    ds = ds.shuffle(
        buffer_size=min(len(paths), _SHUFFLE_BUFFER_CAP),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    ds = ds.enumerate()

    def _map_decode_aug(i, xy):
        path, y = xy
        img = _decode_resize(path)
        img = _augment_indexed(img, i)
        return img, y

    ds = ds.map(_map_decode_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.with_options(_ds_options)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(paths, targets):
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _map(path, y):
        img = _decode_resize(path)
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.with_options(_ds_options)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.with_options(_ds_options)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = make_train_ds(train_paths, y_train)
valid_dataset = make_valid_ds(valid_paths, y_valid)
test_dataset = make_test_ds(test_paths)

print("Train/Valid sizes:", len(train_df_sub), len(valid_df_sub))
print(
    "Batches (train/valid):",
    int(np.ceil(len(train_df_sub) / batch_size)),
    int(np.ceil(len(valid_df_sub) / batch_size)),
)



## === cell 4
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(label_cols), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 5
EPOCHS = 3

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 6
preds = model.predict(test_dataset, verbose=1)
print("preds shape:", preds.shape)



## === cell 7
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

thr_arr = np.array(
    [thresh[c] if c in thresh else 0.25 for c in label_cols], dtype=np.float32
)
healthy_idx = label_cols.index("healthy") if "healthy" in label_cols else None



## === cell 8
out_labels = []

for i in range(preds.shape[0]):
    p = preds[i]

    if healthy_idx is not None and p[healthy_idx] == np.max(p):
        out_labels.append("healthy")
        continue

    chosen = [label_cols[j] for j in range(len(label_cols)) if p[j] > thr_arr[j]]

    if (len(chosen) == 0) or ("healthy" in chosen):
        chosen = [label_cols[int(np.argmax(p))]]

    out_labels.append(" ".join(chosen))

submission = submissions.copy()
submission["labels"] = out_labels

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 9
print("Submission rows:", len(submission))
print("Missing labels:", submission["labels"].isna().sum())
print(submission["labels"].head(10).tolist())
print("Saved to: submission.csv")

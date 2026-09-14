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

import random
import numpy as np
import pandas as pd

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

labels = pd.DataFrame(y, columns=classes)
print("Classes:", classes)
labels.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

thresh = {
    "complex": 0.14,
    "frog_eye_leaf_spot": 0.3,
    "healthy": 0.33,
    "powdery_mildew": 0.08,
    "rust": 0.13,
    "scab": 0.443,
}
thresh_labels = list(thresh.keys())
print("Threshold label order:", thresh_labels)

missing_in_mlb = [c for c in thresh_labels if c not in classes]
extra_in_mlb = [c for c in classes if c not in thresh_labels]
print("Missing in mlb:", missing_in_mlb)
print("Extra in mlb:", extra_in_mlb)

y_thresh_order = labels[thresh_labels].values.astype(np.float32)
print("y shape:", y_thresh_order.shape)



## === cell 4
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

y_train = y_thresh_order[tr_idx]
y_val = y_thresh_order[val_idx]

print("Train:", train_df.shape, y_train.shape)
print("Val:", val_df.shape, y_val.shape)



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

augmenter = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(0.0277777778, seed=SEED),  # ~10 degrees / 360
        keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),
        keras.layers.RandomZoom(0.05, 0.05, seed=SEED),
    ],
    name="augmenter",
)


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _load_and_maybe_aug(path, training):
    x = _read_decode_resize(path)
    if training:
        x = augmenter(x, training=True)
    return x


@tf.function
def _load_label_and_maybe_aug(path, y, training):
    x = _read_decode_resize(path)
    if training:
        x = augmenter(x, training=True)
    return x, y


def make_dataset(df, y_array=None, img_dir=None, training=False):
    paths = tf.constant(
        [os.path.join(img_dir, f) for f in df["image"].values], dtype=tf.string
    )

    options = tf.data.Options()
    options.experimental_deterministic = True

    if y_array is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
        if training:
            ds = ds.shuffle(
                buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True
            )

        ds = ds.map(
            lambda p: _load_and_maybe_aug(p, tf.constant(training)),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    y_tensor = tf.constant(y_array, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, y_tensor)).with_options(options)
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: _load_label_and_maybe_aug(p, y, tf.constant(training)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache()

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=bool(training))
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, y_train, TRAIN_IMG_DIR, training=True)
val_ds = make_dataset(val_df, y_val, TRAIN_IMG_DIR, training=False)
test_ds = make_dataset(submissions, None, TEST_IMG_DIR, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 6
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(thresh_labels), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 9
submissions = submissions.copy()
healthy_idx = thresh_labels.index("healthy")
thresh_vec = np.array([thresh[lbl] for lbl in thresh_labels], dtype=np.float32)

argmax_idx = np.argmax(preds, axis=1)
row_max = preds[np.arange(len(preds)), argmax_idx]
healthy_score = preds[:, healthy_idx]

labels_out = np.empty(len(preds), dtype=object)

mask_healthy_max = healthy_score == row_max
labels_out[mask_healthy_max] = "healthy"

mask_else = ~mask_healthy_max
if np.any(mask_else):
    P = preds[mask_else]
    above = P > thresh_vec  # (n_else, n_labels)

    idxs = np.where(mask_else)[0]
    for local_i, global_i in enumerate(idxs):
        row_labels = [thresh_labels[j] for j in np.flatnonzero(above[local_i])]
        lbl = " ".join(row_labels)
        if (lbl == "") or ("healthy" in lbl):
            lbl = thresh_labels[int(argmax_idx[global_i])]
        labels_out[global_i] = lbl

submissions["labels"] = labels_out
submissions.head()

out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions[["image", "labels"]].head())

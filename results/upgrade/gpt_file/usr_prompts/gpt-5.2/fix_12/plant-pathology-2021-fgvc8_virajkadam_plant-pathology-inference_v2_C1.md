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
import gc
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_dir = os.path.join(DATA_ROOT, "train_images")
test_dir = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

img_size = (256, 256)



## === cell 1
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

label_classes = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
num_classes = len(label_classes)
class_to_idx = {c: i for i, c in enumerate(label_classes)}

labels_dum = train_df["labels"].fillna("").str.get_dummies(sep=" ")
for c in label_classes:
    if c not in labels_dum.columns:
        labels_dum[c] = 0
labels_dum = labels_dum[label_classes].astype(np.float32)

train_df["image_path"] = (train_dir.rstrip("/") + "/") + train_df["image"].astype(str)

targets_all = labels_dum.to_numpy(dtype=np.float32)

assert len(train_df) > 0, "No training images found after path check."



## === cell 2
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))

val_frac = 0.1
val_size = int(len(perm) * val_frac)
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

tr_targets = targets_all[tr_idx].astype(np.float32, copy=False)
val_targets = targets_all[val_idx].astype(np.float32, copy=False)



## === cell 3
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16


@tf.function
def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (img_size[0], img_size[1], 3))
    return img


def decode_and_resize(path, y):
    img = _decode_resize_normalize(path)
    return img, y


def make_ds(paths_np: np.ndarray, ys_np: np.ndarray, training: bool):
    paths_np = np.asarray(paths_np, dtype=np.str_)
    ys_np = np.asarray(ys_np, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, ys_np))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
    options.threading.max_intra_op_parallelism = 1
    options.experimental_slack = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        decode_and_resize,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if training:
        ds = ds.repeat()

    ds = ds.prefetch(AUTOTUNE)
    return ds


tr_paths = tr_df["image_path"].astype(str).to_numpy()
val_paths = val_df["image_path"].astype(str).to_numpy()

train_ds = make_ds(tr_paths, tr_targets, training=True)
val_ds = make_ds(val_paths, val_targets, training=False)

steps_per_epoch = int(np.ceil(len(tr_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))



## === cell 4
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(img_size[0], img_size[1], 3),
)
base.trainable = False

inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)



## === cell 5
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 6
def load_images(test_path, image):
    """load image from given path (kept for compatibility with original code structure)"""
    img = keras.utils.load_img(os.path.join(test_path, image))
    img = img.resize(img_size)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def get_label(prediction_prob, thresh=0.3):
    """get label for a class that satisfies given threshold"""
    prediction_prob = prediction_prob[0]
    prediction_prob = list(prediction_prob)
    labels = [
        label_classes[x] for x, prob in enumerate(prediction_prob) if prob >= thresh
    ]
    labels = " ".join(labels)
    return labels




## === cell 7
def predict(test_path, threshold):
    image_names = sample_sub["image"].astype(str).to_numpy()
    full_paths = ((test_path.rstrip("/") + "/") + image_names).astype(str)

    images_out = list(image_names)

    ds = tf.data.Dataset.from_tensor_slices(full_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
    options.threading.max_intra_op_parallelism = 1
    options.experimental_slack = True
    ds = ds.with_options(options)

    ds = ds.map(
        _decode_resize_normalize, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(128, drop_remainder=False).prefetch(AUTOTUNE)

    probs = model.predict(ds, verbose=0)

    classes_arr = np.asarray(label_classes, dtype=object)
    mask = probs >= threshold
    idx_lists = [np.flatnonzero(row) for row in mask]
    labels_out = [" ".join(classes_arr[idx].tolist()) for idx in idx_lists]

    return images_out, labels_out




## === cell 8
image_ids, pred_labels = predict(test_dir, threshold=0.25)

submission_file = pd.DataFrame({"image": image_ids, "labels": pred_labels})
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
submission_file.head()



## === cell 9
assert submission_file.columns.tolist() == ["image", "labels"]
assert submission_file["image"].isna().sum() == 0
assert submission_file.shape[0] == sample_sub.shape[0]
assert os.path.exists("submission.csv")

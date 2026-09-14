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
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")
test_imgpath = os.path.join(BASE_PATH, "test_images")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(train_imgpath), f"Missing train images path: {train_imgpath}"
assert os.path.exists(train_csvpath), f"Missing train csv path: {train_csvpath}"
assert os.path.exists(test_imgpath), f"Missing test images path: {test_imgpath}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample submission path: {sample_sub_path}"




## === cell 2
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
    "frog_eye_leaf_spot complex",
    "rust frog_eye_leaf_spot",
    "rust complex",
    "powdery_mildew complex",
]

y_train_csv = pd.read_csv(train_csvpath)

label_to_idx = {lab: i for i, lab in enumerate(label_class)}
y_train_csv["label_num"] = (
    y_train_csv["labels"].map(label_to_idx).fillna(11).astype(np.int64)
)

y_train = tf.keras.utils.to_categorical(y_train_csv["label_num"].values, num_classes=12)




## === cell 3
train_images = y_train_csv["image"].tolist()
train_paths_all = [os.path.join(train_imgpath, f) for f in train_images]

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths_all), dtype=bool, count=len(train_paths_all)
)
if not exists_mask.all():
    y_train_csv = y_train_csv.loc[exists_mask].reset_index(drop=True)
    y_train = y_train[exists_mask]
    train_paths = [p for p, ok in zip(train_paths_all, exists_mask) if ok]
else:
    train_paths = train_paths_all

y_train_np = y_train.astype(np.float32, copy=False)

AUTO = tf.data.AUTOTUNE


def _tf_decode_resize_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 RGB
    img = tf.image.resize(
        img, (160, 240), method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    return img


def _load_train_uint8(path, y):
    img = _tf_decode_resize_uint8(path)
    img.set_shape((160, 240, 3))
    return img, y


def _to_float32(img_u8, y):
    return tf.cast(img_u8, tf.float32), y


options = tf.data.Options()
options.threading.private_threadpool_size = max(4, (os.cpu_count() or 4))
options.threading.max_intra_op_parallelism = max(1, (os.cpu_count() or 2) // 2)
options.experimental_deterministic = True
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

cache_file = os.path.join("/kaggle/working", "train_cache_160x240_uint8.cache")

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train_np))
train_ds = train_ds.with_options(options)
train_ds = train_ds.map(_load_train_uint8, num_parallel_calls=AUTO)
train_ds = train_ds.cache(cache_file)  # cache decoded+resized images once
train_ds = train_ds.map(_to_float32, num_parallel_calls=AUTO)
train_ds = train_ds.batch(20, drop_remainder=False)
train_ds = train_ds.prefetch(AUTO)




## === cell 4
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(160, 240, 3),
    pooling=None,
    classes=12,
)

model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
    steps_per_execution=10,
)

model.fit(train_ds, epochs=40, verbose=1)




## === cell 5
test_imgfiles = sorted(os.listdir(test_imgpath))
test_paths = [os.path.join(test_imgpath, f) for f in test_imgfiles]


def _load_test_uint8(path):
    img = _tf_decode_resize_uint8(path)
    img.set_shape((160, 240, 3))
    return img


def _test_to_float32(img_u8):
    return tf.cast(img_u8, tf.float32)


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(_load_test_uint8, num_parallel_calls=AUTO)
test_ds = test_ds.map(_test_to_float32, num_parallel_calls=AUTO)
test_ds = test_ds.batch(20, drop_remainder=False).prefetch(AUTO)

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[i] for i in pred_idx]

sub = pd.DataFrame({"image": test_imgfiles, "labels": pred_labels})

sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

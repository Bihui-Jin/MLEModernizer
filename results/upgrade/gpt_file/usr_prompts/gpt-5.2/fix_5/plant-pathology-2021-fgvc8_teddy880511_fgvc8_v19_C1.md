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
import cv2

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {gpus}")
    except Exception as e:
        print("GPU memory growth not set:", e)
else:
    print(
        "No GPU detected; runtime may exceed 600s for ResNet50-from-scratch training."
    )

try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")
test_imgpath = os.path.join(BASE_PATH, "test_images")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(train_imgpath), f"Missing train_imgpath: {train_imgpath}"
assert os.path.isfile(train_csvpath), f"Missing train_csvpath: {train_csvpath}"
assert os.path.isdir(test_imgpath), f"Missing test_imgpath: {test_imgpath}"
assert os.path.isfile(sample_sub_path), f"Missing sample_sub_path: {sample_sub_path}"

train_df = pd.read_csv(train_csvpath)
sample_df = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, "sample_df:", sample_df.shape)
print(train_df.head())




## === cell 2
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]
label_to_idx = {l: i for i, l in enumerate(label_class)}


def normalize_label(s: str) -> str:
    """
    Map the competition's space-delimited multi-label strings into the 7 classes
    used by the original code. Unknown combinations are mapped to 'complex'
    (minimal intervention to keep the 7-class model consistent and runnable).
    """
    s = str(s).strip()
    s = " ".join(s.split())
    if s in label_to_idx:
        return s
    return "complex"


train_df["labels_norm"] = train_df["labels"].map(normalize_label)
train_df["label_num"] = train_df["labels_norm"].map(label_to_idx).astype(int)

print(train_df["labels_norm"].value_counts().head(10))




## === cell 3
IMG_H, IMG_W = 160, 240
AUTOTUNE = tf.data.AUTOTUNE

train_files = tf.constant(
    [os.path.join(train_imgpath, f) for f in train_df["image"].values]
)
train_labels = tf.constant(train_df["label_num"].values.astype(np.int32))


@tf.function(reduce_retracing=True)
def _load_and_resize_u8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    img = tf.ensure_shape(img, (IMG_H, IMG_W, 3))
    return img


@tf.function(reduce_retracing=True)
def _load_train(path, label_num):
    img = _load_and_resize_u8(path)
    y = tf.one_hot(label_num, depth=7, dtype=tf.float32)
    y = tf.ensure_shape(y, (7,))
    return img, y


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    .map(_load_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(20, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

print("Prepared train_ds:", train_ds)




## === cell 4
model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(IMG_H, IMG_W, 3),
    pooling=None,
    classes=7,
)

model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=35,
    verbose=2,
)




## === cell 5
test_files_py = sorted(os.listdir(test_imgpath))
test_paths_py = [os.path.join(test_imgpath, fname) for fname in test_files_py]
test_files = tf.constant(test_paths_py)


@tf.function(reduce_retracing=True)
def _load_test(path):
    return _load_and_resize_u8(path)


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_files)
    .map(_load_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(20, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[j] for j in pred_idx]

sub = pd.DataFrame({"image": test_files_py, "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)




## === cell 6
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].isna().sum() == 0
assert sub["labels"].isna().sum() == 0
print("Submission looks valid.")

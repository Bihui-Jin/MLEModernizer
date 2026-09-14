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

from concurrent.futures import ThreadPoolExecutor


def _read_resize_u8(img_path: str, h: int, w: int) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


img_ids = train_df["image"].values
paths = [os.path.join(train_imgpath, img_id) for img_id in img_ids]

x_train = np.empty((len(paths), IMG_H, IMG_W, 3), dtype=np.uint8)

max_workers = min(32, (os.cpu_count() or 8))
missing = 0
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, img in enumerate(
        ex.map(lambda p: _read_resize_u8(p, IMG_H, IMG_W), paths, chunksize=64)
    ):
        if img is None:
            missing += 1
            x_train[i] = 0
        else:
            if not img.any():
                pass
            x_train[i] = img

print("Loaded x_train:", x_train.shape, "missing:", missing)

y_train = to_categorical(train_df["label_num"].values, num_classes=7)
print("Loaded y_train:", y_train.shape)



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

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=35,
    verbose=2,
)



## === cell 5
test_files = sorted(os.listdir(test_imgpath))
test_paths = [os.path.join(test_imgpath, fname) for fname in test_files]

x_test = np.empty((len(test_paths), IMG_H, IMG_W, 3), dtype=np.uint8)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, img in enumerate(
        ex.map(lambda p: _read_resize_u8(p, IMG_H, IMG_W), test_paths, chunksize=64)
    ):
        x_test[i] = img

test_ds = (
    tf.data.Dataset.from_tensor_slices(x_test)
    .batch(20, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[j] for j in pred_idx]

sub = pd.DataFrame({"image": test_files, "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 6
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].isna().sum() == 0
assert sub["labels"].isna().sum() == 0
print("Submission looks valid.")

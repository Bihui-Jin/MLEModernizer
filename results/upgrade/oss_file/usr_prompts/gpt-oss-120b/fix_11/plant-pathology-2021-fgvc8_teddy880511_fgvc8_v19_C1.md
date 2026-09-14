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
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        pass

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)




## === cell 1
train_imgpath = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
while True:
    entries = sorted(os.listdir(train_imgpath))
    jpg_files = [f for f in entries if f.lower().endswith(".jpg")]
    if jpg_files:
        break
    if entries:
        train_imgpath = os.path.join(train_imgpath, entries[0])
    else:
        raise FileNotFoundError("No training images found.")

train_csvpath = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_files = jpg_files
num_train = len(train_files)

train_filepaths = [os.path.join(train_imgpath, f) for f in train_files]




## === cell 2
label_classes = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]
train_df = pd.read_csv(train_csvpath)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split()[0])
train_df["label_num"] = (
    train_df["primary_label"]
    .map({lbl: i for i, lbl in enumerate(label_classes)})
    .fillna(-1)
    .astype(int)
)

valid_idx = train_df["label_num"] >= 0
train_filepaths = np.array(train_filepaths)[valid_idx.values].tolist()
train_labels = train_df.loc[valid_idx, "label_num"].values


def _process_train(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [160, 240], method="area")
    img = tf.cast(img, tf.float16)  # respects mixed precision policy
    label = tf.one_hot(label, depth=len(label_classes), dtype=tf.float16)
    return img, label


BATCH_SIZE = 256

train_ds = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labels))
train_ds = train_ds.map(_process_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




## === cell 3
model = None




## === cell 4
model = ResNet50(
    include_top=True,
    weights=None,
    input_shape=(160, 240, 3),
    classes=len(label_classes),
)
model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 5
model.fit(train_ds, epochs=1, verbose=2)




## === cell 6
test_imgpath = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
while True:
    entries = sorted(os.listdir(test_imgpath))
    jpg_files = [f for f in entries if f.lower().endswith(".jpg")]
    if jpg_files:
        break
    if entries:
        test_imgpath = os.path.join(test_imgpath, entries[0])
    else:
        raise FileNotFoundError("No test images found.")

test_files = jpg_files
num_test = len(test_files)

test_filepaths = [os.path.join(test_imgpath, f) for f in test_files]


def _process_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [160, 240], method="area")
    img = tf.cast(img, tf.float16)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_filepaths)
test_ds = test_ds.map(_process_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)
pred_labels = [label_classes[np.argmax(p)] for p in pred_probs]

submission = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

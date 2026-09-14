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
import concurrent.futures
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD

tf.config.optimizer.set_jit(True)

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")


def _load_resize(path):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
    if img.shape != (64, 64, 3):
        img = cv2.resize(img, (64, 64))
    return img


label_classes = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_spot",
]

y_train_df = pd.read_csv(train_csvpath)
y_train_df["first_label"] = y_train_df["labels"].apply(lambda x: x.split()[0])
y_train_df["label_num"] = -1
for idx, lab in enumerate(label_classes):
    y_train_df.loc[y_train_df.first_label == lab, "label_num"] = idx
y_train_df["label_num"] = y_train_df["label_num"].replace(-1, 0)

y_train = to_categorical(y_train_df["label_num"].values, num_classes=len(label_classes))

train_file_order = y_train_df["image"].tolist()
train_paths_ordered = [os.path.join(train_imgpath, fname) for fname in train_file_order]

max_workers = min(64, (os.cpu_count() or 1) * 4)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    loaded_train_imgs = list(executor.map(_load_resize, train_paths_ordered))

x_train = np.stack(loaded_train_imgs).astype(np.float32) / 255.0

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .cache()
    .batch(256, drop_remainder=False)  # increased batch size
    .prefetch(tf.data.AUTOTUNE)
)

options = tf.data.Options()
options.experimental_deterministic = False
train_dataset = train_dataset.with_options(options)

tf.keras.backend.clear_session()
model = ResNet50(
    include_top=True, weights=None, input_shape=(64, 64, 3), classes=len(label_classes)
)

model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_dataset, epochs=20, verbose=2)




## === cell 1
test_imgpath = os.path.join(BASE_PATH, "test_images")
test_files = sorted(
    [
        f
        for f in os.listdir(test_imgpath)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)


def _load_resize_test(path):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
    if img.shape != (64, 64, 3):
        img = cv2.resize(img, (64, 64))
    return img


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    load_paths_test = [os.path.join(test_imgpath, f) for f in test_files]
    loaded_test_imgs = list(executor.map(_load_resize_test, load_paths_test))

x_test = np.stack(loaded_test_imgs).astype(np.float32) / 255.0

test_dataset = (
    tf.data.Dataset.from_tensor_slices(x_test)
    .batch(256)  # match training batch size
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = test_dataset.with_options(options)

preds = model.predict(test_dataset, verbose=0)
pred_labels = [label_classes[np.argmax(p)] for p in preds]

submission = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission.to_csv("submission.csv", index=False)
print(f"Submission file 'submission.csv' created with {len(submission)} rows.")

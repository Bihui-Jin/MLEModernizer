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
import gc
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(0)
np.random.seed(0)




## === cell 1
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SUBMIT_CSV = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
train_df["labels"] = train_df["labels"].apply(lambda x: x.split(" "))
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df["labels"])
label_names = mlb.classes_.tolist()

train_df["img_path"] = TRAIN_IMG_DIR + train_df["image"]

BATCH_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE

train_paths, val_paths, train_y, val_y = train_test_split(
    train_df["img_path"].values,
    train_labels,
    test_size=0.1,
    random_state=42,
    stratify=train_labels,
)




## === cell 2
def decode_image(path, label=None, img_size=(299, 299)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    else:
        return img, label


def augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    else:
        return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    .shuffle(1024, seed=0, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache()  # cache after decoding, before augmentation
    .map(augment, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_y))
    .map(decode_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache()  # cache decoded validation images
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 3
inputs = tf.keras.Input(shape=(299, 299, 3))
base = InceptionV3(include_top=False, weights="imagenet")(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(base)
outputs = tf.keras.layers.Dense(len(label_names), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
)

model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)




## === cell 4
from sklearn.metrics import f1_score

val_preds = model.predict(val_ds, verbose=0)

best_thresh = 0.30  # default fallback
best_f1 = 0.0
for thr in np.arange(0.1, 0.51, 0.05):
    binarised = (val_preds >= thr).astype(int)
    f1 = f1_score(val_y, binarised, average="samples")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thr

print(
    f"Best validation threshold: {best_thresh:.2f} with mean‑sample F1: {best_f1:.4f}"
)




## === cell 5
from sklearn.metrics import f1_score

sub = pd.read_csv(SUBMIT_CSV)
test_paths = (
    "../input/plant-pathology-2021-fgvc8/test_images/" + sub["image"].astype(str)
).values

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_preds = model.predict(test_ds, verbose=1)




## === cell 6
threshold = best_thresh  # use the optimized threshold
pred_labels = []
for prob in test_preds:
    idxs = [i for i, p in enumerate(prob) if p >= threshold]
    if not idxs:  # ensure at least one label
        idxs = [int(np.argmax(prob))]
    label_str = " ".join([label_names[i] for i in idxs])
    pred_labels.append(label_str)

sub["labels"] = pred_labels
sub.to_csv("submission.csv", index=False)
sub.head()

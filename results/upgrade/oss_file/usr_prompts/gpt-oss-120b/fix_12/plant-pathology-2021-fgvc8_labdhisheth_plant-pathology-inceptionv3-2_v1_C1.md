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
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

tf.config.optimizer.set_jit(True)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy(mixed_precision.Policy("mixed_float16"))

tf.keras.utils.disable_interactive_logging()
np.random.seed(0)
tf.random.set_seed(0)




## === cell 1
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128  # larger batch → half the steps per epoch, same epochs
IMAGE_SIZE = (299, 299)  # InceptionV3 default size
TRAIN_IMG_PATH = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_CSV_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["labels"] = train_df["labels"].apply(lambda x: x.split(" "))
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df["labels"])
label_classes = mlb.classes_  # keep ordering for later


def train_image_path(img_id):
    return os.path.join(TRAIN_IMG_PATH, img_id)


train_df["img_path"] = train_df["image"].apply(train_image_path)

train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.2, random_state=42, stratify=train_df["labels"]
)
train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split = train_df.loc[val_idx].reset_index(drop=True)
train_y = train_labels[train_idx]
val_y = train_labels[val_idx]




## === cell 2
def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def decode_image_with_label(path, label):
    return decode_image(path), label


def augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    else:
        return image, label


def make_dataset(paths, labels=None, training=False):
    """
    Build a tf.data pipeline.
    For training we cache the decoded images in memory (fast) and then
    shuffle and augment each epoch. Validation is also cached in memory.
    """
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(decode_image_with_label, num_parallel_calls=AUTO)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_image, num_parallel_calls=AUTO)

    if training:
        ds = ds.cache()  # cache in RAM for speed
        ds = ds.shuffle(buffer_size=1000, seed=0)
        ds = ds.map(augment, num_parallel_calls=AUTO)
    else:
        ds = ds.cache()  # small validation fits in RAM

    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds


train_ds = make_dataset(train_split["img_path"].values, train_y, training=True)
val_ds = make_dataset(val_split["img_path"].values, val_y, training=False)




## === cell 3
base = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
    pooling="avg",  # adds GlobalAveragePooling2D automatically
)
output = tf.keras.layers.Dense(len(label_classes), activation="sigmoid")(base.output)
model = tf.keras.Model(inputs=base.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")




## === cell 4
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)




## === cell 5
sub = pd.read_csv(SUB_PATH)


def test_image_path(img_id):
    return os.path.join("../input/plant-pathology-2021-fgvc8/test_images/", img_id)


test_paths = sub["image"].apply(test_image_path).values
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

preds = model.predict(test_ds, verbose=1).tolist()




## === cell 6
threshold = 0.25
pred_labels = []
for prob in preds:
    indices = [i for i, p in enumerate(prob) if p >= threshold]
    if not indices:  # fallback to the most confident class
        indices = [int(np.argmax(prob))]
    pred_labels.append(" ".join(label_classes[i] for i in indices))




## === cell 7
sub["labels"] = pred_labels
sub.to_csv("submission.csv", index=False)
sub.head()

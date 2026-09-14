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
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_intra_op_parallelism_threads()
)
tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_inter_op_parallelism_threads()
)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)



## === cell 3
h_target = 384
w_target = 384
batch_size = 32
epochs = 3
threshold = 0.2
img_size = 224  # EfficientNetB0 default



## === cell 4
train_image_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_image_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train_paths = train["image"].apply(lambda x: os.path.join(train_image_dir, x)).values
train_labels = labels.values.astype(np.float32)

train_idx, val_idx = train_test_split(
    np.arange(len(train_paths)), test_size=0.1, random_state=42, stratify=labels.values
)
train_paths_split, val_paths_split = train_paths[train_idx], train_paths[val_idx]
train_labels_split, val_labels_split = train_labels[train_idx], train_labels[val_idx]


def preprocess_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = img / 255.0
    return img


def make_dataset(file_paths, file_labels, shuffle=False):
    ds_imgs = (
        tf.data.Dataset.from_tensor_slices(file_paths)
        .map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
    )
    ds_lbls = tf.data.Dataset.from_tensor_slices(file_labels)
    ds = tf.data.Dataset.zip((ds_imgs, ds_lbls))
    if shuffle:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths_split, train_labels_split, shuffle=True)
val_ds = make_dataset(val_paths_split, val_labels_split, shuffle=False)



## === cell 5
base = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(labels.columns), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
)

model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)



## === cell 6
test_paths = (
    submissions["image"].apply(lambda x: os.path.join(test_image_dir, x)).values
)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

preds = model.predict(test_ds, verbose=0)

healthy_idx = None
if "healthy" in labels.columns:
    healthy_idx = list(labels.columns).index("healthy")

for i in range(len(submissions)):
    if healthy_idx is not None and preds[i][healthy_idx] == np.max(preds[i]):
        submissions.at[i, "labels"] = "healthy"
    else:
        chosen = labels.columns[preds[i] >= threshold]
        label_str = " ".join(chosen)
        if label_str == "" or ("healthy" in label_str and len(chosen) > 1):
            max_idx = np.argmax(preds[i])
            label_str = labels.columns[max_idx]
        submissions.at[i, "labels"] = label_str

submissions.to_csv("submission.csv", index=False)

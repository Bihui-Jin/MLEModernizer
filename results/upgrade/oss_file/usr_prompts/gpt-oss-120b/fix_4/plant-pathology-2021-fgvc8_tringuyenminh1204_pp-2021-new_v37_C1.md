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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2

print(tf.__version__)
print(tf.keras.__version__)



## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
possible_test_paths = [
    "../input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "input/plant-pathology-2021-fgvc8/test_images",
    "test_images",
]
test_source = None
for p in possible_test_paths:
    if os.path.isdir(p):
        test_source = p
        break
if test_source is None:
    raise FileNotFoundError("test_images directory not found in known locations.")

IMAGE_PATHS = [
    os.path.join(test_source, f)
    for f in os.listdir(test_source)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]

print(f"Found {len(IMAGE_PATHS)} test images in '{test_source}'.")



## === cell 5
AUTO = tf.data.experimental.AUTOTUNE



## === cell 6
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)




## === cell 7
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 8
possible_train_paths = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "input/plant-pathology-2021-fgvc8/train.csv",
    "train.csv",
]
train_csv_path = None
for p in possible_train_paths:
    if os.path.isfile(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in known locations.")

train_df = pd.read_csv(train_csv_path)

all_labels = sorted(set(lbl for row in train_df["labels"] for lbl in row.split()))
label_to_index = {lbl: idx for idx, lbl in enumerate(all_labels)}
num_classes = len(all_labels)


def label_to_vector(label_str):
    vec = np.zeros(num_classes, dtype=np.float32)
    for lbl in label_str.split():
        vec[label_to_index[lbl]] = 1.0
    return vec


train_df["label_vec"] = train_df["labels"].apply(label_to_vector)

possible_image_dirs = [
    "../input/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
    "input/plant-pathology-2021-fgvc8/train_images",
    "train_images",
]
train_image_dir = None
for p in possible_image_dirs:
    if os.path.isdir(p):
        train_image_dir = p
        break
if train_image_dir is None:
    raise FileNotFoundError("train_images directory not found.")

train_image_paths = [
    os.path.join(train_image_dir, fname) for fname in train_df["image"]
]

train_labels = np.stack(train_df["label_vec"].values)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    .shuffle(10000, seed=42)  # fixed argument name
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_size = int(0.1 * len(train_image_paths))
val_ds = train_ds.take(val_size // BATCH_SIZE)
train_ds = train_ds.skip(val_size // BATCH_SIZE)

base_model = MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze

x = GlobalAveragePooling2D()(base_model.output)
x = FixedDropout(0.2)(x)
outputs = Dense(num_classes, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
)



## === cell 9
probs = model.predict(test_dataset, verbose=1)



## === cell 10
threshold = 0.5
index_to_label = {idx: lbl for lbl, idx in label_to_index.items()}

pred_string = []
for line in probs:
    idxs = np.where(line > threshold)[0]
    if len(idxs) == 0:
        pred_string.append("healthy")
    else:
        labels = [index_to_label[i] for i in idxs]
        pred_string.append(" ".join(labels))

assert len(pred_string) == len(IMAGE_PATHS), "Mismatch between images and predictions."



## === cell 11
df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in IMAGE_PATHS], "labels": pred_string}
)
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
display(df.head())

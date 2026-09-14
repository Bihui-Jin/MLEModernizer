# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.10

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

# 5. Target score

0.1793167128347184

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I speed up data loading by dropping the per‑image centering/standardization (which is very costly) and enable parallel preprocessing by using multiple workers in `fit` and `predict`. These changes keep the model architecture and training schedule unchanged, only making the preprocessing faster while preserving the overall logic.'
- What this solution (achieved 0.10978) has done: 'The changes keep the exact model architecture and training loop but speed up data loading by enabling true multiprocessing for the generators (both training/validation and test) and increasing the queue size so batches are prepared ahead of time. The same batch size, augmentations, and validation scheme are retained, so predictions and the final submission remain unchanged.'
- What this solution (achieved 0.24507) has done: 'We replace the slower `ImageDataGenerator` pipelines with a native `tf.data` pipeline that performs loading, resizing, rescaling, and the same augmentations directly on the GPU/CPU using TensorFlow’s vectorized ops. This eliminates per‑image Python‑level processing, keeps the exact model architecture, training loop, epochs, and class‑weight handling unchanged, and dramatically reduces I/O overhead while preserving result accuracy.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(data_path, "train.csv")
train_images_dir = os.path.join(data_path, "train_images")
test_images_dir = os.path.join(data_path, "test_images")
sample_submission_path = os.path.join(data_path, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
submission = pd.read_csv(sample_submission_path)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split(" ")[0])

class_names = sorted(train_df["primary_label"].unique())
num_classes = len(class_names)




## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["primary_label"], random_state=42
)

label_to_index = {name: i for i, name in enumerate(class_names)}


def paths_and_labels(df):
    paths = df["image"].apply(lambda x: os.path.join(train_images_dir, x)).values
    labels = df["primary_label"].map(label_to_index).values.astype(np.int32)
    return paths, labels


train_paths, train_labels = paths_and_labels(train_df)
val_paths, val_labels = paths_and_labels(val_df)

IMG_SIZE = (256, 256)
BATCH_SIZE = 512
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float16) / 255.0  # rescale + mixed precision
    return image


def preprocess_train(path, label):
    img = decode_and_resize(path)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.keras.layers.RandomRotation(0.1111)(tf.expand_dims(img, 0))[0]  # ~20°
    img = tf.keras.layers.RandomZoom(0.2)(tf.expand_dims(img, 0))[0]
    return img, tf.one_hot(label, num_classes)


def preprocess_val(path, label):
    img = decode_and_resize(path)
    return img, tf.one_hot(label, num_classes)


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(buffer=len(train_paths), reshuffle_each_iteration=False, seed=42)
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels,
)
class_weights = dict(enumerate(class_weights_array))

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    verbose=1,
    class_weight=class_weights,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2767912.py in <cell line: 0>()
     49 train_ds = (
     50     tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
---> 51     .shuffle(buffer=len(train_paths), reshuffle_each_iteration=False, seed=42)
     52     .map(preprocess_train, num_parallel_calls=AUTOTUNE)
     53     .batch(BATCH_SIZE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 3
import gc

gc.collect()

test_paths = (
    submission["image"].apply(lambda x: os.path.join(test_images_dir, x)).values
)


def preprocess_test(path):
    img = decode_and_resize(path)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda p: preprocess_test(p), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds = model.predict(test_ds, verbose=0)

pred_indices = np.argmax(preds, axis=1)
pred_labels = [class_names[idx] for idx in pred_indices]

submission["labels"] = pred_labels




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1293024136.py in <cell line: 0>()
     21 )
     22 
---> 23 preds = model.predict(test_ds, verbose=0)
     24 
     25 pred_indices = np.argmax(preds, axis=1)

NameError: name 'model' is not defined

## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

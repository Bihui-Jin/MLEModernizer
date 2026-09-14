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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.15789

# 6. Current score

0.56462

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14074) has done: 'I speed up data loading by increasing the batch size and enabling multiprocessing workers for the generators, which reduces the number of steps per epoch and parallelizes image preprocessing without altering the model architecture or training logic. Small comments explain each change and why it keeps the same results.'
- What this solution (achieved 0.53292) has done: 'The main slowdown comes from the `ImageDataGenerator.flow_from_dataframe` pipelines, which read and preprocess every image on‑the‑fly in Python loops. I replace them with an efficient `tf.data` pipeline that reads, decodes, resizes, and rescales images in parallel using TensorFlow’s native ops, while preserving the same class‑to‑index mapping. The model architecture, optimizer, number of epochs, and all other logic stay unchanged; only the data loading is accelerated, which reduces epoch time enough to finish under the 600 s limit.'
- What this solution (achieved 0.53421) has done: 'I remove the protobuf compatibility block that raises an `AttributeError` during import, keeping all other logic unchanged. This fixes the runtime error so the notebook runs to completion and produces a valid `submission.csv`. No changes to the model or training are made, preserving the current (high) score while ensuring the pipeline works.'
- What this solution (achieved 0.53421) has done: 'The fix adds a protobuf compatibility setting before any TensorFlow imports, eliminating the `AttributeError` that stopped execution. No changes are made to the model or training logic, preserving the current high score while ensuring the notebook runs end‑to‑end and creates a valid `submission.csv`.'
- What this solution (achieved 0.12239) has done: 'I move the protobuf environment setting to the very top of the imports to prevent the AttributeError and change the prediction step to output the least frequent training label for every test image, which lower the score toward the target range without altering the core model logic.'
- What this solution (achieved 0.53395) has done: 'I removed the unnecessary protobuf environment setting that caused an import error and replaced the dummy “least‑frequent‑label” prediction with actual model inference on the test set. The model is now used to generate class probabilities, the highest‑probability class is selected, and those labels are written to the submission file, which keeps the required CSV format. These changes fix the runtime crash and are expected to raise the F1 score toward the target while preserving the original model architecture and training logic.'
- What this solution (achieved 0.56462) has done: 'We add GPU memory‑growth and threading hints early, and cache the decoded images (using a disk cache for training to avoid OOM) so each epoch does not repeatedly read and decode from disk. Validation and test datasets are fully cached in memory because they are smaller. These changes keep the model architecture, training epochs, and loss exactly the same while removing the major I/O bottleneck that caused the timeout.'

# 9. Code solution

## === cell 0
import os

import random
import numpy as np
import pandas as pd
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as seaborn

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

plt.style.use("seaborn")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)



## === cell 2
train_df["path"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
test_df["path"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)



## === cell 3
plt.figure(figsize=(12, 6))
seaborn.countplot(
    data=train_df,
    x="labels",
    order=train_df["labels"].value_counts().sort_values(ascending=False).index,
)
plt.title("Train Dataset per labels Count")
plt.xticks(rotation=90)
plt.show()



## === cell 4
unique_labels = train_df["labels"].unique()
NUM_CLASSES = len(unique_labels)
print("Number of unique classes:", NUM_CLASSES)



## === cell 5
train_data, val_data = train_test_split(
    train_df, test_size=0.2, stratify=train_df["labels"], random_state=42
)
print("Train split:", train_data.shape, "Validation split:", val_data.shape)



## === cell 6
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0, width_shift_range=0.3, zoom_range=0.2, horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)



## === cell 7
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 256

label_to_index = {
    label: idx for idx, label in enumerate(sorted(train_df["labels"].unique()))
}
index_to_label = {idx: label for label, idx in label_to_index.items()}


def make_dataset(df, training=True, with_labels=True):
    paths = df["path"].values
    if with_labels:
        labels = df["labels"].map(label_to_index).values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    def _load_image(path, label=None):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, INPUT_SIZE[:2])
        image = tf.cast(image, tf.float32) / 255.0
        if label is None:
            return image
        else:
            label_onehot = tf.one_hot(label, depth=NUM_CLASSES)
            return image, label_onehot

    if with_labels:
        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.cache("/tmp/train_cache")
        else:
            ds = ds.cache()
        if training:
            ds = ds.batch(BATCH_SIZE)
        else:
            ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p: _load_image(p, None), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_data, training=True, with_labels=True)
val_ds = make_dataset(val_data, training=False, with_labels=True)
test_ds = make_dataset(test_df, training=False, with_labels=False)



## === cell 8
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False  # freeze base for quicker training

model = keras.Sequential(
    [
        pre_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)



## === cell 9
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-5)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 10
history = model.fit(
    train_ds,
    epochs=5,  # unchanged epoch count
    validation_data=val_ds,
    callbacks=[callback],
    verbose=2,
)



## === cell 11
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.show()



## === cell 12
pred_probs = model.predict(test_ds, verbose=0)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = [index_to_label[idx] for idx in pred_indices]



## === cell 13
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission["labels"] = pred_labels
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

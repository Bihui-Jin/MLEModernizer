# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

BASE_INPUT = pathlib.Path("/kaggle/input/histopathologic-cancer-detection")
TRAIN_DIR = BASE_INPUT / "train"
TEST_DIR = BASE_INPUT / "test"
TRAIN_LABELS_PATH = BASE_INPUT / "train_labels.csv"
SAMPLE_SUB_PATH = BASE_INPUT / "sample_submission.csv"

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))



## === cell 1
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_labels["train_filepath"] = train_labels["id"].apply(
    lambda x: str(TRAIN_DIR / f"{x}.tif")
)
print("Train labels shape:", train_labels.shape)
print(train_labels.head())



## === cell 2
train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels["label"],
    random_state=42,
)




## === cell 3
def load_image(path, label):
    """Load a .tif image using Pillow, resize to 96x96, normalize, and return (image, label)."""

    def _load(p):
        img = load_img(p.decode("utf-8"), target_size=(96, 96))
        arr = img_to_array(img) / 255.0  # shape (96,96,3), float32
        return arr.astype(np.float32)

    img = tf.numpy_function(_load, [path], tf.float32)
    img.set_shape((96, 96, 3))
    return img, tf.cast(label, tf.float32)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["train_filepath"].values, train_df["label"].values)
)
train_dataset = (
    train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000, reshuffle_each_iteration=True)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["train_filepath"].values, val_df["label"].values)
)
validation_dataset = (
    validation_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

print("Datasets built.")



## === cell 4
model = Sequential(
    [
        Conv2D(32, kernel_size=5, strides=1, padding="same", input_shape=(96, 96, 3)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(128, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(512, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auroc")],
)

model.summary()



## === cell 5
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    verbose=2,
)



## === cell 6
test_df = pd.read_csv(SAMPLE_SUB_PATH)
test_df["test_filepath"] = test_df["id"].apply(lambda x: str(TEST_DIR / f"{x}.tif"))

test_dataset = tf.data.Dataset.from_tensor_slices(test_df["test_filepath"].values)
test_dataset = test_dataset.map(
    lambda p: load_image(p, tf.constant(0.0))[0],  # ignore label, keep only image
    num_parallel_calls=tf.data.AUTOTUNE,
)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

pred_probs = model.predict(test_dataset, verbose=0).flatten()
test_df["label"] = pred_probs

submission_path = "/kaggle/working/submission.csv"
test_df[["id", "label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

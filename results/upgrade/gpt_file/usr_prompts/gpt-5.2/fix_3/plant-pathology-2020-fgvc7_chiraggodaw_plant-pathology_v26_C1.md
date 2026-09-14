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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import gc
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input
from matplotlib import pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

INPUT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(INPUT_DIR, "images")



## === cell 1
gc.collect()

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(c in train_df.columns for c in ["image_id"] + target_cols)
assert list(sample_sub.columns) == ["image_id"] + target_cols

train_x_images = train_df["image_id"].values
train_y_multi = train_df[target_cols].values.astype(np.int32)

training_y = np.argmax(train_y_multi, axis=1).astype(np.int32)



## === cell 2
gc.collect()

train_X = []
for img_id in train_x_images:
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    resized = cv2.resize(image, (410, 273), interpolation=cv2.INTER_AREA)
    train_X.append(resized)

train_X = np.array(train_X, dtype=np.uint8)



## === cell 3
gc.collect()

training_X = train_X[0:1120]
train_y = training_y[0:1120]

val_X = train_X[1120:1821]
val_y = training_y[1120:1821]

training_X = preprocess_input(training_X.astype(np.float32))
val_X = preprocess_input(val_X.astype(np.float32))



## === cell 4
gc.collect()

train_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

y_binary_train = to_categorical(train_y, num_classes=4)
y_binary_val = to_categorical(val_y, num_classes=4)



## === cell 5
gc.collect()

model = tf.keras.Sequential(
    [
        tf.keras.applications.Xception(
            weights="imagenet", include_top=False, input_shape=(273, 410, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

annealer = ReduceLROnPlateau(
    monitor="accuracy", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)
checkpoint = ModelCheckpoint("model.h5", verbose=1, save_best_only=True)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("accuracy", 0) > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True




## === cell 6
gc.collect()

steps_per_epoch = int(np.ceil(len(training_X) / 16))

history = model.fit(
    train_datagen.flow(training_X, y_binary_train, batch_size=16, shuffle=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_X, y_binary_val),
    epochs=200,
    callbacks=[annealer],
    verbose=1,
)



## === cell 7
if "history" in globals() and hasattr(history, "history"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(len(acc))

    plt.plot(epochs, acc, "r", label="Training accuracy")
    if len(val_acc) == len(acc) and len(val_acc) > 0:
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training accuracy")
    plt.legend(loc=0)
    plt.figure()
    plt.show()
else:
    print("Skipping plots: history is not available.")



## === cell 8
if "history" in globals() and hasattr(history, "history"):
    acc = history.history.get("accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(len(loss))

    plt.plot(epochs, loss, "r", label="Training Loss")
    if len(val_loss) == len(loss) and len(val_loss) > 0:
        plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training Loss")
    plt.legend(loc=0)
    plt.figure()
    plt.show()
else:
    print("Skipping loss plot: history is not available.")



## === cell 9
gc.collect()

test_ids = test_df["image_id"].values
test_images = []
for img_id in test_ids:
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read test image: {img_path}")
    resized = cv2.resize(image, (410, 273), interpolation=cv2.INTER_AREA)
    test_images.append(resized)

test_images = np.array(test_images, dtype=np.uint8)
test_images = preprocess_input(test_images.astype(np.float32))



## === cell 10
gc.collect()

results = model.predict(test_images, batch_size=16, verbose=1)



## === cell 11
df = pd.DataFrame(results, columns=target_cols)
df.insert(0, "image_id", test_ids)
df = df[["image_id"] + target_cols]



## === cell 12
out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path} with shape {df.shape}")



## === cell 13
df.head(10)



## === cell 14
assert (
    df.shape[0] == sample_sub.shape[0]
), f"Row mismatch: {df.shape[0]} vs {sample_sub.shape[0]}"
assert list(df.columns) == list(
    sample_sub.columns
), f"Column mismatch: {df.columns} vs {sample_sub.columns}"
df.describe(include="all")

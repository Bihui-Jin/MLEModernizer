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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9904

# 6. Current score

0.5092

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5092) has done: 'I fixed the TensorFlow‑1 style code, replaced the broken `tf.data` pipeline with a simple NumPy loader, updated the optimizer arguments, corrected the loss function, and used `model.predict` instead of the non‑existent `predict_proba`. The script now loads the images, trains the model, plots the curves, makes predictions on the test set, and writes a proper Kaggle submission file (`submission.csv`). All changes are minimal and keep the original model architecture.'

# 9. Code solution

## === cell 0
import os
import json
import logging
import random

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
random.seed(1372)
np.random.seed(1372)
tf.random.set_seed(1372)



## === cell 2
data_dir = r"../input"
train_dir = os.path.join(data_dir, "train", "train")
csv_path = os.path.join(data_dir, "train.csv")

df = pd.read_csv(csv_path)
df["full_path"] = df["id"].apply(lambda x: os.path.join(train_dir, x))
filenames = df["full_path"].values
labels = df["has_cactus"].astype(np.float32).values


def load_images(paths):
    imgs = []
    for p in tqdm_notebook(paths, desc="Loading train images"):
        img = cv.imread(p)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # optional, keep RGB order
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32) / 255.0
    return imgs


X = load_images(filenames)



## === cell 3
split_idx = int(len(X) * 0.15)
X_val, y_val = X[:split_idx], labels[:split_idx]
X_train, y_train = X[split_idx:], labels[split_idx:]



## === cell 4
model = Sequential()
model.add(Conv2D(3, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))

model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=32, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation="elu"))
model.add(Dense(1, activation="sigmoid"))



## === cell 5
optimizer = Adam(learning_rate=0.0015, amsgrad=True)
model.compile(
    optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]
)
model.summary()



## === cell 6
checkpoint_path = "weights-aerial-cactus.h5"
callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, verbose=1, mode="min"
    ),
    EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    ),
]



## === cell 7
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)




## === cell 8
def plot_training_curves(hist):
    acc = hist.history.get("accuracy", [])
    val_acc = hist.history.get("val_accuracy", [])
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, "r", label="Training acc")
    plt.plot(epochs, val_acc, "g", label="Validation acc")
    plt.title("Accuracy")
    plt.legend()
    plt.show()


plot_training_curves(history)



## === cell 9
test_csv_path = os.path.join(data_dir, "sample_submission.csv")
test_df = pd.read_csv(test_csv_path)

test_dir = os.path.join(data_dir, "test", "test")
test_images = test_df["id"].values

X_test = []
for img_id in tqdm_notebook(test_images, desc="Loading test images"):
    img_path = os.path.join(test_dir, img_id)
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {img_path}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    X_test.append(img)

X_test = np.asarray(X_test, dtype=np.float32) / 255.0

y_pred = model.predict(X_test, batch_size=64).reshape(-1)

submission = pd.DataFrame({"id": test_images, "has_cactus": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

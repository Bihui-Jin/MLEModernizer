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

No external packages required in the script and installed.

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

0.9989901666666666

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm, tqdm_notebook
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
possible_paths = [
    os.path.abspath(os.path.join(os.getcwd(), "input", "aerial-cactus-identification")),
    os.path.abspath(os.path.join(os.getcwd(), "input")),
]
for p in possible_paths:
    if os.path.isdir(p):
        base_path = p
        break
else:
    raise FileNotFoundError("Unable to locate the input directory.")

train_img_dir = os.path.join(base_path, "train")
test_img_dir = os.path.join(base_path, "test")
train_csv_path = os.path.join(base_path, "train.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")

print("Base path:", base_path)
print("Train images:", train_img_dir)
print("Test images:", test_img_dir)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2791535516.py in <cell line: 0>()
      9         break
     10 else:
---> 11     raise FileNotFoundError("Unable to locate the input directory.")
     12 
     13 train_img_dir = os.path.join(base_path, "train")

FileNotFoundError: Unable to locate the input directory.

## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["full_path"] = train_df["id"].apply(lambda x: os.path.join(train_img_dir, x))

X = []
for fp in tqdm(train_df["full_path"].values, desc="Loading train images"):
    img = cv.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Image not found: {fp}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    X.append(img.astype("float32") / 255.0)
X = np.stack(X, axis=0)  # shape (N, 32, 32, 3)

y = train_df["has_cactus"].astype(np.float32).values



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/653604617.py in <cell line: 0>()
      1 # Load training metadata
----> 2 train_df = pd.read_csv(train_csv_path)
      3 train_df["full_path"] = train_df["id"].apply(lambda x: os.path.join(train_img_dir, x))
      4 
      5 # Read and preprocess training images

NameError: name 'train_csv_path' is not defined

## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/563541522.py in <cell line: 0>()
      1 # Train‑validation split
      2 X_train, X_val, y_train, y_val = train_test_split(
----> 3     X, y, test_size=0.15, random_state=42, stratify=y
      4 )
      5 

NameError: name 'X' is not defined

## === cell 4
model = Sequential()
model.add(Conv2D(3, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))

model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=32, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=256, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True))
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True))
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=1024, kernel_size=1, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))



## === cell 5
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()



## === cell 6
ckpt_path = "weights-aerial-cactus.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min", min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_auc", patience=10, mode="max", verbose=1, restore_best_weights=True
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2121292757.py in <cell line: 0>()
     15 # Training
     16 history = model.fit(
---> 17     X_train,
     18     y_train,
     19     validation_data=(X_val, y_val),

NameError: name 'X_train' is not defined

## === cell 7
test_df = pd.read_csv(sample_sub_path)
test_ids = test_df["id"].values
test_images = []
for img_id in tqdm(test_ids, desc="Loading test images"):
    fp = os.path.join(test_img_dir, img_id)
    img = cv.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {fp}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    test_images.append(img.astype("float32") / 255.0)
X_test = np.stack(test_images, axis=0)

preds = model.predict(X_test, batch_size=32, verbose=0).reshape(-1)

submission = pd.DataFrame({"id": test_ids, "has_cactus": preds})
submission_path = "aerial-cactus-submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3047910435.py in <cell line: 0>()
      1 # Load test data and predict
----> 2 test_df = pd.read_csv(sample_sub_path)
      3 test_ids = test_df["id"].values
      4 test_images = []
      5 for img_id in tqdm(test_ids, desc="Loading test images"):

NameError: name 'sample_sub_path' is not defined

## === cell 8
def plot_training_curves(hist):
    import matplotlib.pyplot as plt

    epochs = range(1, len(hist.history["loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["loss"], "r", label="Train loss")
    plt.plot(epochs, hist.history["val_loss"], "g", label="Val loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history["accuracy"], "r", label="Train acc")
    plt.plot(epochs, hist.history["val_accuracy"], "g", label="Val acc")
    plt.title("Accuracy")
    plt.legend()
    plt.show()


plot_training_curves(history)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/924998196.py in <cell line: 0>()
     20 
     21 
---> 22 plot_training_curves(history)

NameError: name 'history' is not defined

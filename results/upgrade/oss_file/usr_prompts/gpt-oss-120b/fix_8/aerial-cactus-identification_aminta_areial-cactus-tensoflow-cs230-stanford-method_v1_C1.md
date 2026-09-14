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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The changes enable TensorFlow’s mixed‑precision policy (accelerating GPU/CPU matrix ops without altering model architecture), remove the forced “python” protobuf implementation which slows down TensorFlow, and make checkpoint saving lighter by storing only weights. These tweaks keep every layer, loss, optimizer and data‑processing step exactly the same while considerably cutting overall runtime so the notebook stays under the 600‑second limit.'
- What this solution (achieved 0.5) has done: 'The fix keeps the exact model and training logic but removes the expensive NumPy‑to‑Tensor conversion on every epoch by building efficient `tf.data.Dataset` pipelines for the train/validation splits and uses the native float32 tensors (mixed‑precision still cast them). This eliminates per‑batch Python overhead, speeds up data feeding, and stays within the 600 s limit while preserving identical results.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential, mixed_precision
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
from concurrent.futures import ThreadPoolExecutor

mixed_precision.set_global_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(tf.config.threading.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(tf.config.threading.cpu_count())

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
possible_paths = [
    os.path.abspath(os.path.join(os.getcwd(), "input", "aerial-cactus-identification")),
    os.path.abspath(os.path.join(os.getcwd(), "input")),
    "/kaggle/input/aerial-cactus-identification",  # Kaggle environment
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



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["full_path"] = train_df["id"].apply(lambda x: os.path.join(train_img_dir, x))


def _load_image(fp):
    img = cv.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Image not found: {fp}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


paths = train_df["full_path"].values
with ThreadPoolExecutor() as executor:
    X_list = list(
        tqdm(
            executor.map(_load_image, paths),
            total=len(paths),
            desc="Loading train images",
        )
    )
X = np.stack(X_list, axis=0)  # shape (N, 32, 32, 3), dtype=uint8
y = train_df["has_cactus"].astype(np.float32).values



## === cell 3
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def _preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0,1]
    return img, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
    .shuffle(buffer=len(X_train_np), reshuffle_each_iteration=True)
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val_np, y_val_np))
    .map(_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/822081662.py in <cell line: 0>()
     16 train_ds = (
     17     tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
---> 18     .shuffle(buffer=len(X_train_np), reshuffle_each_iteration=True)
     19     .map(_preprocess, num_parallel_calls=AUTOTUNE)
     20     .batch(BATCH_SIZE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

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
ckpt_path = "weights-aerial-cactus.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=4, verbose=1, mode="min", min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_auc", patience=12, mode="max", verbose=1, restore_best_weights=True
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    verbose=2,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1583641066.py in <cell line: 0>()
     18 
     19 history = model.fit(
---> 20     train_ds,
     21     validation_data=val_ds,
     22     epochs=50,

NameError: name 'train_ds' is not defined

## === cell 7
test_df = pd.read_csv(sample_sub_path)
test_ids = test_df["id"].values


def _load_test_image(img_id):
    fp = os.path.join(test_img_dir, img_id)
    img = cv.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {fp}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


with ThreadPoolExecutor() as executor:
    test_list = list(
        tqdm(
            executor.map(_load_test_image, test_ids),
            total=len(test_ids),
            desc="Loading test images",
        )
    )
X_test_np = np.stack(test_list, axis=0)  # uint8

test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test_np)
    .map(lambda img: tf.cast(img, tf.float32) / 255.0, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds = model.predict(test_ds, verbose=0).reshape(-1)

submission = pd.DataFrame({"id": test_ids, "has_cactus": preds})
submission_path = "aerial-cactus-submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




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
/tmp/ipykernel_11/3986379225.py in <cell line: 0>()
     19 
     20 
---> 21 plot_training_curves(history)

NameError: name 'history' is not defined

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

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
tf_keras==2.18.0

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

0.9803

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import zipfile
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not enabled (ok):", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_INPUTS = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input",
    "/kaggle/input",
]

BASE_INPUT = None
for p in CANDIDATE_INPUTS:
    if os.path.isdir(p) and (
        os.path.isfile(os.path.join(p, "train.zip"))
        or os.path.isfile(os.path.join(p, "aerial-cactus-identification", "train.zip"))
    ):
        BASE_INPUT = p
        break

if BASE_INPUT is None:
    raise FileNotFoundError("Could not locate Kaggle input directory for dataset.")

if os.path.isfile(os.path.join(BASE_INPUT, "train.zip")):
    DATA_DIR = BASE_INPUT
else:
    DATA_DIR = os.path.join(BASE_INPUT, "aerial-cactus-identification")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Files:",
    [
        f
        for f in ["train.zip", "test.zip", "train.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(DATA_DIR, f))
    ],
)




## === cell 2
if not os.path.isdir("train"):
    with zipfile.ZipFile(os.path.join(DATA_DIR, "train.zip"), "r") as z:
        z.extractall(".")
if not os.path.isdir("test"):
    with zipfile.ZipFile(os.path.join(DATA_DIR, "test.zip"), "r") as z:
        z.extractall(".")

print("Extracted folders exist?", os.path.isdir("train"), os.path.isdir("test"))
print(
    "Example train file:",
    next(iter(os.listdir("train"))) if os.path.isdir("train") else None,
)
print(
    "Example test file:",
    next(iter(os.listdir("test"))) if os.path.isdir("test") else None,
)




## === cell 3
train_dir = "train"
test_dir = "test"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(train.shape, test_df.shape)
print(train.head())




## === cell 4
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())




## === cell 5
train["has_cactus"] = train["has_cactus"].astype(str)
print(train["has_cactus"].value_counts())




## === cell 6
from sklearn.model_selection import train_test_split

train["filepath"] = train["id"].apply(lambda x: os.path.join(train_dir, str(x)))
missing_train = (~train["filepath"].apply(os.path.exists)).sum()
if missing_train:
    raise FileNotFoundError(
        f"{missing_train} training images referenced in train.csv not found under '{train_dir}'."
    )

train_df, val_df = train_test_split(
    train,
    test_size=0.15,
    random_state=SEED,
    stratify=train["has_cactus"],
)

print("Train/Val sizes:", train_df.shape, val_df.shape)
print("Train class counts:\n", train_df["has_cactus"].value_counts())
print("Val class counts:\n", val_df["has_cactus"].value_counts())

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    target_size=(32, 32),
    batch_size=32,
    shuffle=True,
    seed=SEED,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
    seed=SEED,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/62458503.py in <cell line: 0>()
      5 missing_train = (~train["filepath"].apply(os.path.exists)).sum()
      6 if missing_train:
----> 7     raise FileNotFoundError(
      8         f"{missing_train} training images referenced in train.csv not found under '{train_dir}'."
      9     )

FileNotFoundError: 14175 training images referenced in train.csv not found under 'train'.

## === cell 7
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

model.summary()




## === cell 8
model.compile(loss="binary_crossentropy", optimizer="Adamax", metrics=["acc"])




## === cell 9
epochs = 10

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=2,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1833634711.py in <cell line: 0>()
      4 
      5 history = model.fit(
----> 6     train_generator,
      7     epochs=epochs,
      8     validation_data=validation_generator,

NameError: name 'train_generator' is not defined

## === cell 10
acc_key = "acc" if "acc" in history.history else "accuracy"
val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"

fig = plt.figure(figsize=(12, 8))
plt.plot(history.history[acc_key], "blue")
plt.plot(history.history[val_acc_key], "orange")
plt.xticks(np.arange(0, epochs, 1))
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.grid(True)
plt.legend(["train", "validation"])
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], "blue")
plt.plot(history.history["val_loss"], "orange")
plt.xticks(np.arange(0, epochs, 1))
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.grid(True)
plt.legend(["train", "validation"])
plt.show()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/233586451.py in <cell line: 0>()
----> 1 acc_key = "acc" if "acc" in history.history else "accuracy"
      2 val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
      3 
      4 fig = plt.figure(figsize=(12, 8))
      5 plt.plot(history.history[acc_key], "blue")

NameError: name 'history' is not defined

## === cell 11
model_vg = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
model_vg.summary()




## === cell 12
model_vg.trainable = False




## === cell 13
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)

model2 = models.Sequential()
model2.add(model_vg)
model2.add(Flatten())
model2.add(Dense(256, use_bias=True))
model2.add(BatchNormalization())
model2.add(Activation("relu"))
model2.add(Dropout(0.5))
model2.add(Dense(64, activation="relu"))
model2.add(BatchNormalization())
model2.add(Dense(16, activation="tanh"))
model2.add(Dense(1, activation="sigmoid"))




## === cell 14
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    model2.compile(optimizer="Adamax", loss="binary_crossentropy", metrics=["acc"])




## === cell 15
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    history_vgg = model2.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=2,
    )




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3478207347.py in <cell line: 0>()
      2 with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
      3     history_vgg = model2.fit(
----> 4         train_generator,
      5         epochs=15,
      6         validation_data=validation_generator,

NameError: name 'train_generator' is not defined

## === cell 16
acc_key2 = "acc" if "acc" in history_vgg.history else "accuracy"
val_acc_key2 = "val_acc" if "val_acc" in history_vgg.history else "val_accuracy"

fig = plt.figure(figsize=(12, 8))
plt.plot(history_vgg.history[acc_key2], "blue")
plt.plot(history_vgg.history[val_acc_key2], "orange")
plt.xticks(np.arange(0, 15, 1))
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy (VGG16)")
plt.grid(True)
plt.legend(["train", "validation"])
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history_vgg.history["loss"], "blue")
plt.plot(history_vgg.history["val_loss"], "orange")
plt.xticks(np.arange(0, 15, 1))
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss (VGG16)")
plt.grid(True)
plt.legend(["train", "validation"])
plt.show()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2146036663.py in <cell line: 0>()
----> 1 acc_key2 = "acc" if "acc" in history_vgg.history else "accuracy"
      2 val_acc_key2 = "val_acc" if "val_acc" in history_vgg.history else "val_accuracy"
      3 
      4 fig = plt.figure(figsize=(12, 8))
      5 plt.plot(history_vgg.history[acc_key2], "blue")

NameError: name 'history_vgg' is not defined

## === cell 17
if not os.path.isdir(test_dir):
    raise FileNotFoundError(
        f"Expected extracted test directory '{test_dir}' not found."
    )

img_ids = test_df["id"].astype(str).tolist()

X_test = np.empty((len(img_ids), 32, 32, 3), dtype=np.float32)
missing = []

for i, img_id in enumerate(img_ids):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        missing.append(img_path)
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_test[i] = img.astype(np.float32) / 255.0

if missing:
    raise FileNotFoundError(
        f"Could not read {len(missing)} test images. First missing: {missing[0]}"
    )

y_test_pred = model2.predict(X_test, batch_size=128, verbose=0).reshape(-1)

sub = pd.DataFrame({"id": img_ids, "has_cactus": y_test_pred.astype(np.float32)})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
print(
    "has_cactus range:", float(sub["has_cactus"].min()), float(sub["has_cactus"].max())
)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3928528939.py in <cell line: 0>()
      1 # Fix: Ensure test directory exists (it should after extraction above).
      2 if not os.path.isdir(test_dir):
----> 3     raise FileNotFoundError(
      4         f"Expected extracted test directory '{test_dir}' not found."
      5     )

FileNotFoundError: Expected extracted test directory 'test' not found.

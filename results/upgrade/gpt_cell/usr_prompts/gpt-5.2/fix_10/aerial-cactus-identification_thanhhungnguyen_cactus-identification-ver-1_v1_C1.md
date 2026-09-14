# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
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

0.9881

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    from packaging.version import Version
    import google.protobuf  # noqa: F401

    _pb_ver = getattr(google.protobuf, "__version__", None)
    if _pb_ver is not None and Version(_pb_ver) >= Version("6.0.0"):
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<6"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import *

from tf_keras.preprocessing.image import ImageDataGenerator

import zipfile



## === cell 1
train_data = pd.read_csv(
    "../input/aerial-cactus-identification/train.csv",
    dtype={"id": str, "has_cactus": int},
)
test_data = pd.read_csv(
    "../input/aerial-cactus-identification/sample_submission.csv",
    dtype={"id": str, "has_cactus": float},
)



## === cell 2
train_data.head()



## === cell 3
train_data.describe()



## === cell 4
test_data.describe()



## === cell 5
zip_ref_1 = zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/test.zip")
zip_ref_1.extractall()



## === cell 6
zip_ref_2 = zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/train.zip")
zip_ref_2.extractall()



## === cell 7
train_path = "train/"
test_path = "test/"
print("Training Images:", len(os.listdir(train_path)))
print("Testing Images: ", len(os.listdir(test_path)))



## === cell 8
train_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.20)
test_datagen = ImageDataGenerator(rescale=1 / 255)



## === cell 9
bs = 100

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="has_cactus",
    subset="training",
    batch_size=bs,
    shuffle=True,
    class_mode="binary",
    target_size=(32, 32),
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="has_cactus",
    subset="validation",
    batch_size=bs,
    shuffle=True,
    class_mode="binary",
    target_size=(32, 32),
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=test_path,
    x_col="id",
    y_col=None,
    batch_size=bs,
    seed=1,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32),
)



## === cell 10
tr_steps = math.ceil(train_generator.n / bs)
va_steps = math.ceil(valid_generator.n / bs)

print("Train/Val/Test sizes:", train_generator.n, valid_generator.n, test_generator.n)
print("Steps:", tr_steps, va_steps, len(test_generator))



## === cell 11
cnn = Sequential()

cnn.add(Conv2D(28, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
cnn.add(Conv2D(28, (3, 3), activation="relu", padding="same"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(BatchNormalization())

cnn.add(Conv2D(56, (3, 3), activation="relu", padding="same"))
cnn.add(Conv2D(56, (3, 3), activation="relu", padding="same"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(BatchNormalization())

cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(BatchNormalization())

cnn.add(Dense(1, activation="sigmoid"))

cnn.summary()



## === cell 12
opt = keras.optimizers.Adam(0.001)
cnn.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

h1 = cnn.fit(
    train_generator,
    steps_per_epoch=tr_steps,
    epochs=80,
    validation_data=valid_generator,
    validation_steps=va_steps,
    verbose=1,
)



## === cell 13
start = 1
ep_rng = np.arange(start, len(h1.history["accuracy"]))

plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(ep_rng, h1.history["accuracy"][start:], label="Training Accuracy")
plt.plot(ep_rng, h1.history["val_accuracy"][start:], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ep_rng, h1.history["loss"][start:], label="Training Loss")
plt.plot(ep_rng, h1.history["val_loss"][start:], label="Validation Loss")
plt.xlabel("Epoch")
plt.legend()

plt.show()



## === cell 14
test_pred = cnn.predict(test_generator, steps=len(test_generator), verbose=1)



## === cell 15
pos_proba = test_pred.reshape(-1).astype(np.float64)

submission = pd.DataFrame({"id": test_data["id"].values, "has_cactus": pos_proba})

print(
    "Prediction stats:",
    float(pos_proba.min()),
    float(pos_proba.max()),
    float(pos_proba.mean()),
)
print("Num submission ids:", len(submission), "Num predictions:", len(pos_proba))

assert len(submission) == len(
    test_data
), "Submission length mismatch with sample submission."
assert submission["id"].iloc[0] == test_data["id"].iloc[0], "ID order mismatch."

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 16
import shutil

shutil.rmtree("/kaggle/working/train")
shutil.rmtree("/kaggle/working/test")

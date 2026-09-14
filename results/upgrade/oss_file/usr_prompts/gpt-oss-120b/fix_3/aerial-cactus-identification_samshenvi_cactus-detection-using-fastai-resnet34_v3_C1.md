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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
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

0.6398

# 6. Current score

0.87433

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.98439) has done: 'I fixed the import errors by switching to TensorFlow‑Keras, corrected the paths for training and test data, handled possible missing images, rebuilt the ResNet‑50 model using the proper input shape, and created a proper submission file that contains the image ids and raw probability predictions (required for AUC). The changes keep the original architecture and training loop while making the script runnable end‑to‑end and producing a valid `submission.csv`.'
- What this solution (achieved 0.87433) has done: 'I add a protobuf compatibility fix by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` before importing TensorFlow, and I renumber the notebook cells so they start at 1 as required. No other logic changes are needed; the model and training remain identical, preserving the high AUC while ensuring the script runs without the earlier import error and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import numpy as np
import pandas as pd
import cv2
from tqdm.notebook import tqdm
from pathlib import Path

BASE_PATH = Path("/kaggle/input/aerial-cactus-identification")
TRAIN_CSV = BASE_PATH / "train.csv"
TEST_DIR = BASE_PATH / "test"
TRAIN_DIR = BASE_PATH / "train"

print("Base path contents:", list(BASE_PATH.iterdir()))




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", train_df.shape[0])




## === cell 2
X_tr = []
Y_tr = []
for img_id in tqdm(train_df["id"].values, desc="Loading train images"):
    img_path = TRAIN_DIR / img_id
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_tr.append(img)
    Y_tr.append(train_df.loc[train_df["id"] == img_id, "has_cactus"].values[0])

X_tr = np.asarray(X_tr, dtype="float32") / 255.0
Y_tr = np.asarray(Y_tr, dtype="float32")
print("X_tr shape:", X_tr.shape, "Y_tr shape:", Y_tr.shape)




## === cell 3
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    ZeroPadding2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(s, s), padding="valid", name=conv_name_base + "2a")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(F2, (f, f), strides=(1, 1), padding="same", name=conv_name_base + "2b")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2c")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X_shortcut = Conv2D(
        F3, (1, 1), strides=(s, s), padding="valid", name=conv_name_base + "1"
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X




## === cell 5
def identity_block(X, f, filters, stage, block):
    conv_name_base = f"res{stage}{block}_branch"
    bn_name_base = f"bn{stage}{block}_branch"
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2a")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(F2, (f, f), strides=(1, 1), padding="same", name=conv_name_base + "2b")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=conv_name_base + "2c")(
        X
    )
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X




## === cell 6
def Resnet50_body(input_tensor, classes=1):
    X = ZeroPadding2D((3, 3))(input_tensor)

    X = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="b")
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="c")

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="b")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="c")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="d")

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="b")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="c")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="d")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="e")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="f")
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="b")
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="c")

    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)
    X = Flatten()(X)
    X = Dense(classes, activation="sigmoid", name="fc" + str(classes))(X)

    return X




## === cell 7
input_shape = X_tr.shape[1:]  # (32, 32, 3)
inputs = Input(shape=input_shape)
outputs = Resnet50_body(inputs, classes=1)
model = Model(inputs, outputs, name="Resnet50")
model.summary()




## === cell 8
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
gc.collect()




## === cell 9
model.fit(X_tr, Y_tr, epochs=20, batch_size=128, verbose=1)




## === cell 10
X_tst = []
Test_imgs = []
for img_name in tqdm(os.listdir(TEST_DIR), desc="Loading test images"):
    img_path = TEST_DIR / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_tst.append(img)
    Test_imgs.append(img_name)

X_tst = np.asarray(X_tst, dtype="float32") / 255.0
print("Test set shape:", X_tst.shape, "Number of ids:", len(Test_imgs))




## === cell 11
test_predictions = model.predict(X_tst, batch_size=128, verbose=0).flatten()
print("Predictions shape:", test_predictions.shape)




## === cell 12
sub_df = pd.DataFrame({"id": Test_imgs, "has_cactus": test_predictions})
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, rows:", sub_df.shape[0])

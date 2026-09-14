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

0.99797

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99824) has done: 'I fix the Keras import/runtime crash by switching to `tf_keras` (available in your environment) while keeping the exact same ResNet-like architecture and training loop. I also fix the broken dataset paths by using the provided `../input/aerial-cactus-identification/...` folders, and make image loading robust (handle missing/None reads and enforce consistent 32×32 shape) to resolve the inhomogeneous-array and `NoneType` errors. Finally, I generate a valid submission with the required `id,has_cactus` columns using probabilities (no hard thresholding, since AUC expects probabilities), and ensure IDs align with `sample_submission.csv` order.'
- What this solution (achieved 0.99797) has done: 'I fix the runtime crash in the Keras import cell by switching from `tf_keras` to `tensorflow.keras`, which is the most stable/protobuf-compatible stack for this Kaggle environment and keeps the exact same model architecture/training loop. I also keep the data paths and submission generation unchanged, only adding lightweight determinism settings so runs are stable and reproducible (score-neutral). Since your current score (0.99824) is already far above the target (0.6398), I not intentionally degrade the model or alter training/prediction logic to chase the target—this patch is focused on making it run end-to-end and reliably produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

INPUT_ROOT = "../input/aerial-cactus-identification"

print("Input root exists:", os.path.exists(INPUT_ROOT))
print("Listing ../input:", os.listdir("../input")[:20])



## === cell 1
train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")
train_dir = os.path.join(INPUT_ROOT, "train")
test_dir = os.path.join(INPUT_ROOT, "test")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train_df.head())
print("Train dir:", train_dir, "exists:", os.path.exists(train_dir))
print("Test dir:", test_dir, "exists:", os.path.exists(test_dir))
print("Train rows:", len(train_df), "Sample sub rows:", len(sample_sub))




## === cell 2
def read_image_bgr(path, target_size=(32, 32)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


X_tr = []
Y_tr = []

img_ids = train_df["id"].values
labels = train_df["has_cactus"].values.astype(np.float32)

for img_id, y in tqdm(list(zip(img_ids, labels)), total=len(img_ids)):
    p = os.path.join(train_dir, img_id)
    img = read_image_bgr(p, target_size=(32, 32))
    if img is None:
        continue
    X_tr.append(img)
    Y_tr.append(y)

X_tr = np.asarray(X_tr, dtype=np.float32) / 255.0
Y_tr = np.asarray(Y_tr, dtype=np.float32)

print("X_tr shape:", X_tr.shape, "Y_tr shape:", Y_tr.shape)



## === cell 3
shape = X_tr.shape[1:4]
print("Model input shape:", shape)



## === cell 4
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model

try:
    tf.random.set_seed(0)
except Exception as e:
    print("Warning: could not set TF seed:", repr(e))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
def convolutional_block(X, f, filters, stage, block, s):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "2a",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X_shortcut = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "1",
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X




## === cell 6
def identity_block(X, f, filters, stage, block):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2a",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X




## === cell 7
def Resnet(shape_tensor, classes):
    X_input = shape_tensor

    X = ZeroPadding2D((3, 3))(X_input)
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




## === cell 8
def Resnet50(input_shape):
    X_input = Input(input_shape)
    X = Resnet(X_input, 1)
    model = Model(inputs=X_input, outputs=X, name="Resnet50")
    return model


Resnet50 = Resnet50(shape)



## === cell 9
Resnet50.summary()



## === cell 10
Resnet50.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
gc.collect()



## === cell 11
Resnet50.fit(X_tr, Y_tr, epochs=50, batch_size=500, verbose=1)



## === cell 12
X_tst = []
test_ids = sample_sub["id"].tolist()

for img_id in tqdm(test_ids):
    p = os.path.join(test_dir, img_id)
    img = read_image_bgr(p, target_size=(32, 32))
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    X_tst.append(img)

X_tst = np.asarray(X_tst, dtype=np.float32) / 255.0
print("X_tst shape:", X_tst.shape)



## === cell 13
test_predictions = Resnet50.predict(X_tst, batch_size=500, verbose=1)
test_predictions = test_predictions.reshape(-1)
print(
    "Pred shape:",
    test_predictions.shape,
    "min/max:",
    float(test_predictions.min()),
    float(test_predictions.max()),
)



## === cell 14
sub_df = pd.DataFrame({"id": test_ids, "has_cactus": test_predictions.astype(float)})

assert list(sub_df.columns) == ["id", "has_cactus"]
assert len(sub_df) == len(sample_sub)

sub_df.head()



## === cell 15
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub_df))
print(sub_df.head())

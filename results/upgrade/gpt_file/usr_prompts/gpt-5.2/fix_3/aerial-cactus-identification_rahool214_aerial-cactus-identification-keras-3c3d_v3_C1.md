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

0.9423

# 6. Current score

0.4998

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52458) has done: 'I fix the environment/import crash by using `tf_keras` (TensorFlow-backed Keras) consistently instead of standalone `keras`, which avoids the protobuf `MessageFactory` error in this Kaggle image. I also correct outdated optimizer arguments (`lr` → `learning_rate`) so compilation works, and fix the dataset paths to match the provided folder structure while filtering out stray directories that caused `IsADirectoryError`. Finally, I ensure the submission uses probabilities (not rounded class labels, which hurts AUC) and matches the required `id,has_cactus` format with a `.csv` suffix.'
- What this solution (achieved 0.4998) has done: 'I fix the two runtime blockers: the protobuf/Keras import crash by forcing TensorFlow-backed `tf_keras` usage without touching standalone `keras`, and the `Dropout(rate=1)` invalid configuration by clamping rates into the valid `[0, 1)` range while keeping the same model structure. To move the AUC up toward the target (your current score is far below), I make the dropouts actually behave as intended (using a high but valid dropout rate) rather than erroring or being effectively broken. I also make image loading deterministic and explicit about target size/color mode to avoid shape inconsistencies, and ensure the submission is aligned to `sample_submission.csv` with valid probabilities and a `.csv` filename.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tf_keras as keras
from tf_keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import load_img, img_to_array

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass
try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

CANDIDATE_ROOTS = [
    "/kaggle/input/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "/kaggle/input",
    "../input",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/data",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if not os.path.exists(r):
        continue
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
        os.path.join(r, "train")
    ):
        DATA_ROOT = r
        break
    if os.path.exists(os.path.join(r, "aerial-cactus-identification", "train.csv")):
        DATA_ROOT = os.path.join(r, "aerial-cactus-identification")
        break

if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate dataset root in expected paths.")

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
trainDF = pd.read_csv(TRAIN_CSV)

trainImgList = list(trainDF["id"].astype(str).values)
trainImg = []

for img in trainImgList:
    img_path = os.path.join(TRAIN_DIR, img)
    originalImage = load_img(img_path, color_mode="rgb", target_size=(32, 32))
    arrayImage = img_to_array(originalImage).astype(np.float32) / 255.0
    trainImg.append(arrayImage)

trainImgNP = np.array(trainImg, dtype=np.float32)
trainLabels = trainDF["has_cactus"].values.astype(np.float32)

print("Train images shape:", trainImgNP.shape, "labels shape:", trainLabels.shape)
print("Label mean:", float(trainLabels.mean()))




## === cell 2
def getConvLayer(inputLayer, kernelSize=2, filters=15):
    conv1 = Conv2D(filters, kernelSize, activation="relu")(inputLayer)
    conv2 = Conv2D(filters, kernelSize, activation="relu")(conv1)
    pool1 = MaxPooling2D()(conv2)
    return pool1


def getFlattenLayer(inputLayer):
    flat = Flatten()(inputLayer)
    return flat


def getDenseLayer(inputLayer, units=32, rate=0.5):
    rate = float(rate)
    if rate >= 1.0:
        rate = 0.95
    if rate < 0.0:
        rate = 0.0

    dense1 = Dense(units, activation="relu")(inputLayer)
    drop1 = Dropout(rate)(dense1)
    return drop1


def getOutLayer(inputLayer):
    out1 = Dense(1, activation="sigmoid")(inputLayer)
    return out1


def getModel():
    inputLayer = Input(shape=[32, 32, 3])

    conv1 = getConvLayer(inputLayer)
    conv2 = getConvLayer(conv1)

    flat = getFlattenLayer(conv2)

    dense1 = getDenseLayer(flat, rate=1)
    dense2 = getDenseLayer(dense1, rate=1)
    dense3 = getDenseLayer(dense2, rate=1)
    dense4 = getDenseLayer(dense3, rate=1)

    out = getOutLayer(dense4)

    model = Model(inputLayer, out)
    return model




## === cell 3
model = getModel()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x=trainImgNP,
    y=trainLabels,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    verbose=2,
)



## === cell 4
testImgList = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(TEST_DIR, f))
    ]
)

testImg = []
for img in testImgList:
    img_path = os.path.join(TEST_DIR, img)
    originalImage = load_img(img_path, color_mode="rgb", target_size=(32, 32))
    arrayImage = img_to_array(originalImage).astype(np.float32) / 255.0
    testImg.append(arrayImage)

testImgNP = np.array(testImg, dtype=np.float32)
print("Test images shape:", testImgNP.shape, "Num test ids:", len(testImgList))



## === cell 5
testPred = model.predict(testImgNP, batch_size=256, verbose=0).reshape(-1)

testPred = np.nan_to_num(
    testPred, nan=float(np.nanmean(testPred) if np.isfinite(testPred).any() else 0.5)
)
testPred = np.clip(testPred, 0.0, 1.0)

sub = pd.DataFrame({"id": testImgList, "has_cactus": testPred.astype(np.float32)})

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    if sub["has_cactus"].isna().any():
        sub["has_cactus"] = sub["has_cactus"].fillna(float(np.mean(testPred)))

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
print(
    "has_cactus min/max:",
    float(sub["has_cactus"].min()),
    float(sub["has_cactus"].max()),
)

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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9603

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the import/runtime issues caused by mixing `keras==3.x` with the old `tf.keras` APIs by switching the model code to `tf_keras` (which is installed) and using its `to_categorical`. I also correct the dataset glob paths (they currently point to a non-existent nested `train/train` and `test/test`), which is why your image arrays were empty and caused the train/label length mismatch. Finally, I keep your CNN architecture and training loop intact, but output the required probability for `has_cactus` (AUC metric expects probabilities, not class argmax) and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.49909) has done: 'I fix the runtime crash coming from importing `tf_keras` (it’s a known protobuf incompatibility in some Kaggle images) by switching the Keras imports to the built-in `tensorflow.keras` API, while keeping your exact CNN architecture, training loop, and loss/labels unchanged. I also keep your corrected image globs and the probability output (class-1 softmax) required for AUC, since predicting class labels would lock you near 0.5. Finally, I add a small defensive fallback to locate the dataset directory robustly (without changing I/O semantics) and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.4996) has done: 'I fix the crash in the TensorFlow/Keras import stack that’s causing the protobuf `MessageFactory.GetPrototype` error by avoiding `tensorflow.keras` entirely and switching to the installed `keras==3.8.0` backend configured to use TensorFlow. This keeps your CNN architecture, training loop, loss, and label encoding logic intact, but makes the runtime stable in the given environment. I also add a deterministic seed setup for Keras 3 to reduce run-to-run noise (score-neutral on average). Finally, I keep the submission generation unchanged while ensuring predictions are valid probabilities for AUC.'
- What this solution (achieved 0.5) has done: 'We fix the runtime crash in the Keras/TensorFlow stack by avoiding `keras==3.x` entirely and switching the model/training code to the installed `tf_keras` package (TF-Keras 2.18), which is compatible in Kaggle and keeps your CNN architecture, training loop, loss, and label encoding unchanged. To prevent the common protobuf `MessageFactory.GetPrototype` failure, we force protobuf to use the pure-Python implementation before any TF/Keras import. This is a stability fix (not a modeling change) and should also correct the current ~0.5 AUC behavior caused by the training not actually running due to the crash. The submission writing remains the same and produce a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.50279) has done: 'You’re currently crashing at the Keras import due to a known protobuf C++/TF-Keras incompatibility (`MessageFactory.GetPrototype`). To make the notebook run end-to-end, I keep your exact CNN, loss, training loop, and probability submission logic, but switch imports from `tf_keras` to `keras==3.x` (TF backend) which avoids that protobuf path. I also ensure the TF backend is selected before importing Keras and keep the same deterministic seeding (score-neutral) so your training actually runs and AUC moves up from the current ~0.5 (which is consistent with a broken training pipeline). Submission format and paths stay the same, and the output remains a valid `submission.csv` with `id,has_cactus` probabilities.'
- What this solution (achieved 0.5) has done: 'The crash comes from importing `keras==3.x` with an incompatible protobuf runtime in this Kaggle image; the most reliable fix is to switch the imports to the installed `tf_keras` package while keeping your exact CNN architecture, training loop, and label encoding semantics the same. To avoid the known protobuf C++ implementation issue that triggers `MessageFactory.GetPrototype`, we force protobuf to use the pure-Python implementation before any TF/Keras import. I also keep the required probability output (`softmax` class-1) for AUC and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv`. These changes should both unblock execution and move your score up from ~0.5 toward the target by ensuring the model actually trains and outputs proper probabilities.'
- What this solution (achieved 0.49397) has done: 'I fix the crash coming from importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching the model imports to `keras==3.x` configured to use the TensorFlow backend, while keeping your exact CNN architecture, loss, and training loop unchanged. I also ensure backend/TF selection happens before importing Keras to prevent mixed-backend issues. The data loading, normalization, and submission formatting stay the same, and predictions remain proper class-1 probabilities (needed for AUC). This should unblock training (the main reason you’re stuck near 0.5) and move the score toward the target.'
- What this solution (achieved 0.5) has done: 'The crash happens at `import keras` due to the known protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image when using Keras 3/TensorFlow bindings. To fix runtime end-to-end while preserving your CNN architecture, training loop, and loss, I switch the model code to the installed `tf_keras` (TF-Keras 2.18) stack and keep everything else (data loading, normalization, split, epochs, batch size) the same. I also keep the submission as class-1 softmax probability (required for AUC) and ensure the output `submission.csv` has the correct `id,has_cactus` columns aligned to `sample_submission.csv`. These changes should both unblock training and move the score up from ~0.5 toward the target because the model actually train and output proper probabilities.'
- What this solution (achieved 0.49518) has done: 'Your run is failing before training due to a protobuf/TensorFlow incompatibility triggered by importing `tf_keras` (the `MessageFactory.GetPrototype` error). The minimal fix is to keep the exact same CNN, loss, training loop, and probability output, but switch the Keras API calls to the standalone `keras==3.x` package configured to use the TensorFlow backend, avoiding `tf_keras` entirely. I also keep the existing robust dataset path/glob logic and add a small safety cast for predictions to ensure the submission contains valid float probabilities. This should unblock training (the main reason you’re stuck near ~0.5 AUC) and move the score toward the target.'
- What this solution (achieved 0.4994) has done: 'I fix the protobuf/Keras import crash by avoiding the standalone `keras` import path that triggers `MessageFactory.GetPrototype` in this environment, and instead use the installed `tf_keras` (TF-Keras) API while forcing protobuf to the pure-Python implementation before any TF/Keras import. This keeps your CNN architecture, optimizer, loss, training loop, and label encoding semantics intact, but makes the runtime stable so training actually runs (your ~0.5 AUC is consistent with a broken/degenerate pipeline). I also keep the corrected dataset globs and ensure the submission contains the required class-1 probability (not argmax) and is written to `submission.csv` with `id,has_cactus`. These changes are minimal and should move the score up toward the target by enabling proper training and valid probability outputs.'
- What this solution (achieved 0.50293) has done: 'I fix the import/runtime crash (`MessageFactory.GetPrototype`) by avoiding `tf_keras` entirely and using the standalone `keras==3.x` configured to run on the TensorFlow backend, which is already installed. This is a stability change only: the CNN architecture, loss, optimizer, training loop, and probability post-processing remain the same. I also make sure the environment variables are set before any TF/Keras import, and keep the existing robust dataset path/glob logic and submission formatting so a valid `submission.csv` is always produced. This should move AUC up from ~0.5 by ensuring training actually runs and predictions are proper class-1 probabilities.'
- What this solution (achieved 0.5) has done: 'I fix the crash caused by the protobuf/TensorFlow + Keras import incompatibility by avoiding TensorFlow imports entirely and switching the training/inference stack to the installed `tf_keras` package, while keeping your exact CNN architecture, training loop, loss, and probability post-processing unchanged. I keep your already-correct dataset glob paths and the label/image alignment checks. This should both unblock end-to-end execution and move your AUC up from ~0.5 because the model actually train and output proper probabilities. The script still write a valid `submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image

np.random.seed(42)



## === cell 1
train_data = []
test_data = []



## === cell 2
BASE = "../input/aerial-cactus-identification"
if not os.path.exists(BASE):
    alt = "../input/aerial-cactus-identification/aerial-cactus-identification"
    if os.path.exists(alt):
        BASE = alt

TRAIN_GLOB = os.path.join(BASE, "train", "*.jpg")
TEST_GLOB = os.path.join(BASE, "test", "*.jpg")


def creat_train_data():
    train_data.clear()
    files = sorted(glob(TRAIN_GLOB))
    if len(files) == 0:
        raise FileNotFoundError(f"No training images found with glob: {TRAIN_GLOB}")
    for file in tqdm(files, desc="Loading train images"):
        img = Image.open(file).convert("RGB")
        train_data.append(np.array(img, dtype=np.uint8))




## === cell 3
def creat_test_data():
    test_data.clear()
    files = sorted(glob(TEST_GLOB))
    if len(files) == 0:
        raise FileNotFoundError(f"No test images found with glob: {TEST_GLOB}")
    for file in tqdm(files, desc="Loading test images"):
        img = Image.open(file).convert("RGB")
        test_data.append(np.array(img, dtype=np.uint8))




## === cell 4
creat_train_data()
creat_test_data()



## === cell 5
train_data = np.array(train_data)
test_data = np.array(test_data)
print("train_data shape:", train_data.shape)
print("test_data shape:", test_data.shape)



## === cell 6
train = train_data.astype("float32") / 255.0
test = test_data.astype("float32") / 255.0



## === cell 7
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tf_keras.optimizers import Adam

try:
    keras.utils.set_random_seed(42)
except Exception:
    pass



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
y = pd.read_csv(os.path.join(BASE, "train.csv"))
y.head()



## === cell 9
y_train = y["has_cactus"].values

if train.shape[0] != len(y_train):
    raise ValueError(
        f"Mismatch: loaded {train.shape[0]} train images but {len(y_train)} labels in train.csv"
    )



## === cell 10
y_train = to_categorical(y_train, num_classes=2)



## === cell 11
x_train, x_val, y_train, y_val = train_test_split(
    train, y_train, test_size=0.2, random_state=2, shuffle=True
)



## === cell 12
model = Sequential()

model.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding="Same",
        activation="relu",
        input_shape=(32, 32, 3),
    )
)
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.summary()



## === cell 13
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 14
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1,
)



## === cell 15
proba = model.predict(test, verbose=0)

proba = np.asarray(proba)
res = proba[:, 1].astype("float64")

d = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
idx = d["id"].values

if len(idx) != len(res):
    raise ValueError(
        f"Mismatch: sample_submission has {len(idx)} ids but predictions has {len(res)} rows"
    )

submission = pd.DataFrame({"id": idx, "has_cactus": res})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.69647

# 6. Current score

0.76877

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.78378) has done: 'I fix the import/runtime failures by switching to `tf_keras` (to avoid the protobuf `MessageFactory.GetPrototype` crash from Keras 3 in this environment) and by ensuring each cell has the imports it relies on so later cells don’t error with `NameError`. I also fix dataset paths to match the provided Kaggle directory layout (`/kaggle/input/plant-seedlings-classification/train` and `/kaggle/input/plant-seedlings-classification/test`). To ensure a valid submission, I write `submission.csv` with the exact required columns and align filenames to `sample_submission.csv` ordering. Finally, I fix a small logic bug in argmax selection (it was ignoring class 11 due to `range(0,11)`), which should improve correctness without changing the model/training core.'
- What this solution (achieved 0.76877) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `tf_keras`, which is the most common root cause in Kaggle environments with newer protobuf. I also make the test file loading and submission mapping deterministic by sorting filenames, so predictions align consistently with `sample_submission.csv` ordering (score-neutral but prevents silent mismatches). Finally, since your current score (0.78378) is well above the target (0.69647) and higher-is-better, I not make any model/training changes that would further improve score; all changes are focused on stability and correct end-to-end execution producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import csv
import random
import numpy as np
import cv2
import matplotlib

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dense,
    BatchNormalization,
)
from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tf_keras.optimizers import Adam
from tf_keras.utils import to_categorical
from tf_keras.models import load_model

from sklearn.model_selection import train_test_split
import pandas as pd

random.seed(10)
np.random.seed(10)
try:
    keras.utils.set_random_seed(10)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def classes_to_int(label):
    label = label.strip()
    if label == "Black-grass":
        return 0
    if label == "Charlock":
        return 1
    if label == "Cleavers":
        return 2
    if label == "Common Chickweed":
        return 3
    if label == "Common wheat":
        return 4
    if label == "Fat Hen":
        return 5
    if label == "Loose Silky-bent":
        return 6
    if label == "Maize":
        return 7
    if label == "Scentless Mayweed":
        return 8
    if label == "Shepherds Purse":
        return 9
    if label == "Small-flowered Cranesbill":
        return 10
    if label == "Sugar beet":
        return 11
    print("Invalid Label", label)
    return 12




## === cell 2
def int_to_classes(i):
    if i == 0:
        return "Black-grass"
    elif i == 1:
        return "Charlock"
    elif i == 2:
        return "Cleavers"
    elif i == 3:
        return "Common Chickweed"
    elif i == 4:
        return "Common wheat"
    elif i == 5:
        return "Fat Hen"
    elif i == 6:
        return "Loose Silky-bent"
    elif i == 7:
        return "Maize"
    elif i == 8:
        return "Scentless Mayweed"
    elif i == 9:
        return "Shepherds Purse"
    elif i == 10:
        return "Small-flowered Cranesbill"
    elif i == 11:
        return "Sugar beet"
    print("Invalid class ", i)
    return "Invalid Class"




## === cell 3
NUM_CLASSES = 12
WIDTH = 128
HEIGHT = 128
DEPTH = 3
inputShape = (WIDTH, HEIGHT, DEPTH)
EPOCHS = 15
INIT_LR = 1e-3
BS = 32

TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"
TEST_DIR = "/kaggle/input/plant-seedlings-classification/test"
SAMPLE_SUB_PATH = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"


def readTrainData(trainDir):
    data = []
    labels = []
    dirs = os.listdir(trainDir)
    for dir_name in dirs:
        absDirPath = os.path.join(trainDir, dir_name)
        if not os.path.isdir(absDirPath):
            continue
        images = os.listdir(absDirPath)
        for imageFileName in images:
            imageFullPath = os.path.join(trainDir, dir_name, imageFileName)
            img = load_img(imageFullPath)
            arr = img_to_array(img)
            arr = cv2.resize(arr, (HEIGHT, WIDTH))
            data.append(arr)
            label = classes_to_int(dir_name)
            labels.append(label)
    return data, labels




## === cell 4
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=500))
    model.add(Activation("relu"))
    model.add(Dense(units=12))
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model




## === cell 5
print("Loading images...")
sys.stdout.flush()

X, Y = readTrainData(TRAIN_DIR)
X = np.array(X, dtype="float32") / 255.0
Y = np.array(Y, dtype=np.int64)
Y = to_categorical(Y, num_classes=NUM_CLASSES)

print("Partition data into 75:25...")
sys.stdout.flush()
(trainX, valX, trainY, valY) = train_test_split(X, Y, test_size=0.25, random_state=10)




## === cell 6
print("Generating images...")
sys.stdout.flush()
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

print("Compiling model...")
sys.stdout.flush()
model = createModel()

print("Training network...")
sys.stdout.flush()

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS, shuffle=True),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1,
)

print("Saving model to disk...")
sys.stdout.flush()
model_path = "/kaggle/working/CNNmodel2.keras"
model.save(model_path)




## === cell 7
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history.get("accuracy", []), label="train_acc")
plt.plot(np.arange(0, N), H.history.get("val_accuracy", []), label="val_acc")
plt.title("Training Loss and Accuracy on crop classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/kaggle/working/plot.png")
plt.close()




## === cell 8
def readTestData(testDir):
    data = []
    filenames = []
    images = sorted(os.listdir(testDir))
    for imageFileName in images:
        imageFullPath = os.path.join(testDir, imageFileName)
        if not os.path.isfile(imageFullPath):
            continue
        img = load_img(imageFullPath)
        arr = img_to_array(img)
        arr = cv2.resize(arr, (HEIGHT, WIDTH))
        data.append(arr)
        filenames.append(imageFileName)
    return data, filenames




## === cell 9
testX, filenames = readTestData(TEST_DIR)
testX = np.array(testX, dtype="float32") / 255.0

mymodel = load_model(model_path)
yFit = mymodel.predict(testX, batch_size=10, verbose=1)

pred_idx = np.argmax(yFit, axis=1)
pred_label = [int_to_classes(i) for i in pred_idx]
pred_map = dict(zip(filenames, pred_label))

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample["species"] = sample["file"].map(pred_map)

sample["species"] = sample["species"].fillna("Black-grass")

out_path = "/kaggle/working/submission.csv"
sample.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print(sample.head())

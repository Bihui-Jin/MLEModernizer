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

0.767

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import random
import csv

import numpy as np
import cv2
import matplotlib

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tf_keras.optimizers import Adam
from tf_keras.utils import to_categorical

from sklearn.model_selection import train_test_split

random.seed(10)
np.random.seed(10)

BASE_PATH = "/kaggle/input/plant-seedlings-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print(
    "Train exists:", os.path.isdir(TRAIN_DIR), "Test exists:", os.path.isdir(TEST_DIR)
)




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




## === cell 2
NUM_CLASSES = 12
WIDTH = 128
HEIGHT = 128
DEPTH = 3
inputShape = (WIDTH, HEIGHT, DEPTH)
EPOCHS = 15
INIT_LR = 1e-3
BS = 32


def readTrainData(trainDir):
    data = []
    labels = []
    dirs = [d for d in os.listdir(trainDir) if os.path.isdir(os.path.join(trainDir, d))]
    dirs.sort()
    for dir_name in dirs:
        absDirPath = os.path.join(trainDir, dir_name)
        images = os.listdir(absDirPath)
        for imageFileName in images:
            imageFullPath = os.path.join(absDirPath, imageFileName)
            img = load_img(imageFullPath)
            arr = img_to_array(img)
            arr = cv2.resize(arr, (HEIGHT, WIDTH))
            data.append(arr)
            labels.append(classes_to_int(dir_name))
    return data, labels


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
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




## === cell 3
print("Loading images...")
sys.stdout.flush()

X, Y = readTrainData(TRAIN_DIR)
X = np.array(X, dtype="float32") / 255.0
Y = np.array(Y, dtype="int64")
Y = to_categorical(Y, num_classes=NUM_CLASSES)

print("Partition data into 75:25...")
sys.stdout.flush()
(trainX, valX, trainY, valY) = train_test_split(
    X, Y, test_size=0.25, random_state=10, stratify=Y.argmax(axis=1)
)

print("trainX:", trainX.shape, "trainY:", trainY.shape)
print("valX:", valX.shape, "valY:", valY.shape)



## === cell 4
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
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1,
)

print("Saving model to disk")
sys.stdout.flush()
MODEL_PATH = "/kaggle/working/CNNmodel2.keras"
model.save(MODEL_PATH)



## === cell 5
print("Generating plots...")
sys.stdout.flush()

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

acc_key = (
    "accuracy" if "accuracy" in H.history else ("acc" if "acc" in H.history else None)
)
val_acc_key = (
    "val_accuracy"
    if "val_accuracy" in H.history
    else ("val_acc" if "val_acc" in H.history else None)
)
if acc_key is not None:
    plt.plot(np.arange(0, N), H.history[acc_key], label="train_acc")
if val_acc_key is not None:
    plt.plot(np.arange(0, N), H.history[val_acc_key], label="val_acc")

plt.title("Training Loss and Accuracy on crop classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/kaggle/working/plot.png")




## === cell 6
def readTestData(testDir):
    data = []
    filenames = []
    images = sorted(os.listdir(testDir))
    for imageFileName in images:
        imageFullPath = os.path.join(testDir, imageFileName)
        img = load_img(imageFullPath)
        arr = img_to_array(img)
        arr = cv2.resize(arr, (HEIGHT, WIDTH))
        data.append(arr)
        filenames.append(imageFileName)
    return data, filenames




## === cell 7
print("Reading test data...")
sys.stdout.flush()
testX, filenames = readTestData(TEST_DIR)
testX = np.array(testX, dtype="float32") / 255.0

print("Loading model...")
sys.stdout.flush()
mymodel = keras.models.load_model(MODEL_PATH)

print("Predicting...")
sys.stdout.flush()
yFit = mymodel.predict(testX, batch_size=10, verbose=1)

pred_idx = np.argmax(yFit, axis=1).astype(int)

import pandas as pd

sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = {fn: int_to_classes(i) for fn, i in zip(filenames, pred_idx)}
sub["species"] = sub["file"].map(pred_map)

if sub["species"].isna().any():
    print(
        "Warning: Some test filenames not matched to sample_submission; falling back to sorted filename order."
    )
    sub = pd.DataFrame(
        {"file": filenames, "species": [int_to_classes(i) for i in pred_idx]}
    )

SUB_PATH = "/kaggle/working/submission.csv"
sub.to_csv(SUB_PATH, index=False)

print("Wrote submission to:", SUB_PATH)
print(sub.head())
print("Submission shape:", sub.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2502527059.py in <cell line: 0>()
      1 print("Reading test data...")
      2 sys.stdout.flush()
----> 3 testX, filenames = readTestData(TEST_DIR)
      4 testX = np.array(testX, dtype="float32") / 255.0
      5 

/tmp/ipykernel_11/3514427945.py in readTestData(testDir)
      6     for imageFileName in images:
      7         imageFullPath = os.path.join(testDir, imageFileName)
----> 8         img = load_img(imageFullPath)
      9         arr = img_to_array(img)
     10         arr = cv2.resize(arr, (HEIGHT, WIDTH))

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py in load_img(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)
    420         if isinstance(path, pathlib.Path):
    421             path = str(path.resolve())
--> 422         with open(path, "rb") as f:
    423             img = pil_image.open(io.BytesIO(f.read()))
    424     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/plant-seedlings-classification/test/test'

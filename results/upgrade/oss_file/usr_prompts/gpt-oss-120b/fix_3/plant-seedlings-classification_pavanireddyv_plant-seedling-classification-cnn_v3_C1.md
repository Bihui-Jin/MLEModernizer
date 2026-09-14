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
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_ROOT = "/kaggle/input/plant-seedlings-classification"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def classes_to_int(label):
    label = label.strip()
    mapping = {
        "Black-grass": 0,
        "Charlock": 1,
        "Cleavers": 2,
        "Common Chickweed": 3,
        "Common wheat": 4,
        "Fat Hen": 5,
        "Loose Silky-bent": 6,
        "Maize": 7,
        "Scentless Mayweed": 8,
        "Shepherds Purse": 9,
        "Small-flowered Cranesbill": 10,
        "Sugar beet": 11,
    }
    if label in mapping:
        return mapping[label]
    print("Invalid Label", label)
    return 12




## === cell 2
def int_to_classes(i):
    reverse = {
        0: "Black-grass",
        1: "Charlock",
        2: "Cleavers",
        3: "Common Chickweed",
        4: "Common wheat",
        5: "Fat Hen",
        6: "Loose Silky-bent",
        7: "Maize",
        8: "Scentless Mayweed",
        9: "Shepherds Purse",
        10: "Small-flowered Cranesbill",
        11: "Sugar beet",
    }
    if i in reverse:
        return reverse[i]
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


def _is_image_file(fname):
    ext = os.path.splitext(fname)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def readTrainData(trainDir):
    data = []
    labels = []
    for class_name in os.listdir(trainDir):
        class_path = os.path.join(trainDir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if os.path.isdir(fpath) or not _is_image_file(fname):
                continue
            img = load_img(fpath)
            arr = img_to_array(img)
            arr = cv2.resize(arr, (HEIGHT, WIDTH))
            data.append(arr)
            labels.append(classes_to_int(class_name))
    return data, labels


def readTestData(testDir):
    data = []
    filenames = []
    for fname in os.listdir(testDir):
        fpath = os.path.join(testDir, fname)
        if os.path.isdir(fpath) or not _is_image_file(fname):
            continue
        img = load_img(fpath)
        arr = img_to_array(img)
        arr = cv2.resize(arr, (HEIGHT, WIDTH))
        data.append(arr)
        filenames.append(fname)
    return data, filenames




## === cell 4
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
    model.add(Dense(units=NUM_CLASSES))
    model.add(Activation("softmax"))
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model




## === cell 5
random.seed(10)
print("Loading images...")
sys.stdout.flush()
train_dir = os.path.join(DATA_ROOT, "train")
X, Y = readTrainData(train_dir)
X = np.array(X, dtype="float32") / 255.0
Y = np.array(Y)
Y = to_categorical(Y, num_classes=NUM_CLASSES)

print("Partitioning data into 75:25...")
sys.stdout.flush()
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.25, random_state=10)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3669185774.py in <cell line: 0>()
      2 print("Loading images...")
      3 sys.stdout.flush()
----> 4 train_dir = os.path.join(DATA_ROOT, "train")
      5 X, Y = readTrainData(train_dir)
      6 X = np.array(X, dtype="float32") / 255.0

NameError: name 'DATA_ROOT' is not defined

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
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1,
)

print("Saving model to disk")
sys.stdout.flush()
model.save("/tmp/CNNmodel2")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1138312402.py in <cell line: 0>()
      1 print("Generating images...")
      2 sys.stdout.flush()
----> 3 aug = ImageDataGenerator(
      4     rotation_range=30,
      5     width_shift_range=0.1,

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on crop classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/305322695.py in <cell line: 0>()
      5 plt.figure()
      6 N = EPOCHS
----> 7 plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
      8 plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
      9 plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

NameError: name 'H' is not defined

## === cell 8
print("Preparing test data...")
sys.stdout.flush()
test_dir = os.path.join(DATA_ROOT, "test")
testX, filenames = readTestData(test_dir)
testX = np.array(testX, dtype="float32") / 255.0

print("Loading model and predicting...")
sys.stdout.flush()
mymodel = load_model("/tmp/CNNmodel2")
yFit = mymodel.predict(testX, batch_size=10, verbose=1)

import csv

with open("submission.csv", "w", newline="") as csvfile:
    fieldnames = ["file", "species"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for idx, file in enumerate(filenames):
        probs = yFit[idx]
        maxIdx = int(np.argmax(probs))
        writer.writerow({"file": file, "species": int_to_classes(maxIdx)})
print("Submission file 'submission.csv' written")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/290935479.py in <cell line: 0>()
      1 print("Preparing test data...")
      2 sys.stdout.flush()
----> 3 test_dir = os.path.join(DATA_ROOT, "test")
      4 testX, filenames = readTestData(test_dir)
      5 testX = np.array(testX, dtype="float32") / 255.0

NameError: name 'DATA_ROOT' is not defined

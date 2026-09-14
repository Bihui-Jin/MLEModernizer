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

0.7544

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
import matplotlib.pyplot as plt

import tf_keras as keras
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tf_keras.optimizers import Adam
from tf_keras.utils import to_categorical

from sklearn.model_selection import train_test_split

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
EPOCHS = 25
INIT_LR = 1e-3
BS = 32


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
            arr = img_to_array(img)  # (H,W,3)
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

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=500))
    model.add(Activation("relu"))
    model.add(Dense(units=12))
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model




## === cell 5
TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"
TEST_DIR = "/kaggle/input/plant-seedlings-classification/test"

print("Loading images...")
sys.stdout.flush()

X, Y = readTrainData(TRAIN_DIR)
X = np.array(X, dtype="float32") / 255.0
Y = np.array(Y, dtype="int32")
Y = to_categorical(Y, num_classes=NUM_CLASSES)

print("Partition data into 75:25...")
sys.stdout.flush()
(trainX, valX, trainY, valY) = train_test_split(
    X, Y, test_size=0.25, random_state=10, stratify=Y.argmax(axis=1)
)



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

print("Saving model to disk")
sys.stdout.flush()
model.save("/tmp/CNNmodel2.keras")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3162153611.py in <cell line: 0>()
     13 print("Compiling model...")
     14 sys.stdout.flush()
---> 15 model = createModel()
     16 
     17 print("Training network...")

/tmp/ipykernel_11/446137790.py in createModel()
     20 
     21     # Fix: Adam uses learning_rate in modern Keras; keep same schedule semantics via decay term.
---> 22     opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
     23     model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
     24     return model

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, adaptive_epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, name, **kwargs)
    112         **kwargs
    113     ):
--> 114         super().__init__(
    115             name=name,
    116             weight_decay=weight_decay,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in __init__(self, name, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, **kwargs)
   1161         mesh = kwargs.pop("mesh", None)
   1162         self._mesh = mesh
-> 1163         super().__init__(
   1164             name,
   1165             weight_decay,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in __init__(self, name, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, **kwargs)
    108         self._sharded_variable_builders = self._no_dependency({})
    109         self._create_iteration_variable()
--> 110         self._process_kwargs(kwargs)
    111 
    112     def _create_iteration_variable(self):

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in _process_kwargs(self, kwargs)
    137         for k in kwargs:
    138             if k in legacy_kwargs:
--> 139                 raise ValueError(
    140                     f"{k} is deprecated in the new TF-Keras optimizer, please "
    141                     "check the docstring for valid arguments, or use the "

ValueError: decay is deprecated in the new TF-Keras optimizer, please check the docstring for valid arguments, or use the legacy optimizer, e.g., tf.keras.optimizers.legacy.Adam.

## === cell 7
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")

plt.style.use("ggplot")
plt.figure()

N = EPOCHS
plt.plot(np.arange(0, N), H.history.get("loss", []), label="train_loss")
plt.plot(np.arange(0, N), H.history.get("val_loss", []), label="val_loss")

train_acc = H.history.get("accuracy", H.history.get("acc", []))
val_acc = H.history.get("val_accuracy", H.history.get("val_acc", []))
plt.plot(np.arange(0, N), train_acc, label="train_acc")
plt.plot(np.arange(0, N), val_acc, label="val_acc")

plt.title("Training Loss and Accuracy on crop classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
plt.close()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2962806018.py in <cell line: 0>()
      8 
      9 N = EPOCHS
---> 10 plt.plot(np.arange(0, N), H.history.get("loss", []), label="train_loss")
     11 plt.plot(np.arange(0, N), H.history.get("val_loss", []), label="val_loss")
     12 

NameError: name 'H' is not defined

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

mymodel = load_model("/tmp/CNNmodel2.keras")
yFit = mymodel.predict(testX, batch_size=10, verbose=1)

with open("submission.csv", "w", newline="") as csvfile:
    fieldnames = ["file", "species"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index, file in enumerate(filenames):
        classesProbs = yFit[index]
        maxIdx = int(np.argmax(classesProbs))
        writer.writerow({"file": file, "species": int_to_classes(maxIdx)})

print("Writing complete: submission.csv")
print("Rows:", len(filenames))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/116273396.py in <cell line: 0>()
      3 testX = np.array(testX, dtype="float32") / 255.0
      4 
----> 5 mymodel = load_model("/tmp/CNNmodel2.keras")
      6 yFit = mymodel.predict(testX, batch_size=10, verbose=1)
      7 

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode, **kwargs)
    260 
    261     # Legacy case.
--> 262     return legacy_sm_saving_lib.load_model(
    263         filepath, custom_objects=custom_objects, compile=compile, **kwargs
    264     )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/legacy/save.py in load_model(filepath, custom_objects, compile, options)
    231                     if isinstance(filepath_str, str):
    232                         if not tf.io.gfile.exists(filepath_str):
--> 233                             raise IOError(
    234                                 f"No file or directory found at {filepath_str}"
    235                             )

OSError: No file or directory found at /tmp/CNNmodel2.keras

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.82997

# 6. Current score

0.72973

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.72973) has done: 'Diagnosis: The crash happens because `os.listdir(testDir)` returns both image files and a nested `test/` directory (`/kaggle/working/../input/test/test`), and `load_img()` cannot open a directory path. The loop in cell 14 does not filter out non-files before trying to read them.  

Patch summary: In cell 14, add a simple `os.path.isfile()` guard (and optionally filter to common image extensions) so only actual image files are passed to `load_img()`. This keeps the same preprocessing logic and outputs (`Testdata`, `filenames`) but avoids attempting to open directories.  

Updated cells:  

Compatibility notes for cell k+1: `Testdata` and `filenames` remain lists of loaded/resized images and their corresponding filenames, so cell 15 (`testX = np.array(Testdata, ...)`) continues to work unchanged.  

Assumptions: Test images are stored directly in `testDir` (not only in nested folders), and any non-file entries in `testDir` should be skipped rather than processed.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import sys
import subprocess

try:
    import google.protobuf as _protobuf
    from packaging import version as _version

    _pb_ver = getattr(_protobuf, "__version__", "0")
    if _version.parse(_pb_ver) >= _version.parse("5.0.0"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import tensorflow as tf

import os

print(os.listdir("../input"))


## === cell 1
import numpy as np

import os
import sys
import cv2
from keras.utils import to_categorical
import matplotlib
from keras import backend as k
k.clear_session()


## === cell 2
import random

random.seed(10)
allLabels = os.listdir("../input/train/")  # list of subdirectories and files
trainDir = "/kaggle/working/../input/train/"

from keras.preprocessing.image import img_to_array, load_img

WIDTH = 128
HEIGHT = 128
DEPTH = 3

data = []
labels = []

dirs = os.listdir(trainDir)

for dir in dirs:
    absDirPath = os.path.join(os.path.sep, trainDir, dir)
    if not os.path.isdir(absDirPath):
        continue

    images = os.listdir(absDirPath)
    for imageFileName in images:
        imageFullPath = os.path.join(trainDir, dir, imageFileName)

        if not os.path.isfile(imageFullPath):
            continue

        print(imageFullPath)
        img = load_img(imageFullPath)
        arr = img_to_array(img)  # Numpy array with shape (H,W,3)
        arr = cv2.resize(
            arr, (HEIGHT, WIDTH)
        )  # Numpy array with shape (HEIGHT, WIDTH, 3)
        print(arr.shape)
        data.append(arr)
        label = str(imageFullPath.split("/")[-2])
        print(label)
        labels.append(label)


## === cell 3
len(images)
print('Number of images :-',len(data))
print('Numbe of Labels',len(labels))


## === cell 4
data[0]


## === cell 5
%matplotlib inline
import os
import matplotlib
import matplotlib.pyplot as plt

for i in range(1,10):
    print(i)
    new_image = tf.keras.preprocessing.image.array_to_img(data[i])
    plt.imshow(new_image)
    plt.show()
    


## === cell 6
from sklearn.preprocessing import LabelEncoder

TrainX = np.array(data, dtype="float") / 255.0
Y_labels = np.array(labels)
labelEncoder = LabelEncoder()
labelEncoder.fit(Y_labels)
train_labels_encoded = labelEncoder.transform(Y_labels)
trainY = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=12)
        


## === cell 7
print(TrainX.shape)
print(trainY.shape)


## === cell 8
from sklearn.model_selection import train_test_split
print("Train Validation Split into 80:20...")
sys.stdout.flush()
(x_train, valX, y_train, valY) = train_test_split(TrainX,trainY,test_size=0.20, random_state=10)


## === cell 9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


## === cell 10
from keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam

sys.stdout.flush()
k.clear_session()

inputShape = (WIDTH, HEIGHT, DEPTH)
EPOCHS = 15
INIT_LR = 1e-3
BS = 32

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=500))
model.add(Activation("relu"))

model.add(Dense(units=12))
model.add(Activation("softmax"))

opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()


## === cell 11

sys.stdout.flush()

H = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(valX, valY),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS,
    verbose=1,
)


## === cell 12
from matplotlib import pyplot

sys.stdout.flush()

train_acc_key = "accuracy" if "accuracy" in H.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in H.history else "val_acc"

pyplot.style.use("ggplot")
pyplot.figure()
N = EPOCHS

pyplot.plot(np.arange(0, N), H.history[train_acc_key], label="train_acc")
pyplot.plot(np.arange(0, N), H.history[val_acc_key], label="val_acc")
pyplot.title("Training /Validation and Accuracy on  crop classification")
pyplot.xlabel("Epoch #")
pyplot.ylabel("Accuracy")
pyplot.legend(loc="lower left")


## === cell 13
from matplotlib import pyplot
sys.stdout.flush()

pyplot.style.use("ggplot")
pyplot.figure()
N = EPOCHS
pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")
pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

pyplot.title("Training /Validation Loss on  crop classification")
pyplot.xlabel("Epoch #")
pyplot.ylabel("Loss")
pyplot.legend(loc="lower left")


## === cell 14
testDir = "/kaggle/working/../input/test/"

from keras.preprocessing.image import img_to_array, load_img

WIDTH = 128
HEIGHT = 128
DEPTH = 3
Testdata = []
filenames = []
images = os.listdir(testDir)
for imageFileName in images:
    imageFullPath = os.path.join(testDir, imageFileName)
    if not os.path.isfile(imageFullPath):
        continue
    print(imageFullPath)
    img = load_img(imageFullPath)
    arr = img_to_array(img)  # Numpy array with shape (...,..,3)
    arr = cv2.resize(arr, (HEIGHT, WIDTH))
    Testdata.append(arr)
    filenames.append(imageFileName)


## === cell 15
testX = np.array(Testdata, dtype="float") / 255.0


## === cell 16
testX.shape


## === cell 17
len(filenames)


## === cell 18
print(filenames[0])


## === cell 19
predicted_probs = model.predict(testX, batch_size=10, verbose=1)

predicted_classes = np.argmax(predicted_probs, axis=1)
out_df = pd.DataFrame({'file':filenames , 
                       'species': labelEncoder.inverse_transform(predicted_classes)})
out_df.to_csv('submission.csv', index=False)

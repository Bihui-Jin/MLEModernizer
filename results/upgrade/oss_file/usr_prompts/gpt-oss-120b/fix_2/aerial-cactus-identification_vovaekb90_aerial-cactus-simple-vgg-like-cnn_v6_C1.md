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
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
seaborn==0.12.2
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

0.9927

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import gc
import glob
import os
import cv2
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import imageio as im
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

print("Input root contents:", os.listdir("../input/"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def loadImagesData(glob_path):
    images = []
    names = []
    for img_path in glob.glob(glob_path):
        names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        images.append(img)  # already 32x32
    return images, names


trainData = {}
namesData = {}
for label in os.listdir("../input/train/"):
    imgs, nms = loadImagesData(f"../input/train/{label}/*.jpg")
    print(f"Loaded {len(imgs)} images from ../input/train/{label}/*.jpg")
    trainData[label] = imgs
    namesData[label] = nms

print("train labels:", ", ".join(trainData.keys()))
print("Number of training images:", len(trainData["train"]))

plt.figure(figsize=(4, 2))
columns = 4
for i in range(0, 8):
    plt.subplot(8 // columns + 1, columns, i + 1)
    plt.imshow(trainData["train"][i])
    plt.axis("off")
plt.show()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/628196217.py in <cell line: 0>()
     22 
     23 # Visual sanity check
---> 24 plt.figure(figsize=(4, 2))
     25 columns = 4
     26 for i in range(0, 8):

NameError: name 'plt' is not defined

## === cell 2
train_meta = pd.read_csv("../input/train.csv")
print("train.csv shape:", train_meta.shape)
print("Label distribution:\n", train_meta.has_cactus.value_counts())
lookupY = dict(zip(train_meta.id, train_meta.has_cactus))
train_meta.head()




## === cell 3
maxCount = 4364  # target number per class
counts = {"0": 0, "1": 0}
trainList = []
for i, img in enumerate(trainData["train"]):
    label = int(lookupY[namesData["train"][i]])
    counts[str(label)] += 1
    if counts[str(label)] <= maxCount:
        trainList.append({"label": label, "data": img})

random.shuffle(trainList)
train_df = pd.DataFrame(trainList)
gc.collect()
print("Balanced training dataframe shape:", train_df.shape)
print("Balanced label counts:\n", train_df.label.value_counts())
train_df.head()




## === cell 4
data_stack = np.stack(train_df["data"].values)  # shape (N, 32, 32, 3)
dfloats = data_stack.astype(np.float32)
all_x = dfloats / 255.0
print("all_x shape:", all_x.shape, "dtype:", all_x.dtype)

all_y = np.array(train_df["label"]).astype(np.float32)
print("all_y sample:", all_y[:5])




## === cell 5
train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=7, stratify=all_y
)
print("train_x shape:", train_x.shape, "test_x shape:", test_x.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4220034339.py in <cell line: 0>()
      1 # Split into training and validation sets
----> 2 train_x, test_x, train_y, test_y = train_test_split(
      3     all_x, all_y, test_size=0.2, random_state=7, stratify=all_y
      4 )
      5 print("train_x shape:", train_x.shape, "test_x shape:", test_x.shape)

NameError: name 'train_test_split' is not defined

## === cell 6
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=60,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(train_x)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/212766018.py in <cell line: 0>()
      1 # Data augmentation
----> 2 datagen = ImageDataGenerator(
      3     featurewise_center=False,
      4     samplewise_center=False,
      5     featurewise_std_normalization=False,

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
input_shape = train_x.shape[1:]  # (32, 32, 3)
output_shape = 1
m = Sequential()


def cnnNet(model):
    model.add(Conv2D(30, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(7, activation="relu"))  # small dense layer as in original
    model.add(Flatten())
    model.add(Dense(units=output_shape, activation="sigmoid"))


cnnNet(m)
m.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
m.summary()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/386541920.py in <cell line: 0>()
      1 # Build CNN model (same architecture as original cell 10)
----> 2 input_shape = train_x.shape[1:]  # (32, 32, 3)
      3 output_shape = 1
      4 m = Sequential()
      5 

NameError: name 'train_x' is not defined

## === cell 8
batch_size = 32
epochs = 4  # small number to keep runtime reasonable
history = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=train_x.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(test_x, test_y),
    workers=4,
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3770830674.py in <cell line: 0>()
      2 batch_size = 32
      3 epochs = 4  # small number to keep runtime reasonable
----> 4 history = m.fit(
      5     datagen.flow(train_x, train_y, batch_size=batch_size),
      6     steps_per_epoch=train_x.shape[0] // batch_size,

NameError: name 'm' is not defined

## === cell 9
test_path_patterns = [
    "../input/test/*.jpg",
    "../input/aerial-cactus-identification/test/*.jpg",
]
test_images = []
test_names = []
for pattern in test_path_patterns:
    imgs, nms = loadImagesData(pattern)
    if imgs:
        test_images.extend(imgs)
        test_names.extend(nms)

print(f"Loaded {len(test_images)} test images.")
data_stack = np.stack(test_images)
dfloats = data_stack.astype(np.float32)
unknown_x = dfloats / 255.0

predicted = np.ravel(m.predict(unknown_x, batch_size=64, verbose=0))
submission_df = pd.DataFrame({"id": test_names, "has_cactus": predicted})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, rows:", len(submission_df))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1362632276.py in <cell line: 0>()
     18 unknown_x = dfloats / 255.0
     19 
---> 20 predicted = np.ravel(m.predict(unknown_x, batch_size=64, verbose=0))
     21 submission_df = pd.DataFrame({"id": test_names, "has_cactus": predicted})
     22 submission_path = "submission.csv"

NameError: name 'm' is not defined

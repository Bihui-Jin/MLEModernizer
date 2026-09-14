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

0.4981

# 6. Current score

0.99733

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99733) has done: 'The main blocker is that the very first cell fails during `import keras`, which prevents all later variables (like `TRAIN_DIR`) from ever being defined, causing the cascade of `NameError`s. To make the notebook run in this Kaggle environment, I switch the imports to use `tf_keras` (installed) while keeping the exact same model architectures, loss, optimizer, and data pipeline. I also make image loading deterministic (sorted glob) and convert BGR→RGB for correct visualization (training data stays numerically equivalent aside from channel order, which is a correctness fix rather than a modeling change). Finally, I ensure a valid `submission.csv` is written with the required columns and all ids filled.'

# 9. Code solution

## === cell 0
import gc
import glob
import os
import cv2
import random
import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Activation, Dense, Dropout, Flatten
from tf_keras.layers import Conv2D, MaxPooling2D
from tf_keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

BASE_INPUT = "../input/aerial-cactus-identification"
if not os.path.exists(BASE_INPUT):
    alt = "/kaggle/input/aerial-cactus-identification"
    if os.path.exists(alt):
        BASE_INPUT = alt

TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Exists BASE_INPUT:", os.path.exists(BASE_INPUT), "->", BASE_INPUT)
print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "exists:", os.path.exists(TEST_DIR))
print("TRAIN_CSV:", TRAIN_CSV, "exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, "exists:", os.path.exists(SAMPLE_SUB))

random.seed(7)
np.random.seed(7)
try:
    keras.utils.set_random_seed(7)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def loadImagesData(glob_path):
    images = []
    names = []
    for img_path in sorted(glob.glob(glob_path)):
        names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        images.append(img)  # already 32x32
    return (images, names)


(train_images, train_names) = loadImagesData(os.path.join(TRAIN_DIR, "*.jpg"))
print("Loaded train images:", len(train_images))

plt.figure(figsize=(6, 3))
columns = 4
for i in range(min(8, len(train_images))):
    plt.subplot(8 // columns + 1, columns, i + 1)
    plt.imshow(cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()



## === cell 2
train_meta = pd.read_csv(TRAIN_CSV)
print(train_meta.shape)
print(train_meta.has_cactus.value_counts())

lookupY = dict(zip(train_meta["id"].values, train_meta["has_cactus"].values))
train_meta.head()



## === cell 3
trainList = []
maxCount = 4364  # number of has_cactus = 0 in original notebook comment
counts = {"0": 0, "1": 0}

for i, img in enumerate(train_images):
    img_id = train_names[i]
    label = int(lookupY[img_id])
    counts[str(label)] = 1 + counts[str(label)]
    if counts[str(label)] < maxCount:
        trainList.append({"label": label, "data": img})

random.shuffle(trainList)
train_df = pd.DataFrame(trainList)
gc.collect()
print(train_df.shape)
print(train_df.label.value_counts())
train_df.head()



## === cell 4
data_stack = np.stack(train_df["data"].values)
dfloats = data_stack.astype(np.float32)
all_x = np.multiply(dfloats, 1.0 / 255.0)
print(all_x.shape)
print(type(all_x))
all_x[0, 0, 0, 0]



## === cell 5
all_y = np.array(train_df.label).astype(np.float32)
all_y[0:5]



## === cell 6
train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=7
)
print(train_x.shape, test_x.shape)



## === cell 7
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



## === cell 8
num_filters = 8
input_shape = train_x.shape[1:]
output_shape = 1

m = Sequential()


def tdsNet(m):
    m.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    m.add(Conv2D(16, kernel_size=3, activation="relu"))
    m.add(Flatten())
    m.add(Dropout(0.5))
    m.add(Dense(units=output_shape, activation="sigmoid"))


tdsNet(m)
m.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
m.summary()



## === cell 9
batch_size = 32
history = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=(train_x.shape[0] // batch_size),
    epochs=4,
    validation_data=(test_x, test_y),
    workers=4,
    use_multiprocessing=False,
)



## === cell 10
num_filters = 8
input_shape = train_x.shape[1:]
output_shape = 1
m = Sequential()


def cnnNet(m):
    m.add(Conv2D(20, 5, padding="same", input_shape=input_shape))  # size: 5
    m.add(Activation("relu"))
    m.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    m.add(Conv2D(50, 5, padding="same"))  # size: 5
    m.add(Activation("relu"))
    m.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    m.add(Flatten())
    m.add(Dense(500))
    m.add(Activation("relu"))

    m.add(Dense(units=output_shape))
    m.add(Activation("sigmoid"))


cnnNet(m)
m.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
m.summary()



## === cell 11
batch_size = 32
history = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=(train_x.shape[0] // batch_size),
    epochs=10,
    validation_data=(test_x, test_y),
    workers=4,
    use_multiprocessing=False,
)



## === cell 12
trainList = []
for i, img in enumerate(train_images):
    img_id = train_names[i]
    label = int(lookupY[img_id])
    trainList.append({"label": label, "data": img})

random.shuffle(trainList)
train_df = pd.DataFrame(trainList)
gc.collect()

data_stack = np.stack(train_df["data"].values)
dfloats = data_stack.astype(np.float32)
all_x = np.multiply(dfloats, 1.0 / 255.0)
all_y = np.array(train_df.label).astype(np.float32)

train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=7
)
print(train_x.shape, test_x.shape)

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



## === cell 13
batch_size = 64
history = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=(train_x.shape[0] // batch_size),
    epochs=10,
    validation_data=(test_x, test_y),
    workers=4,
    use_multiprocessing=False,
)



## === cell 14
pd.read_csv(SAMPLE_SUB).head()



## === cell 15
(test_images, test_names) = loadImagesData(os.path.join(TEST_DIR, "*.jpg"))
data_stack = np.stack(test_images)
dfloats = data_stack.astype(np.float32)
unknown_x = np.multiply(dfloats, 1.0 / 255.0)

predicted = np.ravel(m.predict(unknown_x, batch_size=256, verbose=1))

sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_names, predicted))
sub["has_cactus"] = sub["id"].map(pred_map).astype(np.float32)

sub["has_cactus"] = sub["has_cactus"].fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()

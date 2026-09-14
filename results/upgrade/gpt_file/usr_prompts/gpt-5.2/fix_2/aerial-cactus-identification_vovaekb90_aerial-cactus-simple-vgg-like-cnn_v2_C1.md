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

0.9858

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
import random

import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


DATA_ROOT = "../input/aerial-cactus-identification"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "../input"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "count:",
    len(glob.glob(os.path.join(TRAIN_DIR, "*.jpg"))),
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "count:",
    len(glob.glob(os.path.join(TEST_DIR, "*.jpg"))),
)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))

random.seed(7)
np.random.seed(7)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_images_by_ids(img_dir, ids, color_flag=cv2.IMREAD_COLOR):
    """Load images in the order of `ids` from `img_dir`."""
    images = []
    names = []
    for img_id in ids:
        img_path = os.path.join(img_dir, img_id)
        img = cv2.imread(img_path, color_flag)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        images.append(img)  # already 32x32x3
        names.append(img_id)
    return images, names


def load_images_glob(glob_path):
    """Load images from a glob, returning images and basenames."""
    images = []
    names = []
    for img_path in sorted(glob.glob(glob_path)):
        names.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        images.append(img)
    return images, names




## === cell 2
train_meta = pd.read_csv(TRAIN_CSV)
print(train_meta.shape)
print(train_meta["has_cactus"].value_counts())

lookupY = dict(zip(train_meta["id"].values, train_meta["has_cactus"].values))

train_ids = train_meta["id"].tolist()
train_images, train_names = load_images_by_ids(TRAIN_DIR, train_ids)

print("Loaded train images:", len(train_images), "unique names:", len(set(train_names)))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1854107218.py in <cell line: 0>()
----> 1 train_meta = pd.read_csv(TRAIN_CSV)
      2 print(train_meta.shape)
      3 print(train_meta["has_cactus"].value_counts())
      4 
      5 # Create lookup from id -> label

NameError: name 'TRAIN_CSV' is not defined

## === cell 3
trainList = []
maxCount = (
    4364  # number of has_cactus = 0 (approx), used for downsampling the majority class
)
counts = {"0": 0, "1": 0}

for i, img in enumerate(train_images):
    img_id = train_names[i]
    label = int(lookupY[img_id])
    counts[str(label)] += 1
    if counts[str(label)] < maxCount:
        trainList.append({"label": label, "data": img})

random.shuffle(trainList)
train_df = pd.DataFrame(trainList)

gc.collect()
print(train_df.shape)
print(train_df.label.value_counts())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3284507845.py in <cell line: 0>()
      6 counts = {"0": 0, "1": 0}
      7 
----> 8 for i, img in enumerate(train_images):
      9     img_id = train_names[i]
     10     label = int(lookupY[img_id])

NameError: name 'train_images' is not defined

## === cell 4
data_stack = np.stack(train_df["data"].values)
dfloats = data_stack.astype(np.float32)
all_x = dfloats * (1.0 / 255.0)

all_y = np.array(train_df["label"].values).astype(np.float32)

print(all_x.shape, all_y.shape, all_y[:5])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2972214558.py in <cell line: 0>()
      1 # Encode training data (fix deprecated np.float usage)
----> 2 data_stack = np.stack(train_df["data"].values)
      3 dfloats = data_stack.astype(np.float32)
      4 all_x = dfloats * (1.0 / 255.0)
      5 

NameError: name 'train_df' is not defined

## === cell 5
train_x, test_x, train_y, test_y = train_test_split(
    all_x, all_y, test_size=0.2, random_state=7, stratify=all_y
)
print(train_x.shape, test_x.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3243952103.py in <cell line: 0>()
      1 # Split train/validation
----> 2 train_x, test_x, train_y, test_y = train_test_split(
      3     all_x, all_y, test_size=0.2, random_state=7, stratify=all_y
      4 )
      5 print(train_x.shape, test_x.shape)

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
/tmp/ipykernel_11/2338094075.py in <cell line: 0>()
      1 # Data augmentation (same parameters; fit() kept for API compatibility)
----> 2 datagen = ImageDataGenerator(
      3     featurewise_center=False,
      4     samplewise_center=False,
      5     featurewise_std_normalization=False,

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
input_shape = train_x.shape[1:]
output_shape = 1

m = Sequential()


def cnnNet(m_):
    m_.add(Conv2D(30, kernel_size=3, activation="relu", input_shape=input_shape))
    m_.add(MaxPooling2D(2, 2))
    m_.add(Conv2D(15, kernel_size=3, activation="relu"))
    m_.add(MaxPooling2D(2, 2))
    m_.add(Dense(7, activation="relu"))  # kept as in original code
    m_.add(Flatten())
    m_.add(Dense(units=output_shape, activation="sigmoid"))


cnnNet(m)
m.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])
m.summary()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2513566021.py in <cell line: 0>()
      1 # Create the network (preserve original cnnNet architecture used later)
----> 2 input_shape = train_x.shape[1:]
      3 output_shape = 1
      4 
      5 m = Sequential()

NameError: name 'train_x' is not defined

## === cell 8
batch_size = 32
history = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=(train_x.shape[0] // batch_size),
    epochs=6,
    validation_data=(test_x, test_y),
    workers=4,
    verbose=2,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4091152910.py in <cell line: 0>()
      1 # Train model (Keras 3: fit_generator removed; use fit with generator)
      2 batch_size = 32
----> 3 history = m.fit(
      4     datagen.flow(train_x, train_y, batch_size=batch_size),
      5     steps_per_epoch=(train_x.shape[0] // batch_size),

NameError: name 'm' is not defined

## === cell 9
batch_size = 64
history2 = m.fit(
    datagen.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch=(train_x.shape[0] // batch_size),
    epochs=6,
    validation_data=(test_x, test_y),
    workers=4,
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2292542011.py in <cell line: 0>()
      1 # (Optional continuation training as in the original notebook flow)
      2 batch_size = 64
----> 3 history2 = m.fit(
      4     datagen.flow(train_x, train_y, batch_size=batch_size),
      5     steps_per_epoch=(train_x.shape[0] // batch_size),

NameError: name 'm' is not defined

## === cell 10
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["id"].tolist()

test_images, test_names = load_images_by_ids(TEST_DIR, test_ids)
test_stack = np.stack(test_images).astype(np.float32)
unknown_x = test_stack * (1.0 / 255.0)

predicted = np.ravel(m.predict(unknown_x, batch_size=256, verbose=0)).astype(float)

submission_df = pd.DataFrame({"id": test_names, "has_cactus": predicted})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2422631484.py in <cell line: 0>()
      1 # Prepare test set and create submission in the exact sample_submission order
----> 2 sample_sub = pd.read_csv(SAMPLE_SUB)
      3 test_ids = sample_sub["id"].tolist()
      4 
      5 test_images, test_names = load_images_by_ids(TEST_DIR, test_ids)

NameError: name 'SAMPLE_SUB' is not defined

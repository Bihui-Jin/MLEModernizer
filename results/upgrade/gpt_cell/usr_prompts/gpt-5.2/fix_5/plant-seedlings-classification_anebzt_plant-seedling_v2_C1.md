# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

seed = 42
np.random.seed(seed)

print(os.listdir("../input"))



## === cell 1
CLASS = {
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

dim = 64



## === cell 2
sample_sub = pd.read_csv("../input/sample_submission.csv")



## === cell 3
sample_sub.head(10)



## === cell 4
import imageio.v2 as imageio
from skimage.transform import resize as imresize
from tqdm import tqdm

BASE = "../input/plant-seedlings-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")


def _list_image_paths_train(train_dir):
    classes = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    paths = []
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(".png"):
                paths.append(os.path.join(cls_dir, f))
    return paths


def _list_image_paths_test(test_dir):
    return [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ]


def img_reshape(img):
    img = imresize(img, (dim, dim, 3), preserve_range=True, anti_aliasing=True).astype(
        np.float32, copy=False
    )
    return img


def img_label(path):
    return str(path.split("/")[-1])


def img_class(path):
    return str(path.split("/")[-2])


train_path = _list_image_paths_train(TRAIN_DIR)
test_path = _list_image_paths_test(TEST_DIR)

train_path.sort()
test_path.sort()

xtrain_arr = np.empty((len(train_path), dim, dim, 3), dtype=np.float32)
train_labels = []
train_classes = []

for i, p in enumerate(tqdm(train_path, ascii=True, ncols=85, desc="Loading train")):
    img = imageio.imread(p)
    xtrain_arr[i] = img_reshape(img)
    train_labels.append(img_label(p))
    train_classes.append(img_class(p))

xtest_arr = np.empty((len(test_path), dim, dim, 3), dtype=np.float32)
test_labels = []

for i, p in enumerate(tqdm(test_path, ascii=True, ncols=85, desc="Loading test")):
    img = imageio.imread(p)
    xtest_arr[i] = img_reshape(img)
    test_labels.append(img_label(p))

train_dict = {"image": xtrain_arr, "label": train_labels, "class": train_classes}
test_dict = {"image": xtest_arr, "label": test_labels}

file_ext = ["png"]



## === cell 5
train_dict["image"][:5]



## === cell 6
file_ext



## === cell 7
train_path[:10]




## === cell 8
def to_categorical_np(y, num_classes=None, dtype="float32"):
    y = np.asarray(y, dtype="int64").ravel()
    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if y.size else 0
    out = np.zeros((y.shape[0], num_classes), dtype=dtype)
    if y.size:
        out[np.arange(y.shape[0]), y] = 1
    return out


xtrain = train_dict["image"]
_ytrain = np.array([CLASS[l] for l in train_dict["class"]], dtype=np.int64)
ytrain = to_categorical_np(_ytrain, num_classes=len(CLASS))



## === cell 9
import seaborn as sns

sns.set(style="white", context="notebook", palette="deep")

sns.countplot(_ytrain)

print(_ytrain.shape)
print(type(_ytrain))
__ytrain = pd.Series(_ytrain)

vals_class = __ytrain.value_counts()
print(vals_class)

cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class, ddof=1)

print("The mean amount of elements per class is", cls_mean)
print("The standard deviation in the element per class distribution is", cls_std)

if cls_std > cls_mean * (0.6827 / 2):
    print("The standard deviation is high")



## === cell 10
xtest = test_dict["image"]
label = test_dict["label"]



## === cell 11
xtrain[:5]



## === cell 12
xtrain.shape  # 4750, 64, 64, 3



## === cell 13
xtrain[:5]



## === cell 14
ytrain.shape  # (4750, 12)
nclasses = 12



## === cell 15
from sklearn.model_selection import train_test_split

split_pct = 0.05

xtrain, xval, ytrain, yval = train_test_split(
    xtrain,
    ytrain,
    test_size=split_pct,
    random_state=seed,
    stratify=np.argmax(ytrain, axis=1),
)

print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)



## === cell 16
from tf_keras import backend as K

from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import ReduceLROnPlateau
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from tf_keras.layers import Conv2D, MaxPool2D, AvgPool2D


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
model = Sequential()

ksize = 5

model.add(
    Conv2D(
        filters=32,
        kernel_size=(ksize, ksize),
        padding="same",
        activation="relu",
        input_shape=(dim, dim, 3),
    )
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(
    Conv2D(filters=64, kernel_size=(ksize, ksize), padding="same", activation="relu")
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(
    Conv2D(filters=64, kernel_size=(ksize, ksize), padding="same", activation="relu")
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(nclasses, activation="softmax"))

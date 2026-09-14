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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

3.2273404955968377

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import json
import csv
import numpy as np
import pandas as pd

import cv2
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    for rootDir, dirNames, filenames in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind(".") :].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized




## === cell 2
train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/"
final_test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown/"

train_img_paths = sorted(list(list_images(train_path)))
final_test_img_paths = sorted(list(list_images(final_test_path)))

len(train_img_paths), len(final_test_img_paths), train_img_paths[
    :2
], final_test_img_paths[:2]



## === cell 3
trainLabels = []
rej = 0
for p in train_img_paths:
    fname = os.path.basename(p).lower()
    if not fname.endswith(".jpg"):
        rej += 1
        continue
    parent = os.path.basename(os.path.dirname(p)).lower()
    if parent in ("cat", "dog"):
        trainLabels.append(parent)
    else:
        rej += 1

trainLabels = np.array(trainLabels)
print(
    "num images:",
    len(trainLabels),
    "rejected:",
    rej,
    "classes:",
    np.unique(trainLabels, return_counts=True),
)



## === cell 4
le = LabelEncoder()
trainLabels_enc = le.fit_transform(trainLabels)
class_to_index = {c: int(i) for i, c in enumerate(le.classes_)}
dog_index = class_to_index.get("dog", 1)
print("LabelEncoder classes_:", le.classes_, "dog_index:", dog_index)



## === cell 5
plt.figure(figsize=(4, 3))
sns.countplot(x=trainLabels)
plt.title("Class counts")
plt.tight_layout()
plt.show()



## === cell 6
NUM_CLASSES = 2

NUM_VAL_IMAGES = 1250 * NUM_CLASSES  # 2500
NUM_TEST_IMAGES = 1250 * NUM_CLASSES  # 2500

train_hdf5 = "/kaggle/working/train.hdf5"
val_hdf5 = "/kaggle/working/val.hdf5"
test_hdf5 = "/kaggle/working/test.hdf5"
MODEL_PATH = "/kaggle/working/alexnet_dogs_vs_cats.model"
dataset_mean = "/kaggle/working/dogs_vs_cats_mean.json"
output_path = "/kaggle/working/"




## === cell 7
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




## === cell 8
class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dH, dW = 0, 0

        if w < h:
            image = resize(image, width=self.width, inter=self.inter)
            dH = (image.shape[0] - self.height) // 2
        else:
            image = resize(image, height=self.height, inter=self.inter)
            dW = (image.shape[1] - self.width) // 2

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




## === cell 9
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError(
                'The supplied "outputPath" already exists. Manually delete the file before continuing.',
                outputPath,
            )

        self.db = h5py.File(outputPath, mode="w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float32")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx : i] = self.buffer["data"]
        self.labels[self.idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_name", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()




## === cell 10
n_total = len(train_img_paths)
test_size = min(NUM_TEST_IMAGES, n_total // 5)  # keep <= 20% if dataset smaller
val_size = min(NUM_VAL_IMAGES, n_total // 5)

train_paths_tmp, test_img_paths, y_train_tmp, y_test = train_test_split(
    train_img_paths,
    trainLabels_enc,
    test_size=test_size,
    random_state=42,
    stratify=trainLabels_enc,
)
train_img_paths, val_img_paths, y_train, y_val = train_test_split(
    train_paths_tmp,
    y_train_tmp,
    test_size=val_size,
    random_state=42,
    stratify=y_train_tmp,
)

len(train_img_paths), len(val_img_paths), len(test_img_paths)



## === cell 11
aap = AspectAwarePreprocessor(227, 227)
R, G, B = [], [], []

sample_for_mean = min(5000, len(train_img_paths))
for path in tqdm(
    train_img_paths[:sample_for_mean], desc="Computing dataset mean (subset)"
):
    image = cv2.imread(path)
    if image is None:
        continue
    image = aap.preprocess(image)
    (b, g, r) = cv2.mean(image)[:3]
    R.append(r)
    G.append(g)
    B.append(b)

B_mean = float(np.mean(B)) if len(B) else 0.0
G_mean = float(np.mean(G)) if len(G) else 0.0
R_mean = float(np.mean(R)) if len(R) else 0.0
B_mean, G_mean, R_mean



## === cell 12
with open(dataset_mean, "w") as f:
    json.dump({"R": R_mean, "G": G_mean, "B": B_mean}, f)


class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        return cv2.merge([B, G, R])




## === cell 13
class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        if h <= self.height or w <= self.width:
            image = aap.preprocess(image)
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]




## === cell 14
class CropPreprocessor:
    def __init__(self, height, width, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        crops = []
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h],
        ]
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        for startX, startY, endX, endY in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)




## === cell 15
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array


class imageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(R_mean, G_mean, B_mean)
iap = imageToArrayPreprocessor()




## === cell 16
def image_data_generator(
    directory_list,
    labels,
    bs=128,
    binarize=True,
    preprocessors=None,
    aug=None,
    classes=2,
):
    while True:
        for i in range(0, labels.shape[0], bs):
            imagePaths = directory_list[i : i + bs]
            label_vals = labels[i : i + bs]

            if binarize:
                label_vals = to_categorical(label_vals, classes)

            if preprocessors is not None:
                procImages = []
                for path in imagePaths:
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    for p in preprocessors:
                        image = p.preprocess(image)
                    procImages.append(image)
                images = np.array(procImages)
            else:
                images = np.array([])

            if aug is not None and len(images):
                (images, label_vals) = next(
                    aug.flow(images, label_vals, batch_size=bs, shuffle=False)
                )

            yield (images, label_vals)


def test_data_generator(directory_list, bs=128, preprocessors=None, passes=1):
    epochs = 0
    while epochs < passes:
        for i in range(0, len(directory_list), bs):
            imagePaths = directory_list[i : i + bs]
            procImages = []
            for path in imagePaths:
                image = cv2.imread(path)
                if image is None:
                    continue
                if preprocessors is not None:
                    for p in preprocessors:
                        image = p.preprocess(image)
                procImages.append(image)
            images = np.array(procImages)
            yield images
        epochs += 1




## === cell 17
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
from tensorflow.keras.regularizers import l2


class Alexnet:
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(
            Conv2D(
                96,
                (11, 11),
                strides=(4, 4),
                input_shape=inputShape,
                padding="same",
                kernel_regularizer=l2(reg),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                256, (5, 5), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(
            Conv2D(
                384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(
            Conv2D(
                256, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation="softmax", kernel_regularizer=l2(reg)))
        return model




## === cell 18
model = Alexnet.build(227, 227, 3, NUM_CLASSES, reg=0.0002)
opt = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

train_gen = image_data_generator(
    train_img_paths,
    y_train,
    bs=64,
    preprocessors=[pp, mp, iap],
    aug=aug,
    classes=NUM_CLASSES,
)
val_gen = image_data_generator(
    val_img_paths,
    y_val,
    bs=64,
    preprocessors=[sp, mp, iap],
    aug=None,
    classes=NUM_CLASSES,
)

steps_per_epoch = int(np.ceil(len(y_train) / 64))
val_steps = int(np.ceil(len(y_val) / 64))

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=1,
    verbose=1,
)

model.save(MODEL_PATH)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/180563512.py in <cell line: 0>()
     35 )
     36 
---> 37 model.save(MODEL_PATH)
     38 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in save_model(model, filepath, overwrite, zipped, **kwargs)
    112             model, filepath, overwrite, include_optimizer
    113         )
--> 114     raise ValueError(
    115         "Invalid filepath extension for saving. "
    116         "Please add either a `.keras` extension for the native Keras "

ValueError: Invalid filepath extension for saving. Please add either a `.keras` extension for the native Keras format (recommended) or a `.h5` extension. Use `model.export(filepath)` if you want to export a SavedModel for use with TFLite/TFServing/etc. Received: filepath=/kaggle/working/alexnet_dogs_vs_cats.model.

## === cell 19
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)


def predict_with_crops(paths, batch_size=32):
    out = []
    for imgs in tqdm(
        test_data_generator(paths, bs=batch_size, preprocessors=[mp], passes=1),
        total=int(np.ceil(len(paths) / batch_size)),
        desc="Predicting with 10-crop TTA",
    ):
        for image in imgs:
            if image is None:
                continue
            (h, w) = image.shape[:2]
            if h <= 227 or w <= 227:
                image = aap2.preprocess(image)
            crops = cp.preprocess(image)
            crops = np.array([iap.preprocess(c) for c in crops])
            pred = model.predict(crops, verbose=0)
            out.append(pred.mean(axis=0))
    return np.array(out)


final_predict = predict_with_crops(final_test_img_paths, batch_size=32)
final_predict.shape




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2691734812.py in <cell line: 0>()
     24 
     25 
---> 26 final_predict = predict_with_crops(final_test_img_paths, batch_size=32)
     27 final_predict.shape
     28 

/tmp/ipykernel_11/2691734812.py in predict_with_crops(paths, batch_size)
      6 def predict_with_crops(paths, batch_size=32):
      7     out = []
----> 8     for imgs in tqdm(
      9         test_data_generator(paths, bs=batch_size, preprocessors=[mp], passes=1),
     10         total=int(np.ceil(len(paths) / batch_size)),

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/tmp/ipykernel_11/3232359665.py in test_data_generator(directory_list, bs, preprocessors, passes)
     51                         image = p.preprocess(image)
     52                 procImages.append(image)
---> 53             images = np.array(procImages)
     54             yield images
     55         epochs += 1

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (32,) + inhomogeneous part.

## === cell 20
def extract_test_id(path):
    base = os.path.basename(path)
    m = re.match(r"^(\d+)\.jpg$", base)
    if m:
        return int(m.group(1))
    stem = os.path.splitext(base)[0]
    return int(stem)


final_ids = np.array([extract_test_id(p) for p in final_test_img_paths], dtype=np.int64)

dog_probs = final_predict[:, dog_index].astype(np.float64)

eps = 1e-7
dog_probs = np.clip(dog_probs, eps, 1.0 - eps)

submission = pd.DataFrame({"id": final_ids, "label": dog_probs})
submission = submission.sort_values("id").reset_index(drop=True)

submission.head(), submission.shape



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2776160169.py in <cell line: 0>()
     13 
     14 # Probability of dog = column at dog_index
---> 15 dog_probs = final_predict[:, dog_index].astype(np.float64)
     16 
     17 # Safety for logloss: keep strictly inside (0,1)

NameError: name 'final_predict' is not defined

## === cell 21
sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.describe(include="all"))
print("Min/Max label:", submission["label"].min(), submission["label"].max())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1187586610.py in <cell line: 0>()
      1 # Write submission with required name and columns
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 submission.to_csv(sub_path, index=False)
      4 
      5 print("Wrote:", sub_path)

NameError: name 'submission' is not defined

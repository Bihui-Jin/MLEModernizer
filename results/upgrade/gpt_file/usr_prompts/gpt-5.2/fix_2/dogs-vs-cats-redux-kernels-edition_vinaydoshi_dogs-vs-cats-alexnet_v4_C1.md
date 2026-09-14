# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import json
import csv
import numpy as np
import pandas as pd

import cv2
import h5py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

import random

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"



## === cell 1
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
base_input = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

train_path = os.path.join(base_input, "train")
final_test_path = os.path.join(base_input, "test", "unknown")

train_img_paths = list(list_images(train_path))
final_test_img_paths = list(list_images(final_test_path))

print("Train images:", len(train_img_paths))
print("Test images:", len(final_test_img_paths))
print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(final_test_path))



## === cell 3
NUM_CLASSES = 2

NUM_VAL_IMAGES = 2500
NUM_TEST_IMAGES = 2500

train_hdf5 = "/kaggle/working/train.hdf5"
val_hdf5 = "/kaggle/working/val.hdf5"
test_hdf5 = "/kaggle/working/test.hdf5"
MODEL_PATH = "/kaggle/working/alexnet_dogs_vs_cats.model"
dataset_mean = "/kaggle/working/dogs_vs_cats_mean.json"
output_path = "/kaggle/working/"




## === cell 4
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


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




## === cell 5
trainLabels = []
valid_train_img_paths = []

for p in train_img_paths:
    ext = os.path.splitext(p)[1].lower()
    if ext != ".jpg":
        continue
    cls = os.path.basename(os.path.dirname(p)).lower()
    if cls not in ("cat", "dog"):
        continue
    trainLabels.append(cls)
    valid_train_img_paths.append(p)

train_img_paths = valid_train_img_paths
print("Filtered train images:", len(train_img_paths))
print("Unique string labels:", sorted(set(trainLabels))[:10], " ...")

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
print("Encoded labels:", {c: int(le.transform([c])[0]) for c in le.classes_})

if sns is not None and len(trainLabels) > 0:
    try:
        sns.countplot(x=trainLabels)
        if plt is not None:
            plt.show()
    except Exception as e:
        print("Skipping plot due to:", repr(e))



## === cell 6
n = len(train_img_paths)
num_test = min(NUM_TEST_IMAGES, max(1, n // 10))
num_val = min(NUM_VAL_IMAGES, max(1, n // 10))

max_holdout = n - 2
if num_test + num_val > max_holdout:
    num_test = max(1, (max_holdout // 2))
    num_val = max(1, max_holdout - num_test)

train_img_paths, test_img_paths, y_train, y_test = train_test_split(
    train_img_paths,
    trainLabels,
    test_size=num_test,
    random_state=42,
    stratify=trainLabels,
)

train_img_paths, val_img_paths, y_train, y_val = train_test_split(
    train_img_paths, y_train, test_size=num_val, random_state=42, stratify=y_train
)

print("Split sizes:", len(train_img_paths), len(val_img_paths), len(test_img_paths))



## === cell 7
aap = AspectAwarePreprocessor(227, 227)
R, G, B = [], [], []

for path in train_img_paths:
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
print("Channel means (B,G,R):", B_mean, G_mean, R_mean)

with open(dataset_mean, "w") as f:
    json.dump({"B": B_mean, "G": G_mean, "R": R_mean}, f)




## === cell 8
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


class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        if h <= self.height or w <= self.width:
            image = aap.preprocess(image)
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]


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




## === cell 9
import tensorflow as tf
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




## === cell 10
def image_data_generator(
    directory_list,
    labels,
    bs=128,
    mode="train",
    binarize=True,
    preprocessors=None,
    aug=None,
    classes=2,
):
    labels = np.asarray(labels)
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
                images = np.array(procImages, dtype="float32")
            else:
                images = np.array([], dtype="float32")

            if aug is not None:
                (images, label_vals) = next(
                    aug.flow(images, label_vals, batch_size=len(images), shuffle=False)
                )

            yield (images, label_vals)




## === cell 11
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
from tensorflow.keras import backend as K


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




## === cell 12
from tensorflow.keras.optimizers import Adam

model = Alexnet.build(227, 227, 3, NUM_CLASSES, reg=0.0002)
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

train_gen = image_data_generator(
    train_img_paths, y_train, bs=64, preprocessors=[pp, mp, iap], aug=aug
)
val_gen = image_data_generator(
    val_img_paths, y_val, bs=64, preprocessors=[sp, mp, iap], aug=None
)

steps_per_epoch = max(1, len(train_img_paths) // 64)
val_steps = max(1, len(val_img_paths) // 64)

EPOCHS = 1
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=EPOCHS,
    verbose=1,
)

model.save(os.path.join(output_path, "alexnet_trained_local.h5"))




## === cell 13
def test_data_generator(directory_list, bs=128, preprocessors=None, passes=1):
    directory_list = list(directory_list)
    for _ in range(passes):
        for i in range(0, len(directory_list), bs):
            imagePaths = directory_list[i : i + bs]
            if preprocessors is not None:
                procImages = []
                for path in imagePaths:
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    for p in preprocessors:
                        image = p.preprocess(image)
                    procImages.append(image)
                images = np.array(procImages, dtype="float32")
            else:
                images = np.array([], dtype="float32")
            yield images




## === cell 14
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)

final_predict = []
bs = 64

for batch_images in test_data_generator(
    final_test_img_paths, bs=bs, preprocessors=[mp], passes=1
):
    for image in batch_images:
        (h, w) = image.shape[:2]
        if h <= 227 or w <= 227:
            image = aap2.preprocess(image)
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")
        pred = model.predict(crops, verbose=0)
        final_predict.append(pred.mean(axis=0))

final_predict = np.asarray(final_predict, dtype="float32")
print("Pred shape:", final_predict.shape)



## === cell 15
class_to_index = {c: int(le.transform([c])[0]) for c in le.classes_}
dog_index = class_to_index.get("dog", 1)

dog_prob = final_predict[:, dog_index]

final_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in final_test_img_paths
]

submission = pd.DataFrame({"id": final_ids, "label": dog_prob})
submission = submission.sort_values("id").reset_index(drop=True)

eps = 1e-7
submission["label"] = submission["label"].clip(eps, 1 - eps)

print(submission.head())
print("Rows:", len(submission))

submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)

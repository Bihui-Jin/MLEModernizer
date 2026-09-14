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
import re
import json
import csv
import numpy as np
import pandas as pd

import cv2
import h5py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d

import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
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

np.random.seed(42)
tf.random.set_seed(42)



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
train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train"
final_test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown"

train_img_paths = sorted(list(list_images(train_path)))
final_test_img_paths = sorted(list(list_images(final_test_path)))

len(train_img_paths), len(final_test_img_paths), train_img_paths[
    :2
], final_test_img_paths[:2]



## === cell 3
trainLabels = []
bad = 0
for p in train_img_paths:
    fname = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    label = None
    if parent in ("cat", "dog"):
        label = parent
    else:
        if fname.startswith("cat."):
            label = "cat"
        elif fname.startswith("dog."):
            label = "dog"
    if label is None:
        bad += 1
        continue
    trainLabels.append(label)

trainLabels = np.array(trainLabels)
bad, trainLabels.shape, np.unique(trainLabels, return_counts=True)



## === cell 4
le = LabelEncoder()
trainLabels_enc = le.fit_transform(trainLabels)
le.classes_, np.unique(trainLabels_enc, return_counts=True)



## === cell 5
if trainLabels_enc.shape[0] > 0:
    ax = sns.countplot(x=trainLabels_enc)
    ax.figure.savefig("/kaggle/working/label_dist.png")
plt.close("all")



## === cell 6
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

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
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
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
if trainLabels_enc.shape[0] != len(train_img_paths):
    raise RuntimeError("Label/path length mismatch. Check label extraction logic.")

train_img_paths_split, test_img_paths, y_train, y_test = train_test_split(
    train_img_paths,
    trainLabels_enc,
    test_size=NUM_TEST_IMAGES,
    random_state=42,
    stratify=trainLabels_enc,
)

train_img_paths, val_img_paths, y_train, y_val = train_test_split(
    train_img_paths_split,
    y_train,
    test_size=NUM_VAL_IMAGES,
    random_state=42,
    stratify=y_train,
)

len(train_img_paths), len(val_img_paths), len(
    test_img_paths
), y_train.shape, y_val.shape, y_test.shape



## === cell 11
aap = AspectAwarePreprocessor(227, 227)

R, G, B = [], [], []

for path, label in tqdm.tqdm(
    list(zip(train_img_paths, y_train))[:5000],
    desc="Computing mean (subset=5000)",
    total=min(5000, len(train_img_paths)),
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

means = {"R": R_mean, "G": G_mean, "B": B_mean}
with open(dataset_mean, "w") as f:
    json.dump(means, f)

(B_mean, G_mean, R_mean)




## === cell 12
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
class imageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)




## === cell 16
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)



## === cell 17
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(R_mean, G_mean, B_mean)
iap = imageToArrayPreprocessor()




## === cell 18
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
                images = np.array([cv2.imread(p) for p in imagePaths])

            if aug is not None:
                (images, label_vals) = next(
                    aug.flow(images, label_vals, batch_size=bs, shuffle=False)
                )

            yield (images, label_vals)




## === cell 19
def test_data_generator(
    directory_list,
    bs=128,
    preprocessors=None,
    aug=None,
    passes=1,
):
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

            if aug is not None:
                images = next(aug.flow(images, batch_size=bs, shuffle=False))

            yield images
        epochs += 1




## === cell 20
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




## === cell 21
model = Alexnet.build(227, 227, 3, 2, reg=0.0002)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()



## === cell 22
train_gen = image_data_generator(
    train_img_paths, y_train, bs=64, preprocessors=[pp, mp, iap], aug=aug
)
val_gen = image_data_generator(
    val_img_paths, y_val, bs=64, preprocessors=[sp, mp, iap], aug=None
)

steps_per_epoch = int(np.ceil(len(train_img_paths) / 64.0))
val_steps = int(np.ceil(len(val_img_paths) / 64.0))

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=2,
    verbose=1,
)

model.save(MODEL_PATH)



## === cell 23
final_predict = []
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)

total_batches = int(np.ceil(len(final_test_img_paths) / 128.0))
for batch_idx, images in enumerate(
    test_data_generator(final_test_img_paths, bs=128, preprocessors=[mp], passes=1)
):
    for image in images:
        (h, w) = image.shape[:2]
        if h <= 227 or w <= 227:
            image = aap2.preprocess(image)
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops])
        pred = model.predict(crops, verbose=0)
        final_predict.append(pred.mean(axis=0))
    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        print(f"Processed batch {batch_idx+1}/{total_batches}")

len(final_predict)



## === cell 24
dog_idx = int(np.where(le.classes_ == "dog")[0][0]) if "dog" in le.classes_ else 1
val_outs = [float(p[dog_idx]) for p in final_predict]
len(val_outs), val_outs[:5]




## === cell 25
def extract_id_from_path(p):
    base = os.path.basename(p)
    m = re.search(r"(\d+)", base)
    if m is None:
        raise ValueError(f"Could not extract numeric id from filename: {base}")
    return int(m.group(1))


final_ids = [extract_id_from_path(p) for p in final_test_img_paths]

submission = pd.DataFrame({"id": final_ids, "label": val_outs})
submission = submission.sort_values("id").reset_index(drop=True)

sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
sample = pd.read_csv(sample_path)
submission = sample[["id"]].merge(submission, on="id", how="left")

submission["label"] = submission["label"].fillna(0.5).astype(float)

submission.head(), submission.shape



## === cell 26
out_file = "/kaggle/working/submission.csv"
submission.to_csv(out_file, index=False)

print("Wrote:", out_file)
print(submission.describe(include="all"))

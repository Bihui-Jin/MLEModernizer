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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import json
import math
import numpy as np
import pandas as pd

import cv2
import h5py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
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

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

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




## === cell 1
train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train"
final_test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown"

train_img_paths_all = sorted(list(list_images(train_path)))
final_test_img_paths = sorted(list(list_images(final_test_path)))

len(train_img_paths_all), len(final_test_img_paths), train_img_paths_all[
    :2
], final_test_img_paths[:2]



## === cell 2
train_img_paths = []
trainLabels = []
bad = 0

for p in train_img_paths_all:
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
    train_img_paths.append(p)
    trainLabels.append(label)

train_img_paths = np.array(train_img_paths)
trainLabels = np.array(trainLabels)

bad, trainLabels.shape, len(train_img_paths), np.unique(trainLabels, return_counts=True)



## === cell 3
le = LabelEncoder()
trainLabels_enc = le.fit_transform(trainLabels)
le.classes_, np.unique(trainLabels_enc, return_counts=True)



## === cell 4
if trainLabels_enc.shape[0] > 0:
    fig, ax = plt.subplots(figsize=(4, 3))
    vals, counts = np.unique(trainLabels_enc, return_counts=True)
    ax.bar(vals.astype(str), counts)
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    fig.savefig("/kaggle/working/label_dist.png", bbox_inches="tight")
plt.close("all")



## === cell 5
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

train_hdf5 = "/kaggle/working/train.hdf5"
val_hdf5 = "/kaggle/working/val.hdf5"
test_hdf5 = "/kaggle/working/test.hdf5"
MODEL_PATH = "/kaggle/working/alexnet_dogs_vs_cats.model"
dataset_mean = "/kaggle/working/dogs_vs_cats_mean.json"
output_path = "/kaggle/working/"




## === cell 6
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




## === cell 7
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




## === cell 8
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




## === cell 9
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



## === cell 10
aap = AspectAwarePreprocessor(227, 227)

if os.path.exists(dataset_mean):
    with open(dataset_mean, "r") as f:
        means = json.load(f)
    R_mean = float(means.get("R", 0.0))
    G_mean = float(means.get("G", 0.0))
    B_mean = float(means.get("B", 0.0))
else:
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




## === cell 11
class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = float(rMean)
        self.gMean = float(gMean)
        self.bMean = float(bMean)

    def preprocess(self, image):
        image = image.astype("float32", copy=False)
        image[..., 2] -= self.rMean  # R
        image[..., 1] -= self.gMean  # G
        image[..., 0] -= self.bMean  # B
        return image




## === cell 12
class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        if h <= self.height or w <= self.width:
            image = aap.preprocess(image)
            (h, w) = image.shape[:2]

        max_y = h - self.height
        max_x = w - self.width
        y = 0 if max_y <= 0 else np.random.randint(0, max_y + 1)
        x = 0 if max_x <= 0 else np.random.randint(0, max_x + 1)
        patch = image[y : y + self.height, x : x + self.width]
        return patch




## === cell 13
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




## === cell 14
class imageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)




## === cell 15
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)



## === cell 16
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(R_mean, G_mean, B_mean)
iap = imageToArrayPreprocessor()




## === cell 17
class ImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        paths,
        labels=None,
        bs=64,
        binarize=True,
        preprocessors=None,
        aug=None,
        classes=2,
        shuffle=False,
    ):
        self.paths = np.asarray(paths)
        self.labels = None if labels is None else np.asarray(labels)
        self.bs = int(bs)
        self.binarize = binarize
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.aug = aug
        self.classes = int(classes)
        self.shuffle = bool(shuffle)

        self.indexes = np.arange(len(self.paths))
        self.on_epoch_end()
        self._pp_chain = tuple(self.preprocessors)

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.bs)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min((idx + 1) * self.bs, len(self.paths))
        batch_ids = self.indexes[start:end]
        batch_paths = self.paths[batch_ids]

        proc = []
        kept_ids = []
        for _i, p in enumerate(batch_paths):
            im = cv2.imread(p)
            if im is None:
                continue
            for pr in self._pp_chain:
                im = pr.preprocess(im)
            proc.append(im)
            kept_ids.append(batch_ids[_i])

        if len(proc) == 0:
            x = np.zeros((0, 227, 227, 3), dtype="float32")
            if self.labels is None:
                return x
            y = np.zeros((0, self.classes), dtype="float32")
            return x, y

        x = np.stack(proc, axis=0)

        if self.aug is not None:
            it = self.aug.flow(x, batch_size=x.shape[0], shuffle=False)
            x = next(it)

        if self.labels is None:
            return x

        y_vals = self.labels[np.asarray(kept_ids)]
        if self.binarize:
            y_vals = to_categorical(y_vals, self.classes)

        return x, y_vals


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
    return ImageSequence(
        directory_list,
        labels=labels,
        bs=bs,
        binarize=binarize,
        preprocessors=preprocessors,
        aug=aug,
        classes=classes,
        shuffle=False,
    )




## === cell 18
def test_data_generator(
    directory_list,
    bs=128,
    preprocessors=None,
    aug=None,
    passes=1,
):
    seq = ImageSequence(
        directory_list,
        labels=None,
        bs=bs,
        binarize=False,
        preprocessors=preprocessors,
        aug=aug,
        classes=2,
        shuffle=False,
    )
    for _ in range(int(passes)):
        for i in range(len(seq)):
            yield seq[i]




## === cell 19
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




## === cell 20
model = Alexnet.build(227, 227, 3, 2, reg=0.0002)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()



## === cell 21
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



## === cell 22
from concurrent.futures import ThreadPoolExecutor

final_predict = []
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)

bs = 128
total_batches = int(np.ceil(len(final_test_img_paths) / float(bs)))

test_preprocessors = [aap2]

rMean, gMean, bMean = mp.rMean, mp.gMean, mp.bMean


def _make_10crops(img_256):
    return cp.preprocess(img_256).astype(np.float32, copy=False)


max_workers = min(8, os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for batch_idx, images in enumerate(
        test_data_generator(
            final_test_img_paths, bs=bs, preprocessors=test_preprocessors, passes=1
        )
    ):
        if images.shape[0] == 0:
            continue

        B = images.shape[0]

        crops_list = list(ex.map(_make_10crops, (images[j] for j in range(B))))
        crop_batch = np.stack(crops_list, axis=0)  # (B,10,227,227,3)

        crop_batch[..., 2] -= rMean
        crop_batch[..., 1] -= gMean
        crop_batch[..., 0] -= bMean

        batch_crops = crop_batch.reshape((-1, 227, 227, 3))
        preds = model.predict(batch_crops, verbose=0, batch_size=256)
        preds = preds.reshape(B, 10, preds.shape[-1]).mean(axis=1)
        final_predict.extend(preds.tolist())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"Processed batch {batch_idx+1}/{total_batches}")

len(final_predict)



## === cell 23
dog_idx = int(np.where(le.classes_ == "dog")[0][0]) if "dog" in le.classes_ else 1
val_outs = [float(p[dog_idx]) for p in final_predict]
len(val_outs), val_outs[:5]




## === cell 24
def extract_id_from_path(p):
    base = os.path.basename(p)
    m = re.search(r"(\d+)", base)
    if m is None:
        raise ValueError(f"Could not extract numeric id from filename: {base}")
    return int(m.group(1))


final_ids = [extract_id_from_path(p) for p in final_test_img_paths]

pred_df = pd.DataFrame({"id": final_ids, "label": val_outs})

sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
sample = pd.read_csv(sample_path)

submission = sample[["id"]].merge(pred_df, on="id", how="left")
submission["label"] = submission["label"].fillna(0.5).astype(float)

submission["label"] = submission["label"].clip(1e-6, 1 - 1e-6)

submission.head(), submission.shape



## === cell 25
out_file = "/kaggle/working/submission.csv"
submission.to_csv(out_file, index=False)

print("Wrote:", out_file)
print(submission.describe(include="all"))
print("Columns:", submission.columns.tolist())
print("Null labels:", int(submission["label"].isna().sum()))

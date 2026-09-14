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

1.1773532322739644

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

try:
    cv2.setNumThreads(max(1, min(4, (os.cpu_count() or 2) // 2)))
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


base_input = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

train_path = os.path.join(base_input, "train")
final_test_path = os.path.join(base_input, "test", "unknown")

train_img_paths = list(list_images(train_path))
final_test_img_paths = list(list_images(final_test_path))

train_img_paths.sort()
final_test_img_paths.sort()

print("Train images:", len(train_img_paths))
print("Test images:", len(final_test_img_paths))
print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(final_test_path))



## === cell 1
NUM_CLASSES = 2

NUM_VAL_IMAGES = 2500
NUM_TEST_IMAGES = 2500

train_hdf5 = "/kaggle/working/train.hdf5"
val_hdf5 = "/kaggle/working/val.hdf5"
test_hdf5 = "/kaggle/working/test.hdf5"
MODEL_PATH = "/kaggle/working/alexnet_dogs_vs_cats.model"
dataset_mean = "/kaggle/working/dogs_vs_cats_mean.json"
output_path = "/kaggle/working/"

test256_hdf5 = "/kaggle/working/test256.hdf5"




## === cell 2
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




## === cell 3
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



## === cell 4
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



## === cell 5
aap = AspectAwarePreprocessor(227, 227)

from concurrent.futures import ThreadPoolExecutor

if os.path.exists(dataset_mean):
    with open(dataset_mean, "r") as f:
        m = json.load(f)
    B_mean, G_mean, R_mean = float(m["B"]), float(m["G"]), float(m["R"])
    print("Loaded cached channel means (B,G,R):", B_mean, G_mean, R_mean)
else:

    def _mean_one(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = aap.preprocess(img)
        return img.reshape(-1, 3).mean(axis=0)

    max_workers = min(8, (os.cpu_count() or 2))
    sum_mean = np.zeros((3,), dtype=np.float64)
    count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for m1 in ex.map(_mean_one, train_img_paths, chunksize=512):
            if m1 is not None:
                sum_mean += m1.astype(np.float64, copy=False)
                count += 1

    if count:
        means = (sum_mean / float(count)).astype(np.float64, copy=False)
        B_mean, G_mean, R_mean = map(float, means.tolist())
    else:
        B_mean = G_mean = R_mean = 0.0

    print("Channel means (B,G,R):", B_mean, G_mean, R_mean)
    with open(dataset_mean, "w") as f:
        json.dump({"B": B_mean, "G": G_mean, "R": R_mean}, f)




## === cell 6
class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        if image.dtype != np.float32:
            image = image.astype("float32", copy=False)
        image[..., 2] -= self.rMean  # R
        image[..., 1] -= self.gMean  # G
        image[..., 0] -= self.bMean  # B
        return image


class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        if h <= self.height or w <= self.width:
            image = aap.preprocess(image)
            (h, w) = image.shape[:2]
        startY = (h - self.height) // 2
        startX = (w - self.width) // 2
        return image[startY : startY + self.height, startX : startX + self.width]


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




## === cell 7
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

try:
    tf.random.set_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass


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

from concurrent.futures import ThreadPoolExecutor

_MAX_WORKERS = min(8, (os.cpu_count() or 2))
_SHARED_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


def _hdf5_cache_load_or_build(
    h5_path, paths, y, preprocessors, chunksize=256, write_bs=1024
):
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            X = f["X"][:]
            y_kept = f["y"][:]
        return X, y_kept

    paths = list(paths)
    y = np.asarray(y)
    n = len(paths)

    with h5py.File(h5_path, "w") as f:
        dset_X = f.create_dataset(
            "X",
            shape=(n, 227, 227, 3),
            maxshape=(n, 227, 227, 3),
            dtype="float32",
            chunks=(min(write_bs, n), 227, 227, 3),
            shuffle=False,
            compression=None,
        )
        dset_y = f.create_dataset(
            "y",
            shape=(n,),
            maxshape=(n,),
            dtype="int64",
            chunks=(min(write_bs, n),),
            shuffle=False,
            compression=None,
        )

        def _load_and_preprocess(idx_path):
            i, pth = idx_path
            img = cv2.imread(pth, cv2.IMREAD_COLOR)
            if img is None:
                return i, None
            for pr in preprocessors:
                img = pr.preprocess(img)
            if img.dtype != np.float32:
                img = img.astype("float32", copy=False)
            return i, img

        write_pos = 0
        for start in range(0, n, write_bs):
            end = min(n, start + write_bs)
            idx_paths = list(enumerate(paths[start:end], start))
            buf_X = np.empty((end - start, 227, 227, 3), dtype="float32")
            buf_y = np.empty((end - start,), dtype="int64")
            k = 0

            for i, img in _SHARED_POOL.map(
                _load_and_preprocess, idx_paths, chunksize=chunksize
            ):
                if img is None:
                    continue
                buf_X[k] = img
                buf_y[k] = int(y[i])
                k += 1

            if k:
                dset_X[write_pos : write_pos + k] = buf_X[:k]
                dset_y[write_pos : write_pos + k] = buf_y[:k]
                write_pos += k

        dset_X.resize((write_pos, 227, 227, 3))
        dset_y.resize((write_pos,))

    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]
        y_kept = f["y"][:]
    return X, y_kept


def _hdf5_cache_test256_load_or_build(
    h5_path, paths, preprocessors, chunksize=256, write_bs=1024
):
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            X = f["X"][:]
        return X

    paths = list(paths)
    n = len(paths)

    with h5py.File(h5_path, "w") as f:
        dset_X = f.create_dataset(
            "X",
            shape=(n, 256, 256, 3),
            maxshape=(n, 256, 256, 3),
            dtype="float32",
            chunks=(min(write_bs, n), 256, 256, 3),
            shuffle=False,
            compression=None,
        )

        def _load_and_preprocess(pth):
            img = cv2.imread(pth, cv2.IMREAD_COLOR)
            if img is None:
                return None
            for pr in preprocessors:
                img = pr.preprocess(img)
            if img.dtype != np.float32:
                img = img.astype("float32", copy=False)
            return img

        write_pos = 0
        for start in range(0, n, write_bs):
            end = min(n, start + write_bs)
            batch_paths = paths[start:end]
            buf_X = np.empty((end - start, 256, 256, 3), dtype="float32")
            k = 0
            for img in _SHARED_POOL.map(
                _load_and_preprocess, batch_paths, chunksize=chunksize
            ):
                if img is None:
                    continue
                buf_X[k] = img
                k += 1
            if k:
                dset_X[write_pos : write_pos + k] = buf_X[:k]
                write_pos += k

        dset_X.resize((write_pos, 256, 256, 3))

    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]
    return X




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from tensorflow.keras.utils import Sequence


class CachedArraySequence(Sequence):
    def __init__(self, X, y_int, bs, classes, aug=None):
        self.X = X
        self.y_int = np.asarray(y_int, dtype=np.int64)
        self.bs = int(bs)
        self.classes = int(classes)
        self.aug = aug
        self.n = int(self.X.shape[0])

    def __len__(self):
        return int(np.ceil(self.n / float(self.bs)))

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(self.n, start + self.bs)
        batchX = self.X[start:end]
        batchY = to_categorical(self.y_int[start:end], self.classes)
        if self.aug is not None and batchX.shape[0]:
            batchX, batchY = next(
                self.aug.flow(batchX, batchY, batch_size=batchX.shape[0], shuffle=False)
            )
        return batchX, batchY




## === cell 9
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




## === cell 10
from tensorflow.keras.optimizers import Adam

model = Alexnet.build(227, 227, 3, NUM_CLASSES, reg=0.0002)
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

BATCH = 64
EPOCHS = 1

X_train, y_train_kept = _hdf5_cache_load_or_build(
    train_hdf5, train_img_paths, y_train, [pp, mp, iap]
)
X_val, y_val_kept = _hdf5_cache_load_or_build(
    val_hdf5, val_img_paths, y_val, [sp, mp, iap]
)

train_seq = CachedArraySequence(
    X_train, y_train_kept, bs=BATCH, classes=NUM_CLASSES, aug=aug
)
val_seq = CachedArraySequence(
    X_val, y_val_kept, bs=BATCH, classes=NUM_CLASSES, aug=None
)

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    verbose=1,
    workers=1,
    use_multiprocessing=False,
)

model.save(os.path.join(output_path, "alexnet_trained_local.h5"))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/793665179.py in <cell line: 0>()
     24 )
     25 
---> 26 history = model.fit(
     27     train_seq,
     28     validation_data=val_seq,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 11
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)


def _ten_crops_batch_from_256(batch256):
    tw = th = 227
    w = h = 256
    dW = (w - tw) // 2
    dH = (h - th) // 2

    n = batch256.shape[0]
    out = np.empty((n * 10, th, tw, 3), dtype=batch256.dtype)

    out[0 * n : 1 * n] = batch256[:, 0:th, 0:tw, :]  # tl
    out[1 * n : 2 * n] = batch256[:, 0:th, w - tw : w, :]  # tr
    out[2 * n : 3 * n] = batch256[:, h - th : h, w - tw : w, :]  # br
    out[3 * n : 4 * n] = batch256[:, h - th : h, 0:tw, :]  # bl
    out[4 * n : 5 * n] = batch256[:, dH : h - dH, dW : w - dW, :]  # cc

    out[5 * n : 6 * n] = out[0 * n : 1 * n, :, ::-1, :]
    out[6 * n : 7 * n] = out[1 * n : 2 * n, :, ::-1, :]
    out[7 * n : 8 * n] = out[2 * n : 3 * n, :, ::-1, :]
    out[8 * n : 9 * n] = out[3 * n : 4 * n, :, ::-1, :]
    out[9 * n : 10 * n] = out[4 * n : 5 * n, :, ::-1, :]

    return out


X_test256 = _hdf5_cache_test256_load_or_build(
    test256_hdf5, final_test_img_paths, [mp, aap2]
)

final_predict = []
bs = 256  # fewer Python iterations; does not change math

for i in range(0, X_test256.shape[0], bs):
    batch256 = X_test256[i : i + bs]  # (n,256,256,3)
    if batch256.shape[0] == 0:
        continue

    n_img = batch256.shape[0]
    crops_batch = _ten_crops_batch_from_256(batch256)  # (n*10,227,227,3)

    pred = model.predict_on_batch(crops_batch)
    pred = pred.reshape(n_img, 10, NUM_CLASSES).mean(axis=1)
    final_predict.append(pred)

final_predict = np.concatenate(final_predict, axis=0).astype("float32", copy=False)
print("Pred shape:", final_predict.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1879887628.py in <cell line: 0>()
     42 
     43     n_img = batch256.shape[0]
---> 44     crops_batch = _ten_crops_batch_from_256(batch256)  # (n*10,227,227,3)
     45 
     46     pred = model.predict_on_batch(crops_batch)

/tmp/ipykernel_11/1879887628.py in _ten_crops_batch_from_256(batch256)
     16     out[2 * n : 3 * n] = batch256[:, h - th : h, w - tw : w, :]  # br
     17     out[3 * n : 4 * n] = batch256[:, h - th : h, 0:tw, :]  # bl
---> 18     out[4 * n : 5 * n] = batch256[:, dH : h - dH, dW : w - dW, :]  # cc
     19 
     20     out[5 * n : 6 * n] = out[0 * n : 1 * n, :, ::-1, :]

ValueError: could not broadcast input array from shape (256,228,228,3) into shape (256,227,227,3)

## === cell 12
class_to_index = {c: int(le.transform([c])[0]) for c in le.classes_}
dog_index = class_to_index.get("dog", 1)

dog_prob = final_predict[:, dog_index]

final_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in final_test_img_paths
]

if len(dog_prob) != len(final_ids):
    final_ids = final_ids[: len(dog_prob)]

submission = pd.DataFrame({"id": final_ids, "label": dog_prob})
submission = submission.sort_values("id").reset_index(drop=True)

eps = 1e-7
submission["label"] = submission["label"].clip(eps, 1 - eps)

print(submission.head())
print("Rows:", len(submission))

submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2997544116.py in <cell line: 0>()
      2 dog_index = class_to_index.get("dog", 1)
      3 
----> 4 dog_prob = final_predict[:, dog_index]
      5 
      6 final_ids = [

TypeError: list indices must be integers or slices, not tuple

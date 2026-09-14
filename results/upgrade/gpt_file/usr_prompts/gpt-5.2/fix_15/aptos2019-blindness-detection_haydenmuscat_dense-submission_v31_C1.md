# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import math
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import psutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

IMG_DIM = 224
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        min(4, psutil.cpu_count(logical=True))
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

_CANDIDATE_INPUT_FOLDERS = [
    "../input/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/",
    "../input/",
    "/kaggle/input/",
]

INPUT_FOLDER = None
for p in _CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p if p.endswith("/") else (p + "/")
        break

if INPUT_FOLDER is None:
    for base in ["../input", "/kaggle/input"]:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if "train.csv" in files and "test.csv" in files:
                    INPUT_FOLDER = root if root.endswith("/") else (root + "/")
                    break
            if INPUT_FOLDER is not None:
                break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset folder containing train.csv/test.csv"
    )

print("INPUT_FOLDER =", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
print("Listing INPUT_FOLDER:", os.listdir(INPUT_FOLDER)[:20])

TRAIN_IMG_DIR = os.path.join(INPUT_FOLDER, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_FOLDER, "test_images")

if not os.path.isdir(TRAIN_IMG_DIR) or not os.path.isdir(TEST_IMG_DIR):
    nested = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection")
    if os.path.exists(nested):
        INPUT_FOLDER = nested if nested.endswith("/") else (nested + "/")
        TRAIN_IMG_DIR = os.path.join(INPUT_FOLDER, "train_images")
        TEST_IMG_DIR = os.path.join(INPUT_FOLDER, "test_images")

print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR)
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR)

gc.collect()




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while top < bottom and middleCol[top] == 0:
        top += 1
    while bottom > top and middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while left < right and middleRow[left] == 0:
        left += 1
    while right > left and middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100 or top >= bottom or left >= right:
        return img

    return img[top:bottom, left:right]


def benSimple(img, weight=4, gamma=20):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def reflectAndSquareUp(img):
    height, width = img.shape[:2]

    if height > width:
        offset = (height - width) // 2
        return img[offset : offset + width]
    else:
        if img.ndim == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        h1 = (width - height) // 2
        h2 = h1 + height
        new_img[h1:h2, :] = img

        if h1 > 0:
            new_img[:h1, :] = img[:h1][::-1, :]

        bot = width - h2
        if bot > 0:
            new_img[h2:, :] = img[height - bot : height][::-1, :]

        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image_in, gamma=1.0):
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        gamma = 1.0
    invGamma = 1.0 / gamma
    key = invGamma
    table = _GAMMA_LUT_CACHE.get(key)
    if table is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((x**invGamma) * 255.0, 0, 255).astype(np.uint8)
        _GAMMA_LUT_CACHE[key] = table
    return cv2.LUT(image_in, table)


def processBenWeird(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    if med <= 0:
        g = 1.0
    else:
        g = 1 + np.log(90) - np.log(med)
    equalised = adjust_gamma(circled, g)

    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 3
gc.collect()



## === cell 4
train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
test_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=train_df["diagnosis"].values,
)

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)


def to_onehot(y, num_classes=NUM_CLASSES):
    y = np.asarray(y).astype(int)
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


y_train_oh = to_onehot(train_split["diagnosis"].values)
y_val_oh = to_onehot(val_split["diagnosis"].values)

print("Train size:", len(train_split), "Val size:", len(val_split))
gc.collect()




## === cell 5
def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    return model


model = create_model()

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
gc.collect()



## === cell 6
from concurrent.futures import ThreadPoolExecutor

_PREPROCESS_DISK_DIR = "/kaggle/working/_preprocessed_cache_v3_mmap"
os.makedirs(_PREPROCESS_DISK_DIR, exist_ok=True)


def _img_path(images_dir, id_code):
    fn = id_code + ".png" if not str(id_code).endswith(".png") else str(id_code)
    return os.path.join(images_dir, fn)


def _cache_key_for_paths(paths, img_dim=IMG_DIM):
    h = hashlib.md5()
    h.update(str(img_dim).encode("utf-8"))
    for p in paths:
        h.update(b"\n")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _mmap_paths(cache_prefix):
    data_path = os.path.join(_PREPROCESS_DISK_DIR, f"{cache_prefix}_data.uint8.mmap")
    meta_path = os.path.join(_PREPROCESS_DISK_DIR, f"{cache_prefix}_meta.json")
    return data_path, meta_path


def _process_one_path_to_uint8(path):
    bgr = cv2.imread(path)
    rgb = processBenWeird(bgr)  # returns uint8 RGB IMG_DIMxIMG_DIMx3
    return rgb


def _build_or_load_mmap_uint8(paths, cache_prefix, max_workers=None, chunksize=32):
    data_path, meta_path = _mmap_paths(cache_prefix)
    n = len(paths)
    shape = (n, IMG_DIM, IMG_DIM, CHANNELS)

    if os.path.exists(data_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if tuple(meta.get("shape", ())) == shape and meta.get("dtype") == "uint8":
            return np.memmap(data_path, mode="r", dtype=np.uint8, shape=shape)

    if max_workers is None:
        cpu = psutil.cpu_count(logical=True) or 2
        max_workers = min(8, max(2, cpu))

    arr = np.memmap(data_path, mode="w+", dtype=np.uint8, shape=shape)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, rgb in enumerate(
            ex.map(_process_one_path_to_uint8, paths, chunksize=chunksize)
        ):
            arr[i] = rgb

    arr.flush()
    with open(meta_path, "w") as f:
        json.dump({"shape": list(shape), "dtype": "uint8"}, f)

    return np.memmap(data_path, mode="r", dtype=np.uint8, shape=shape)


train_ids = train_split["id_code"].values
val_ids = val_split["id_code"].values
test_ids = test_df["id_code"].values

all_train_paths = [_img_path(TRAIN_IMG_DIR, idc) for idc in train_df["id_code"].values]
test_paths = [_img_path(TEST_IMG_DIR, idc) for idc in test_ids]

print("Preprocessing all train images once (threaded + single mmap cache)...")
cache_prefix_train = "train_" + _cache_key_for_paths(all_train_paths)
x_all_train_uint8 = _build_or_load_mmap_uint8(all_train_paths, cache_prefix_train)

id_to_pos = pd.Series(
    np.arange(len(train_df), dtype=np.int64), index=train_df["id_code"].values
)
train_pos = id_to_pos.loc[train_ids].values.astype(np.int64, copy=False)
val_pos = id_to_pos.loc[val_ids].values.astype(np.int64, copy=False)

print("Preprocessing test images (threaded + single mmap cache)...")
cache_prefix_test = "test_" + _cache_key_for_paths(test_paths)
x_test_uint8 = _build_or_load_mmap_uint8(test_paths, cache_prefix_test)

gc.collect()




## === cell 7
class IndexedArraySequence(keras.utils.Sequence):
    def __init__(self, x_mmap, indices, y, datagen, batch_size, shuffle, seed=42):
        self.x = x_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = y
        self.datagen = datagen
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.RandomState(seed)
        self._order = np.arange(len(self.indices), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self._order)

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        batch_order = self._order[sl]
        batch_idx = self.indices[batch_order]
        batch_x = np.asarray(self.x[batch_idx], dtype=np.uint8)  # batch-sized copy only
        batch_y = self.y[batch_order] if self.y is not None else None

        if batch_y is None:
            out = self.datagen.flow(batch_x, batch_size=len(batch_x), shuffle=False)
            return next(out)

        out = self.datagen.flow(
            batch_x, batch_y, batch_size=len(batch_x), shuffle=False
        )
        return next(out)


train_datagen = dataGenerator(0.10)
val_datagen = dataGenerator(0.00)

train_seq = IndexedArraySequence(
    x_all_train_uint8,
    train_pos,
    y_train_oh,
    train_datagen,
    BATCH_SIZE,
    shuffle=True,
    seed=42,
)
val_seq = IndexedArraySequence(
    x_all_train_uint8,
    val_pos,
    y_val_oh,
    val_datagen,
    BATCH_SIZE,
    shuffle=False,
    seed=42,
)

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=2,
    verbose=1,
)
gc.collect()




## === cell 8
class IndexedPredictSequence(keras.utils.Sequence):
    def __init__(self, x_mmap, indices, datagen, batch_size):
        self.x = x_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.datagen = datagen
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.indices[sl]
        batch_x = np.asarray(self.x[idx], dtype=np.uint8)
        out = self.datagen.flow(batch_x, batch_size=len(batch_x), shuffle=False)
        return next(out)


def _predict_with_datagen_on_mmap(x_mmap, indices, datagen, batch_size=32):
    seq = IndexedPredictSequence(x_mmap, indices, datagen, batch_size)
    preds = model.predict(seq, verbose=0)
    return preds.astype(np.float32, copy=False)


def make_predictions(d_set, jitters=5):
    if d_set == "test":
        x_mmap = x_test_uint8
        indices = np.arange(x_mmap.shape[0], dtype=np.int64)
    elif d_set == "train":
        x_mmap = x_all_train_uint8
        indices = train_pos
    else:
        raise ValueError("d_set must be 'test' or 'train'")

    total = len(indices)
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    jit_values = [0.02 * j for j in range(jitters)]
    all_preds = np.empty((total, jitters, NUM_CLASSES), dtype=np.float32)

    for j, jit in enumerate(jit_values):
        dg = dataGenerator(jit)
        all_preds[:, j, :] = _predict_with_datagen_on_mmap(
            x_mmap, indices, dg, batch_size=BATCH_SIZE
        )
        print(f"jitter {j+1}/{jitters} finished")

    predictions = np.median(all_preds, axis=1)
    return predictions


def label_convert(preds):
    y_val = preds > 0.5
    out = y_val.astype(int).sum(axis=1) - 1
    out = np.clip(out, 0, 4).astype(int)
    return out


val_eval_datagen = dataGenerator(0.00)
val_preds = _predict_with_datagen_on_mmap(
    x_all_train_uint8, val_pos, val_eval_datagen, batch_size=BATCH_SIZE
)

val_classes_pred = label_convert(val_preds)
val_true = val_split["diagnosis"].values.astype(int)
print("Val QWK:", cohen_kappa_score(val_true, val_classes_pred, weights="quadratic"))
print("Val confusion matrix:\n", confusion_matrix(val_true, val_classes_pred))

gc.collect()



## === cell 9
test_predictions = make_predictions("test", jitters=5)
test_classes = label_convert(test_predictions)

print(test_predictions[:2])
print(test_classes[:10])

sub_df = test_df.copy()
sub_df["diagnosis"] = test_classes.astype(int)

sub_df = sub_df[["id_code", "diagnosis"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
gc.collect()

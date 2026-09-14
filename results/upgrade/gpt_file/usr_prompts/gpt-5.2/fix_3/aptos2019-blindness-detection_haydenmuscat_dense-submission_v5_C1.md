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
import gc
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt  # noqa: F401

from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: F401


IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

BASE_INPUT = "../input"
CANDIDATE_APTOS = [
    os.path.join(BASE_INPUT, "aptos2019-blindness-detection"),
    os.path.join(
        BASE_INPUT, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
    ),
]
INPUT_FOLDER = None
for p in CANDIDATE_APTOS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p.rstrip("/") + "/"
        break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder under ../input"
    )

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"

print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("TRAIN_IMAGES_DIR exists:", os.path.exists(TRAIN_IMAGES_DIR))
print("TEST_IMAGES_DIR exists:", os.path.exists(TEST_IMAGES_DIR))
print("List ../input:", os.listdir(BASE_INPUT)[:50])



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].astype(str)
test_df["filename"] = test_df["id_code"].apply(lambda x: x + ".png")
test_df.head()




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    y_val = np.clip(y_val, 0, 4)
    return y_val


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1
    bottom = len(rowMaxes) - 1
    while bottom >= 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    if top >= bottom:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right >= 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or left >= right:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified


def load_image_rgb_from_path(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    return processImageBgrToRgb(bgr)




## === cell 3
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment. "
        "This notebook requires TensorFlow/Keras to build the model."
    ) from e


def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",  # minimal change: use standard pretrained weights since external .h5 is unavailable
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNEL_SIZE),
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

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["filename"] = train_df["id_code"].apply(lambda x: x + ".png")


def diagnosis_to_multilabel(d):
    d = int(d)
    return np.array([1, d >= 1, d >= 2, d >= 3, d >= 4], dtype=np.float32)


y_multi = np.stack(
    [diagnosis_to_multilabel(d) for d in train_df["diagnosis"].values], axis=0
)

idx = np.arange(len(train_df))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
train_df_va = train_df.iloc[va_idx].reset_index(drop=True)
y_tr = y_multi[tr_idx]
y_va = y_multi[va_idx]


class SimpleSequence(tf.keras.utils.Sequence):
    def __init__(self, df, y, images_dir, batch_size=32, shuffle=True):
        self.df = df
        self.y = y
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx_batch):
        sl = slice(idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size)
        batch_ids = self.indexes[sl]
        bs = len(batch_ids)
        X = np.empty((bs, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
        Y = self.y[batch_ids].astype(np.float32)
        for j, ridx in enumerate(batch_ids):
            fn = self.df.loc[ridx, "filename"]
            path = os.path.join(self.images_dir, fn)
            try:
                X[j] = load_image_rgb_from_path(path)
            except Exception:
                X[j] = 128.0
        X = X / 255.0
        return X, Y


train_seq = SimpleSequence(
    train_df_tr, y_tr, TRAIN_IMAGES_DIR, batch_size=BATCH_SIZE, shuffle=True
)
val_seq = SimpleSequence(
    train_df_va, y_va, TRAIN_IMAGES_DIR, batch_size=BATCH_SIZE, shuffle=False
)

model.fit(train_seq, validation_data=val_seq, epochs=1, verbose=1)

gc.collect()




## === cell 4
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1 + jitter],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 5
block_size = 500
total = test_df.index.size

y_pred_list = np.zeros(total, dtype=int)

for start in range(0, total, block_size):
    gc.collect()
    end = min(start + block_size, total)

    img_list = np.empty((end - start, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for i, filename in enumerate(test_df.iloc[start:end].filename):
        try:
            img_list[i, :, :, :] = load_image_rgb_from_path(
                os.path.join(TEST_IMAGES_DIR, filename)
            )
        except Exception:
            img_list[i, :, :, :] = 128.0

    num = 7
    prediction_lists = np.zeros((len(img_list), num, 5), dtype=np.float32)
    for i in range(num):
        datagen = dataGenerator(0.03).flow(
            img_list, shuffle=False, batch_size=BATCH_SIZE
        )
        prediction_lists[:, i] = model.predict(datagen, steps=len(datagen), verbose=0)

    predictions = np.median(prediction_lists, axis=1)
    y_pred_list[start:end] = label_convert(predictions > 0.5)

    print(f"{start} - {end} finished")



## === cell 6
sub_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df["diagnosis"] = y_pred_list.astype(int)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
print("diagnosis value counts:\n", sub_df["diagnosis"].value_counts().sort_index())

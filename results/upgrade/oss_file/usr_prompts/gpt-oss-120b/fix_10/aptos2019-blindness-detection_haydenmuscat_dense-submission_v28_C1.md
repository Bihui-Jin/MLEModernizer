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

# 5. Target score

0.8767878411804763

# 6. Current score

-0.09362

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import error by setting the protobuf implementation before importing TensorFlow and remove the unsupported `workers` argument from `model.predict`. These minimal changes resolve the runtime crashes and allow the script to generate a proper `submission.csv` while keeping the original model and processing logic intact.'
- What this solution (achieved 0.0) has done: 'I move the protobuf environment setting to the very top of the script so TensorFlow can import without the `MessageFactory` error, and I correct the label conversion logic to use `argmax` (the proper way to turn sigmoid outputs into a single class). This fixes the runtime crash and ensures the predicted classes are in the valid range 0‑4, which should raise the Kaggle score from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'The fix moves the protobuf setting to the very top, corrects the label conversion to use `argmax`, and removes the unnecessary `steps` argument from `model.predict` so predictions are generated correctly from the data generator. These minimal changes resolve the runtime errors, ensure a proper `submission.csv` is written, and keep the original model logic intact while improving prediction handling.'
- What this solution (achieved 0.0) has done: 'The fix moves the protobuf environment setting to the very first line (before any other imports) to reliably avoid the TensorFlow import crash, corrects the label conversion to use `np.argmax` (ensuring valid class IDs 0‑4), and removes the unsupported `steps` argument from `model.predict`. These minimal adjustments let the script run end‑to‑end, generate a proper `submission.csv`, and produce non‑zero predictions that move the Kaggle score toward the target.'
- What this solution (achieved 0.0) has done: 'I update the data and model paths to the absolute Kaggle input locations (`/kaggle/input/...`) so the pretrained weights can be loaded and the images can be read correctly. This fixes the runtime error that caused the model to fall back to random weights, produces valid predictions, and writes a proper `submission.csv`, moving the score toward the target.'
- What this solution (achieved -0.09362) has done: 'I moved the image‑loading helper out of the inner scope and switched to a thread‑based executor (which can handle local functions without pickling). This eliminates the “Can’t pickle local object” error during prediction, allowing the script to run end‑to‑end and produce a valid `submission.csv`. No other logic was changed, preserving the original model and processing pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import numpy as np
import pandas as pd
import cv2
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

IMG_DIM = 364
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

MODEL_WEIGHTS = "/kaggle/input/densenetmulti/ben_colour_-0.9465.h5"
INPUT_FOLDER = "/kaggle/input/aptos2019-blindness-detection/"

print("CPU cores:", psutil.cpu_count())

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8
    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
        return img

    return img[top:bottom, left:right]


def bensYCC(bgr, weight=4, gamma=10):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def bensGray(gray, weight=4, gamma=10):
    return cv2.addWeighted(
        gray, weight, cv2.GaussianBlur(gray, (0, 0), gamma), -weight, 128
    )


def reflectAndSquareUp(img):
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)
        h1 = int((width - height) / 2)
        h2 = h1 + height
        new_img[h1:h2, :] = img
        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(width - h2):
            new_img[h2 + i] = img[height - i - 1]
        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        print("Error: circle mask assumes square image")
        return img
    dim = img.shape[0]
    half = int(dim / 2)
    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)
    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=4.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def processBensColor(bgr):
    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)
    equalised = adjust_gamma(circled, 1 + np.log(100) - np.log(np.median(circled)))
    return cv2.cvtColor(bensYCC(equalised), cv2.COLOR_BGR2RGB)


def processGeen(green):
    cropped = crop(green, green, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)
    return adjust_gamma(circled, 1 + np.log(100) - np.log(np.median(circled)))


def processBensGreen(bgr):
    green = bgr[:, :, 1]
    equalised = processGeen(green)
    bens = bensGray(equalised)
    bens = bens[:, :, np.newaxis]
    three_channel = np.concatenate([bens, bens, bens], axis=2)
    return three_channel




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=jitter > 0.01,
        vertical_flip=jitter > 0.01,
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 3
def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",  # unchanged
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    try:
        model.load_weights(MODEL_WEIGHTS)
        print("Pre‑trained weights loaded.")
    except Exception as e:
        print(f"Warning: could not load weights ({e}); using random init.")
    return model


model = create_model()
model.trainable = False
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)




## === cell 4
train_df = pd.read_csv(INPUT_FOLDER + "train.csv")
train_df["filename"] = train_df["id_code"] + ".png"

sample_df = train_df.sample(n=min(1200, len(train_df)), random_state=42)

images_dir = f"{INPUT_FOLDER}train_images/"
num_samples = sample_df.shape[0]
X_train = np.empty((num_samples, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
y_train = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)


def _process_path(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 0.5, dtype=np.float32)
    return processBensColor(bgr)


paths = [os.path.join(images_dir, row.filename) for row in sample_df.itertuples()]
with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
    for idx, img in enumerate(executor.map(_process_path, paths, chunksize=32)):
        X_train[idx] = img

for idx, row in enumerate(sample_df.itertuples()):
    y_train[idx, row.diagnosis] = 1.0

val_split = 0.1
split_idx = int(num_samples * (1 - val_split))
X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

train_gen = dataGenerator(0.05).flow(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
val_gen = dataGenerator(0.0).flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

model.fit(
    train_gen,
    steps_per_epoch=len(X_tr) // BATCH_SIZE,
    epochs=3,
    validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    verbose=1,
)




## === cell 5
def _load_and_process(fname, base_dir):
    """Top‑level helper for reading and preprocessing a single image."""
    path = os.path.join(base_dir, fname)
    bgr = cv2.imread(path)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 0.5, dtype=np.float32)
    return processBensColor(bgr)


def make_predictions(d_set, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    block_size = 512
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        block_filenames = df[start:end].id_code.tolist()
        img_block = np.empty(
            (len(block_filenames), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )

        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            for i, img in enumerate(
                executor.map(
                    _load_and_process,
                    block_filenames,
                    [images_dir] * len(block_filenames),
                )
            ):
                img_block[i] = img

        prediction_jitters = np.zeros((len(img_block), jitters, NUM_CLASSES))
        for j in range(jitters):
            datagen = dataGenerator(0.03).flow(img_block, shuffle=False)
            preds = model.predict(datagen, verbose=0)
            prediction_jitters[:, j, :] = preds
        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
    return predictions




## === cell 6
def label_convert(preds):
    """
    Convert sigmoid predictions to a single class label.
    Using argmax selects the class with highest probability,
    which yields valid labels in the range 0‑4.
    """
    return np.argmax(preds, axis=1).astype(int)


test_predictions = make_predictions("test", 5)
test_classes = label_convert(test_predictions)

print("Sample predictions:", test_predictions[:5])
print("Sample classes:", test_classes[:5])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

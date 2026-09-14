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

import multiprocessing as mp

mp.set_start_method("fork", force=True)

import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

try:
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator as KImageDataGenerator,
    )
except Exception:
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator as KImageDataGenerator,
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("Current folder contents:", os.listdir("."))
print("Input folder contents:", os.listdir("../input/"))
print("Aptos folder contents:", os.listdir(INPUT_FOLDER))




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


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def benSimple(img, weight=4, gamma=20):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


def reflectAndSquareUp(img):
    height, width = img.shape[:2]
    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        pad_total = width - height
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        if img.ndim == 3:
            return np.pad(
                img, ((pad_before, pad_after), (0, 0), (0, 0)), mode="reflect"
            )
        else:
            return np.pad(img, ((pad_before, pad_after), (0, 0)), mode="reflect")


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        print("Error: circle mask assumes square image")
        return img
    dim = img.shape[0]
    half = int(dim / 2)
    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)
    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


_gamma_lut_cache = {}


def adjust_gamma(image, gamma=1.0):
    if gamma not in _gamma_lut_cache:
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        _gamma_lut_cache[gamma] = table
    else:
        table = _gamma_lut_cache[gamma]
    return cv2.LUT(image, table)


def processBenNormal(bgr):
    green = bgr[:, :, 1]  # use green channel as grayscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    """
    Returns an ImageDataGenerator with modest augmentations.
    The jitter parameter controls the magnitude of the augmentations.
    """
    datagen = KImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=(jitter > 0.01),
        vertical_flip=(jitter > 0.01),
        zoom_range=[max(0.7, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def _process_image(args):
    """Helper for multiprocessing – returns preprocessed image or placeholder."""
    filename, images_dir = args
    path = os.path.join(images_dir, filename)
    bgr = cv2.imread(path)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    try:
        return pre_process_function(bgr)
    except Exception:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)


def load_images(df, images_dir):
    """
    Efficiently read and preprocess images defined in df.
    Uses all CPU cores and a larger chunksize to reduce inter‑process overhead.
    Pre‑allocates the NumPy array to avoid Python‑list staging.
    """
    n = len(df)
    imgs = np.empty((n, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    args_list = [(filename, images_dir) for filename in df.id_code]

    pool_size = max(1, mp.cpu_count() - 1)

    with mp.Pool(processes=pool_size) as pool:
        for i, img in enumerate(
            pool.imap_unordered(_process_image, args_list, chunksize=64)
        ):
            imgs[i] = img

    gc.collect()
    return imgs




## === cell 4
def create_model():
    """Builds the original DenseNet121‑based classifier."""
    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )
    base.trainable = False  # keep pretrained weights frozen

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["id_code"] = train_df["id_code"].apply(lambda x: f"{x}.png")
train_images_dir = os.path.join(INPUT_FOLDER, "train_images/")

print("Loading and preprocessing training images...")
train_images = load_images(train_df, train_images_dir)

train_labels = to_categorical(train_df["diagnosis"].values, NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    train_labels,
    test_size=0.1,
    random_state=42,
    stratify=train_df["diagnosis"],
)

model = create_model()

print("Starting model training...")
model.fit(
    dataGenerator(jitter=0.1).flow(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=True
    ),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=5,
    validation_data=dataGenerator(jitter=0.0).flow(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
    ),
    validation_steps=len(X_val) // BATCH_SIZE,
    verbose=2,
)

print("Training completed.")




## === cell 6
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: f"{x}.png")

    print(f"Loading and preprocessing {d_set} images...")
    img_array = load_images(df, images_dir)  # reuse existing efficient loader

    total = img_array.shape[0]
    prediction_jitters = np.zeros((total, jitters, NUM_CLASSES))

    jitter_val = 0.0
    for j in range(jitters):
        datagen = dataGenerator(jitter_val).flow(
            img_array, batch_size=BATCH_SIZE, shuffle=False
        )
        prediction_jitters[:, j] = model.predict(datagen, steps=len(datagen), verbose=0)
        jitter_val += 0.02
        gc.collect()

    predictions = np.median(prediction_jitters, axis=1)
    print(f"Completed predictions on {total} images.")
    return predictions




## === cell 7
def prediction_convert(predictions, thresholds=None):
    """
    Convert class‑probability predictions to integer diagnosis labels.
    Uses argmax to select the most likely class (0‑4). The optional
    thresholds argument is kept for compatibility but ignored.
    """
    if predictions.ndim != 2 or predictions.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected predictions shape (n, {NUM_CLASSES}), got {predictions.shape}"
        )
    return np.argmax(predictions, axis=1)




## === cell 8
test_predictions = make_predictions("test", processBenNormal, model, jitters=5)
test_classes = prediction_convert(test_predictions)

print("Sample predictions (probabilities):", test_predictions[:10])
print("Sample classes:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

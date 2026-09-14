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
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from concurrent.futures import ThreadPoolExecutor

tf.config.optimizer.set_jit(True)

_candidate_folders = [
    "./input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
]

INPUT_FOLDER = None
for _cand in _candidate_folders:
    if os.path.isdir(_cand) and os.path.exists(os.path.join(_cand, "train.csv")):
        INPUT_FOLDER = _cand
        break

if INPUT_FOLDER is None:
    INPUT_FOLDER = "."

if not INPUT_FOLDER.endswith("/"):
    INPUT_FOLDER = INPUT_FOLDER + "/"

IMG_DIM = 224  # size expected by DenseNet121
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 128  # larger batch reduces number of steps
NORMAL_WEIGHTS = ""  # path to custom weights (empty => use ImageNet)

np.random.seed(42)
tf.random.set_seed(42)

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    col = gray[:, gray.shape[1] // 2] > thresh
    idx = np.where(col)[0]
    if idx.size == 0:
        return img
    top, bottom = idx[0], idx[-1]

    row = gray[gray.shape[0] // 2, :] > thresh
    idx = np.where(row)[0]
    if idx.size == 0:
        return img
    left, right = idx[0], idx[-1]

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


def benYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def processBenNormal(bgr):
    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    bens = benYCC(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def create_model(weights_path):
    """Create DenseNet121‑based model.
    If a custom weight file exists it is loaded, otherwise ImageNet weights are used."""
    base_weights = None if os.path.exists(weights_path) else "imagenet"
    model = Sequential()
    base = DenseNet121(
        weights=base_weights,
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )
    base.trainable = False  # <<< freeze backbone, speeds up training dramatically
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print("Info: Using ImageNet pretrained weights (no custom .h5 found).")

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 4
def _load_and_process(filepath, processing_function):
    """Helper to read an image and apply the shared preprocessing."""
    bgr = cv2.imread(filepath)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    return processing_function(bgr)


def train_on_full_data(model, processing_function, epochs=3):
    """Very light training on the provided train.csv to give the model task‑specific tuning.
    Images are pre‑processed in parallel and cached in memory to avoid repeated disk I/O.
    """
    np.random.seed(42)

    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    train_df["filename"] = train_df.id_code.apply(lambda x: x + ".png")
    N = len(train_df)

    print("Pre‑processing all training images (once, parallel)...")
    imgs_all = np.empty((N, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    labels_all = np.zeros((N, NUM_CLASSES), dtype=np.float32)

    def _loader(row):
        filepath = f"{INPUT_FOLDER}train_images/{row.filename}"
        img = _load_and_process(filepath, processing_function)
        label = int(row.diagnosis)
        return img, label

    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, (img, label) in enumerate(
            executor.map(_loader, train_df.itertuples())
        ):
            imgs_all[idx] = img
            labels_all[idx, label] = 1

    imgs_all = imgs_all.astype(np.float32) / 255.0

    model.fit(
        imgs_all,
        labels_all,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        shuffle=True,
        verbose=1,
    )
    return model




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=7):
    """
    Load all images of the given dataset once (parallel), then apply
    jitter‑augmented predictions on the whole set. This removes the
    per‑block thread‑pool overhead while preserving the exact TTA
    (test‑time augmentation) logic used originally.
    """
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    N = len(df)
    print(f"Loading and preprocessing {N} images for {d_set} set...")
    imgs_all = np.empty((N, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    def _loader(filename):
        filepath = images_dir + filename
        return _load_and_process(filepath, processing_function)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, img in enumerate(executor.map(_loader, df.id_code)):
            imgs_all[i] = img

    imgs_all = imgs_all.astype(np.float32) / 255.0

    prediction_jitters = np.zeros((N, jitters, NUM_CLASSES), dtype=np.float32)

    jit = 0.0
    for i in range(jitters):
        datagen = dataGenerator(jit).flow(
            imgs_all, shuffle=False, batch_size=BATCH_SIZE
        )
        prediction_jitters[:, i] = model.predict(datagen, verbose=0)
        gc.collect()
        jit += 0.0075

    predictions = np.median(prediction_jitters, axis=1)
    print(f"Finished predictions for {d_set} set.")
    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thresholded = np.zeros(predictions.shape)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]
    y_val = thresholded.astype(int).sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros(predictions.shape)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]

    y_val = np.zeros((predictions.shape[0]), dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(4, -1, -1):
            if thresholded[i][j]:
                y_val[i] = j
                break
    return y_val




## === cell 7
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1




## === cell 8
model = create_model(NORMAL_WEIGHTS)
model = train_on_full_data(model, processBenNormal, epochs=3)

preds = make_predictions("test", processBenNormal, model)

thresholds = [0.5, 0.625, 0.515625, 0.421875, 0.4375]
test_classes = prediction_convert_sum(preds, thresholds)
print("First 10 predicted classes:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

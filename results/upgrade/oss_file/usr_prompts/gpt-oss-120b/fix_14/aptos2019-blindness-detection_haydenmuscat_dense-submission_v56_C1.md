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
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd
import gc
import cv2
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.callbacks import (
        Callback,
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    tf = None
    print("TensorFlow import failed:", e)
    raise ImportError("TensorFlow is required for this script.")

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

from sklearn.metrics import cohen_kappa_score  # needed for threshold search

IMG_DIM = 256
BATCH_SIZE = 256
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/normal.h5"

print("Current folder listing:", os.listdir("."))
print("Parent folder listing:", os.listdir(".."))
print("Input root listing:", os.listdir("../input/"))
print("Aptos data listing:", os.listdir("../input/aptos2019-blindness-detection"))

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"




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
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=20):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


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


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


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
    """
    Minimal dummy generator that mimics the .flow() interface
    but simply returns the input array normalised to [0,1].
    This avoids the unavailable ImageDataGenerator class.
    """

    class _DummyGen:
        def flow(self, x, shuffle=False):
            return x.astype("float32") / 255.0

    return _DummyGen()




## === cell 3
def load_images(d_set):
    """Load and preprocess images for a given dataset (train or test) using parallel I/O.
    Results are cached to a .npy file to avoid re‑processing on subsequent runs.
    """
    cache_path = f"{d_set}_images.npy"
    if os.path.exists(cache_path):
        imgs = np.load(cache_path)
        df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
        df["id_code"] = df["id_code"].apply(lambda x: f"{x}.png")
        return imgs, df

    from concurrent.futures import ThreadPoolExecutor

    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["id_code"] = df["id_code"].apply(lambda x: f"{x}.png")
    num_images = len(df)
    imgs = np.empty((num_images, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    def read_and_process(filename):
        try:
            bgr = cv2.imread(images_dir + filename)
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            return pre_process_function(bgr)
        except Exception:
            return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(read_and_process, df["id_code"]))
    imgs[:] = results

    np.save(cache_path, imgs)
    return imgs, df




## === cell 4
def create_model(weights_path):
    """Create DenseNet121‑based model. Load weights only if the file exists."""
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pretrained weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        print(
            "Pretrained weight file not found – proceeding with random initialization."
        )
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def fine_tune_model(model, train_imgs, train_labels):
    """Quick fine‑tuning on the training set (3 epochs)."""
    train_imgs_norm = train_imgs.astype("float32") / 255.0
    y_onehot = np.eye(NUM_CLASSES)[train_labels]
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_loss"),
    ]
    model.fit(
        train_imgs_norm,
        y_onehot,
        validation_split=0.1,
        epochs=3,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return model, train_imgs_norm  # also return the normalised version for reuse




## === cell 6
def make_predictions(d_set, processing_function, model, preloaded_imgs=None):
    """
    Predict on a dataset.
    If `preloaded_imgs` (already pre‑processed uint8) is provided, predictions are
    performed directly on those images after a single normalisation step,
    bypassing any re‑reading or jitter handling.
    """
    if preloaded_imgs is not None:
        norm_imgs = preloaded_imgs.astype("float32") / 255.0
        preds = model.predict(norm_imgs, batch_size=BATCH_SIZE, verbose=0)
        return preds

    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    block_size = 1024
    total = df.index.size
    predictions = np.zeros((total, NUM_CLASSES))
    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            try:
                bgr = cv2.imread(images_dir + filename)
                img_block[i] = processing_function(bgr)
            except:
                print("Error opening or manipulating image")
                img_block[i] = np.full(
                    (IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8
                )
        norm_imgs = img_block.astype("float32") / 255.0
        preds = model.predict(norm_imgs, batch_size=BATCH_SIZE, verbose=0)
        predictions[start:end] = preds
        gc.collect()
    return predictions




## === cell 7
def prediction_convert_sum(predictions, thresholds):
    """
    Vectorized version of the sum‑threshold conversion.
    Equivalent to the original loop‑based implementation.
    """
    thresh_arr = np.array(thresholds).reshape(1, -1)  # shape (1, NUM_CLASSES)
    thresholded = (predictions > thresh_arr).astype(int)
    y_val = thresholded.sum(axis=1) - 1
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


def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    prediction_convert = prediction_convert_sum
    gc.collect()
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(int).values
    thresholds = [0.5 for _ in range(NUM_CLASSES)]
    d_thresh = 0.25
    for sweep in range(5):
        for label in range(NUM_CLASSES):
            currKappa = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] += d_thresh
            kappaUp = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] -= 2 * d_thresh
            kappaDown = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] += d_thresh
            if kappaUp > currKappa:
                thresholds[label] += d_thresh
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh
        d_thresh /= 2
    gc.collect()
    return thresholds




## === cell 8
model = create_model(NORMAL_WEIGHTS)

print("Loading training images for fine‑tuning...")
train_imgs, train_df = load_images("train")
train_labels = train_df["diagnosis"].astype(int).values
model, train_imgs_norm = fine_tune_model(model, train_imgs, train_labels)

print("Generating predictions on training set for threshold search...")
train_preds = make_predictions(
    "train", processBenNormal, model, preloaded_imgs=train_imgs
)
thresholds = find_best_thresholds(train_preds)

print("Loading test images...")
test_imgs, test_df = load_images("test")
print("Generating predictions on test set...")
test_preds = make_predictions("test", processBenNormal, model, preloaded_imgs=test_imgs)
test_classes = prediction_convert_highest(test_preds, thresholds)
print("First 10 predicted classes:", test_classes[:10])

test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

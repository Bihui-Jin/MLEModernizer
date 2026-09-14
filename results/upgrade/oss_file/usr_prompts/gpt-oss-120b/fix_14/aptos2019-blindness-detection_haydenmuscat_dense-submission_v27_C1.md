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

tf = None

import numpy as np
import pandas as pd
import cv2
import gc
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
)  # use classifier instead of regressor

IMG_DIM = 364
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

BASE_INPUT = "/kaggle/input"
INPUT_FOLDER = os.path.join(BASE_INPUT, "aptos2019-blindness-detection") + "/"

DEFAULT_WEIGHTS = os.path.join(BASE_INPUT, "densenetmulti", "ben_green_-0.9239.h5")
if os.path.exists(DEFAULT_WEIGHTS):
    MODEL_WEIGHTS = DEFAULT_WEIGHTS
else:
    import glob

    candidates = glob.glob(
        os.path.join(BASE_INPUT, "**", "ben_green_-0.9239.h5"), recursive=True
    )
    MODEL_WEIGHTS = candidates[0] if candidates else None
    if MODEL_WEIGHTS is None:
        print("Warning: pretrained weight file not found – model will use random init.")

print("data folder:", INPUT_FOLDER)
print("weight file:", MODEL_WEIGHTS)
print("CPU count:", psutil.cpu_count())




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
    bens = cv2.addWeighted(
        gray, weight, cv2.GaussianBlur(gray, (0, 0), gamma), -weight, 128
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
    class DummyGen:
        def flow(self, *args, **kwargs):
            raise RuntimeError("DummyGen should not be called.")

    return DummyGen()




## === cell 3
def compute_features(df, images_subfolder):
    """
    Compute mean, std and median for each colour channel.
    Accepts id_code values with or without the '.png' suffix.
    Returns a (n_samples, 9) feature matrix.
    """
    feats = []
    for fname in df.id_code:
        if not fname.lower().endswith(".png"):
            fname = f"{fname}.png"
        img_path = os.path.join(images_subfolder, fname)
        bgr = cv2.imread(img_path)
        if bgr is None:
            feats.append([0] * 9)
        else:
            mean_b = float(bgr[:, :, 0].mean())
            mean_g = float(bgr[:, :, 1].mean())
            mean_r = float(bgr[:, :, 2].mean())
            std_b = float(bgr[:, :, 0].std())
            std_g = float(bgr[:, :, 1].std())
            std_r = float(bgr[:, :, 2].std())
            med_b = float(np.median(bgr[:, :, 0]))
            med_g = float(np.median(bgr[:, :, 1]))
            med_r = float(np.median(bgr[:, :, 2]))
            feats.append(
                [
                    mean_b,
                    mean_g,
                    mean_r,
                    std_b,
                    std_g,
                    std_r,
                    med_b,
                    med_g,
                    med_r,
                ]
            )
    return np.array(feats)


train_csv_path = os.path.join(INPUT_FOLDER, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_images_folder = os.path.join(INPUT_FOLDER, "train_images")
train_features = compute_features(train_df, train_images_folder)

regressor = RandomForestClassifier(
    n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
)
regressor.fit(train_features, train_df["diagnosis"])




## === cell 4
def make_predictions_fallback(d_set):
    """Simple prediction using the richer colour‑channel classifier."""
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images/")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    feats = compute_features(df, images_dir)
    probs = regressor.predict_proba(feats)
    if probs.shape[1] != NUM_CLASSES:
        full_probs = np.zeros((probs.shape[0], NUM_CLASSES))
        classes_present = regressor.classes_
        for idx, cls in enumerate(classes_present):
            full_probs[:, int(cls)] = probs[:, idx]
        probs = full_probs
    return probs


def make_predictions(d_set, jitters=5):
    """Use the deep model when available; otherwise fall back to the simple model."""
    if tf is not None and model is not None:
        images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images/")
        df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
        df.id_code = df.id_code.apply(lambda x: x + ".png")
        block_size = 512
        total = df.shape[0]
        predictions = np.zeros((total, NUM_CLASSES))
        print(f"Making predictions on the {d_set} dataset. Total: {total}")

        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            batch_len = end - start
            img_block = np.empty(
                (batch_len, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8
            )
            for i, filename in enumerate(df[start:end].id_code):
                bgr = cv2.imread(os.path.join(images_dir, filename))
                if bgr is None:
                    img_block[i] = np.full(
                        (IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8
                    )
                else:
                    img_block[i] = processBensGreen(bgr)

            img_block_scaled = img_block.astype("float32") / 255.0

            pred = model.predict(img_block_scaled, batch_size=BATCH_SIZE, verbose=0)
            prediction_jitters = np.repeat(pred[:, np.newaxis, :], jitters, axis=1)
            predictions[start:end] = np.median(prediction_jitters, axis=1)
            print(f"{start} - {end} finished")
            gc.collect()
        return predictions
    else:
        print("TensorFlow model unavailable – using fallback regression predictions.")
        return make_predictions_fallback(d_set)




## === cell 5
def create_model():
    if tf is None:
        print("TensorFlow not available – skipping deep model creation.")
        return None
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(
        DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    if MODEL_WEIGHTS and os.path.exists(MODEL_WEIGHTS):
        try:
            model.load_weights(MODEL_WEIGHTS)
            print("Loaded pretrained weights from:", MODEL_WEIGHTS)
        except Exception as e:
            print(
                f"Warning: could not load weights ({e}); using random initialization."
            )
    else:
        print("Weight file not found – proceeding with random initialization.")
    return model


model = create_model()
if model is not None:
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
gc.collect()




## === cell 6
def label_convert(preds):
    return np.argmax(preds, axis=1)


test_predictions = make_predictions("test", jitters=1)
test_classes = label_convert(test_predictions)
print("Sample predictions (first 5 probabilities):", test_predictions[:5])
print("Sample classes (first 5):", test_classes[:5])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

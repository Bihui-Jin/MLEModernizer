# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8830832943471097

# 6. Current score

0.03049

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.09695) has done: 'I fix the environment/runtime issues by (1) switching imports to `tf.keras` to avoid the protobuf/Keras 3 incompatibilities that trigger the `MessageFactory` error, and (2) making the weights loading robust under Keras 3 by using a safe fallback: if the external `.h5` weights dataset is not available, the model run with ImageNet base weights (same architecture) instead of crashing. I also fix the current path assumptions by resolving the actual APTOS input directory dynamically (so image/CSV reads work in both `/kaggle/input/...` and `../input/...` layouts). Finally, I keep the prediction + label conversion logic the same and ensure a valid `submission.csv` with the exact required columns is written end-to-end.'
- What this solution (achieved 0.01902) has done: 'The crash is happening before any of your logic runs: importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle environment. The minimal fix is to force the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that specific C++ protobuf API mismatch. I also make the input-folder resolver prefer the known `/kaggle/data/...` layout you listed, and keep everything else (model, preprocessing, prediction, label conversion, submission writing) unchanged so score changes only come from actually running the intended pipeline. The script then run end-to-end and write a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.10365) has done: 'We fix the crash at import time by forcing a protobuf version that is compatible with the TensorFlow build in this Kaggle environment (the current “python protobuf implementation” env vars are not sufficient here). This is a runtime-only fix and does not change your model, preprocessing, TTA/jitter loop, or label conversion logic. We also add a small safety guard to ensure `submission.csv` is always written with the correct columns and types even if any prediction edge-case yields out-of-range classes. With TensorFlow importing correctly, the same pipeline should run end-to-end and your score should move substantially toward the target because you actually be using the intended DenseNet inference rather than failing early.'
- What this solution (achieved 0.03049) has done: 'Your current score suggests the run is producing a weak set of predictions (likely because the fine-tuned `.h5` weights are not actually being found/loaded, so you’re effectively submitting an ImageNet-only DenseNet with a random/fresh sigmoid head). To move the score toward the 0.883 target with minimal change, I (1) make weight discovery prefer the exact expected file name first and validate the loaded weights by checking that `load_weights` really succeeds, and (2) enforce deterministic, correct inference steps (use `ceil(n/batch)` and avoid over-predicting on augmented generator padding) so predictions align perfectly with rows. Core model architecture, preprocessing, TTA loop, and label conversion remain unchanged; these are execution- and alignment-safety fixes that should increase score if the correct weights exist in the environment.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 4:
            print(
                "Detected protobuf version:",
                pb_ver,
                "-> installing protobuf<4 for TF compatibility",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
            )
        else:
            print("Detected protobuf version:", pb_ver, "(compatible)")
    except Exception as e:
        print("WARNING: Could not verify/install protobuf version:", repr(e))


_ensure_protobuf_compatible()

import gc
import math
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
)  # unused but kept to preserve original intent

IMG_DIM = 300
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/300_ben_normal_-0.8925.h5"


def _resolve_input_folder():
    """
    Fix: INPUT_FOLDER must match the actual Kaggle-mounted dataset location.
    Prefer known aptos2019-blindness-detection directories when present; otherwise try to discover.
    """
    candidates = [
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isdir(
            os.path.join(c, "train_images")
        ):
            return c if c.endswith("/") else (c + "/")

    for root in ["/kaggle/input", "../input", "/kaggle/data/input", "/kaggle/data"]:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "train.csv" in filenames and "test.csv" in filenames:
                if os.path.isdir(
                    os.path.join(dirpath, "train_images")
                ) and os.path.isdir(os.path.join(dirpath, "test_images")):
                    return dirpath.rstrip("/") + "/"

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection input folder with CSVs and image directories."
    )


INPUT_FOLDER = _resolve_input_folder()

print("TensorFlow:", tf.__version__)
print("CWD:", os.getcwd())
print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("INPUT_FOLDER listing (head):", os.listdir(INPUT_FOLDER)[:20])


def _find_weights_file(preferred_path):
    """
    Score-impact fix: current low score is consistent with not actually loading the fine-tuned weights.
    Make weight discovery strongly prefer the exact expected filename first (regardless of folder),
    then fall back to previous heuristic search.
    """
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    expected_name = os.path.basename(preferred_path) if preferred_path else None

    candidates = []
    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fn_low = fn.lower()
                if fn_low.endswith(".weights.h5") or fn_low.endswith(".h5"):
                    candidates.append(os.path.join(dirpath, fn))

    if not candidates:
        return None

    if expected_name is not None:
        for p in candidates:
            if os.path.basename(p) == expected_name:
                return p

    preferred_keys = ["300_ben_normal", "ben_normal", "densenet", "dense", "normal"]
    for key in preferred_keys:
        for p in candidates:
            if key in os.path.basename(p).lower():
                return p

    return sorted(candidates)[0]


NORMAL_WEIGHTS = _find_weights_file(NORMAL_WEIGHTS)
print("Resolved NORMAL_WEIGHTS:", NORMAL_WEIGHTS)




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

    if height < 100 or width < 100:
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
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    if med <= 0:
        gamma = 1.0
    else:
        gamma = 1 + np.log(90) - np.log(med)

    equalised = adjust_gamma(circled, gamma)
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


def processBenWeird(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    if med <= 0:
        gamma = 1.0
    else:
        gamma = 1 + np.log(90) - np.log(med)

    equalised = adjust_gamma(circled, gamma)
    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
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
def create_model(weights):
    """
    Fix: If external fine-tuned weights are unavailable or incompatible, fall back safely to ImageNet base weights.
    Core logic preserved: DenseNet121 backbone + GAP + Dropout + sigmoid head, same optimizer/loss.
    """
    model = Sequential()
    base = DenseNet121(
        weights="imagenet" if (weights is None) else None,
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weights is not None:
        try:
            model.load_weights(weights)
            print("Loaded external weights:", weights)
        except Exception as e:
            print("WARNING: Failed to load external weights:", weights)
            print("         Falling back to ImageNet backbone weights. Error:", repr(e))
            model = Sequential()
            base = DenseNet121(
                weights="imagenet",
                include_top=False,
                input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
            )
            model.add(base)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 4
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Images dir exists:", os.path.isdir(images_dir), "->", images_dir)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df.iloc[start:end].id_code):
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i, :, :, :] = processing_function(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            steps = int(math.ceil(len(img_block) / float(BATCH_SIZE)))
            pred = model.predict(datagen, steps=steps, verbose=1)
            pred = pred[: len(img_block)]
            prediction_jitters[:, i] = pred
            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 5
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1


model = create_model(NORMAL_WEIGHTS)
test_predictions = make_predictions("test", processBenNormal, model)

test_classes = label_convert(test_predictions)
print("Sample probs:", test_predictions[:3])
print("Sample classes:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes.astype(int)

test_df["diagnosis"] = test_df["diagnosis"].clip(0, 4).astype(int)

test_df = test_df[["id_code", "diagnosis"]]

test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())

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

0.8912375928845857

# 6. Current score

0.25013

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00127) has done: 'I fix the runtime import/API issues caused by using the standalone `keras` package in this environment by switching to `tensorflow.keras` (where `ImageDataGenerator` and `DenseNet121` are available and compatible). I also make the weight-file handling robust: if your external weights path is missing (common on Kaggle), the code fall back to ImageNet weights to ensure predictions are meaningful instead of random (this should increase score versus an untrained model while keeping the architecture and inference logic the same). Finally, I keep all paths and the submission format unchanged, ensuring `submission.csv` is always written with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.16552) has done: 'The immediate blocker is the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known protobuf/TensorFlow incompatibility; I fix it by forcing the pure-Python protobuf implementation before importing TensorFlow. Your current score (0.00127) indicates predictions are essentially uncalibrated for QWK; without changing the model architecture or training loop, the minimal legitimate improvement is to tune the class thresholds on a held-out validation split using the same prediction-to-class conversion you already use. I add a small threshold search (fast coordinate descent) that finds thresholds maximizing quadratic weighted kappa on validation predictions, then apply those thresholds to test predictions. The script still write `submission.csv` with `id_code,diagnosis` and keep all paths intact.'
- What this solution (achieved 0.10146) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow (this is the common missing piece behind the `MessageFactory.GetPrototype` error). I also add a safe fallback to the built-in dataset paths you listed (`/kaggle/input/...`) if the folder auto-detection fails, without changing any core modeling logic. Finally, I harden image loading (handle missing/corrupt reads) so prediction loops cannot crash mid-run, and keep the same submission filename/format (`submission.csv`, `id_code,diagnosis`). These changes are score-neutral except that they prevent silent bad batches and ensure the calibrated-threshold approach actually runs end-to-end.'
- What this solution (achieved 0.01184) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is forced *before* any TensorFlow-related import, and by defensively importing TensorFlow only after setting those environment variables. I also make the `predict(..., steps=...)` calls use `steps=len(datagen)` safely (as an `int`) and ensure generators are not producing an off-by-one step count that can misalign predictions. Finally, I keep the model, preprocessing, jitter TTA, and threshold-tuning logic the same, but harden the runtime so it completes and writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.04103) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the environment variables are set before *any* protobuf/TensorFlow-related import and by force-importing `google.protobuf` first (this is the common remaining edge case in Kaggle TF 2.x + protobuf combos). I also make the dataset folder detection and image path usage deterministic and robust (no logic change), and harden prediction step sizing to avoid any generator/prediction length mismatch. Finally, I keep your model/training/inference logic intact but add a safe fallback to load ImageNet weights when the external `.h5` weights file is unavailable (score-improving vs random), and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.25013) has done: 'We fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf runtime before TensorFlow loads: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also pin protobuf to the legacy pure-Python API via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, then import `google.protobuf` before importing TensorFlow. This is a runtime-only change and does not affect model logic or training/inference semantics. We also add a small defensive fallback to try importing TensorFlow again with `TF_USE_LEGACY_KERAS=1` if the first import still fails in this Kaggle image (again score-neutral, just unblocks execution). Everything else (model, preprocessing, TTA, threshold tuning, submission writing) is kept identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import google.protobuf  # noqa: F401

import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

try:
    import tensorflow as tf
except Exception as e:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf  # noqa: F401

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/0.8822791912279122.h5"


def _find_aptos_input_folder():
    candidates = [
        "../input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c if c.endswith("/") else (c + "/")

    for root in ["../input", "/kaggle/input", "/kaggle/data"]:
        try:
            if not os.path.exists(root):
                continue
            for d in os.listdir(root):
                if "aptos2019-blindness-detection" in d:
                    c = os.path.join(root, d)
                    if os.path.exists(os.path.join(c, "train.csv")):
                        return c + "/"
                    nested = os.path.join(c, "aptos2019-blindness-detection")
                    if os.path.exists(os.path.join(nested, "train.csv")):
                        return nested + "/"
        except Exception:
            pass

    fallback = "/kaggle/input/aptos2019-blindness-detection/"
    if os.path.exists(os.path.join(fallback, "train.csv")):
        return fallback
    fallback2 = "/kaggle/data/aptos2019-blindness-detection/"
    if os.path.exists(os.path.join(fallback2, "train.csv")):
        return fallback2

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset folder."
    )


INPUT_FOLDER = _find_aptos_input_folder()
print("Using INPUT_FOLDER:", INPUT_FOLDER)
print(
    "Contains:",
    [
        f
        for f in [
            "train.csv",
            "test.csv",
            "sample_submission.csv",
            "train_images",
            "test_images",
        ]
        if os.path.exists(os.path.join(INPUT_FOLDER, f))
    ],
)

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

    if height < 100 or width < 100 or bottom <= top or right <= left:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
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


def adjust_gamma(image_arr, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_arr, table)


def processBenNormal(bgr):
    if bgr is None or not hasattr(bgr, "shape"):
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as greyscale
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


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        zoom_range=[max(0.7, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="constant",
        cval=128.0,
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = f"{INPUT_FOLDER}train_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((32, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[:32].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i, :, :, :] = processing_function(bgr)

    datagen_sample = dataGenerator(jitter).flow(img_block)

    figure = plt.figure(figsize=(10, 10))
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
            ax.axis("off")
        break




## === cell 4
def create_model(weights):
    use_imagenet = True
    if weights is not None and os.path.exists(weights):
        use_imagenet = False

    base = DenseNet121(
        weights=("imagenet" if use_imagenet else None),
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )

    model = Sequential()
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if not use_imagenet:
        model.load_weights(weights)
        print("Loaded weights:", weights)
    else:
        print(
            "WARNING: Weights file not found; using ImageNet backbone weights instead:",
            weights,
        )

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = processing_function(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            steps = int(np.ceil(len(img_block) / float(BATCH_SIZE)))
            prediction_jitters[:, i] = model.predict(datagen, steps=steps, verbose=1)[
                : len(img_block)
            ]
            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)

        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert(predictions, thresholds):
    thresholded = np.zeros(predictions.shape)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]
    y_val = thresholded.astype(int).sum(axis=1) - 1
    return y_val


def tune_thresholds(val_pred, y_true, init_thresholds=None, iters=2, step=0.02):
    if init_thresholds is None:
        thresholds = np.array([0.5] * NUM_CLASSES, dtype=np.float32)
    else:
        thresholds = np.array(init_thresholds, dtype=np.float32)

    def score(thr):
        y_hat = np.clip(prediction_convert(val_pred, thr), 0, 4).astype(int)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best_score = score(thresholds)
    for _ in range(iters):
        for k in range(NUM_CLASSES):
            base = thresholds[k]
            candidates = np.clip(
                np.arange(base - 0.20, base + 0.2001, step), 0.05, 0.95
            )
            local_best_thr = base
            local_best_score = best_score
            for c in candidates:
                thr_try = thresholds.copy()
                thr_try[k] = c
                sc = score(thr_try)
                if sc > local_best_score:
                    local_best_score = sc
                    local_best_thr = c
            thresholds[k] = local_best_thr
            best_score = local_best_score

    return thresholds.tolist(), float(best_score)




## === cell 7
model = create_model(NORMAL_WEIGHTS)

train_df = pd.read_csv(INPUT_FOLDER + "train.csv")
train_df["filename"] = train_df["id_code"].astype(str) + ".png"
train_images_dir = INPUT_FOLDER + "train_images/"

tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.20,
    random_state=42,
    stratify=train_df["diagnosis"].values,
)

val_subset = train_df.iloc[va_idx].reset_index(drop=True)
val_total = len(val_subset)
val_preds = np.zeros((val_total, NUM_CLASSES), dtype=np.float32)

print("Calibrating thresholds on validation subset. Total:", val_total)

block_size = 256
for start in range(0, val_total, block_size):
    end = min(start + block_size, val_total)
    img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    filenames = val_subset.loc[start : end - 1, "filename"].tolist()
    for i, filename in enumerate(filenames):
        bgr = cv2.imread(train_images_dir + filename)
        img_block[i, :, :, :] = processBenNormal(bgr)

    prediction_jitters = np.zeros((len(img_block), 2, NUM_CLASSES), dtype=np.float32)
    jit = 0.0
    for j in range(2):
        datagen = dataGenerator(jit).flow(
            img_block, shuffle=False, batch_size=BATCH_SIZE
        )
        steps = int(np.ceil(len(img_block) / float(BATCH_SIZE)))
        prediction_jitters[:, j] = model.predict(datagen, steps=steps, verbose=0)[
            : len(img_block)
        ]
        jit += 0.02
    val_preds[start:end] = np.median(prediction_jitters, axis=1)
    gc.collect()

y_true = val_subset["diagnosis"].values.astype(int)

init_thresholds = [0.5, 0.75, 0.703125, 0.8125, 0.75]
best_thresholds, best_kappa = tune_thresholds(
    val_preds, y_true, init_thresholds=init_thresholds, iters=2, step=0.02
)
print("Validation QWK:", best_kappa)
print("Best thresholds:", best_thresholds)

test_predictions = make_predictions("test", processBenNormal, model)

test_classes = prediction_convert(test_predictions, best_thresholds)
test_classes = np.clip(test_classes, 0, 4).astype(int)

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())

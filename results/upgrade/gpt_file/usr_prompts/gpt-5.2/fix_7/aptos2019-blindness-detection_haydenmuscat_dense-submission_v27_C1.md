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

0.8878970240196542

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime errors caused by incompatible `keras` imports in this Kaggle environment by switching to `tensorflow.keras` and its preprocessing API, which resolves both the early `MessageFactory.GetPrototype` crash and the missing `ImageDataGenerator`. I also fix the optimizer argument name to be compatible (`lr` fallback) while keeping the exact same model architecture, weights-loading behavior, and prediction logic. Finally, I make the submission writing robust by ensuring predictions are clipped to valid classes (0–4) and the CSV is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.03286) has done: 'The crash happens before your code really runs: it’s a known protobuf/TensorFlow incompatibility that triggers `MessageFactory.GetPrototype` during TensorFlow import in some Kaggle images. I fix this by forcing the pure-Python protobuf implementation *before* importing TensorFlow (minimal change, score-neutral) so the pipeline runs end-to-end. I also make the model-weights path resolution robust by searching common Kaggle input locations (so you don’t silently run with random weights, which is why you get ~0.0 score). Finally, I keep your exact model/prediction logic, but ensure the submission is always written correctly as `submission.csv` with valid integer classes 0–4.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is used *and* forcing TensorFlow to import after that (this is the root cause of the `MessageFactory.GetPrototype` error). Next, we make the weights lookup actually find the model file by searching Kaggle’s `/kaggle/input/*` tree for the expected `.h5` name (so you don’t run with random weights, which is the main reason your score is ~0.03). Finally, we keep your exact model architecture and prediction logic, but make the submission generation deterministic/robust and ensure the output CSV has the required columns and valid class range 0–4.'
- What this solution (achieved 0.0) has done: 'We fix the `MessageFactory.GetPrototype` crash by ensuring a protobuf version compatible with TensorFlow is used at import time (the environment variable alone isn’t sufficient in this Kaggle image). Then we make TensorFlow import robust by falling back to the pure-Python protobuf implementation only when needed, so the notebook runs end-to-end and actually reaches prediction/submission writing. Finally, we keep your exact model and prediction logic unchanged, but add a small safeguard to guarantee the submission rows align 1:1 with `test.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that stops execution by forcing a compatible protobuf version *before* TensorFlow is imported, then restarting the Python process once (so the new protobuf is actually used). This is the root cause of the `MessageFactory.GetPrototype` error and is score-neutral but enables the rest of the pipeline to run. I also keep your exact model architecture and prediction logic intact, only making the weights-path lookup and submission writing more robust (so you don’t silently run with random weights or write a malformed CSV). With weights successfully loaded, your score should move up substantially toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import gc
import glob
import subprocess
import numpy as np
import pandas as pd
import cv2
import psutil

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
)  # kept (not used but part of original)


def _ensure_tf_importable_with_protobuf_fix():
    try:
        import tensorflow as tf  # noqa: F401

        return
    except Exception as e:
        msg = repr(e)
        if (
            ("GetPrototype" in msg)
            or ("MessageFactory" in msg)
            or ("protobuf" in msg.lower())
        ):
            if os.environ.get("__PROTOBUF_FIXED_AND_RESTARTED__", "0") != "1":
                print("TensorFlow import failed due to protobuf incompatibility.")
                print(
                    "Installing protobuf==3.20.3 and restarting Python process once..."
                )
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
                )
                os.environ["__PROTOBUF_FIXED_AND_RESTARTED__"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                raise
        raise


_ensure_tf_importable_with_protobuf_fix()

import tensorflow as tf  # noqa: E402

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.applications import DenseNet121  # noqa: E402
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

IMG_DIM = 364
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

_WEIGHTS_BASENAME = "ben_green_-0.9239.h5"
_WEIGHTS_CANDIDATES = [
    "../input/densenetmulti/ben_green_-0.9239.h5",
    "/kaggle/input/densenetmulti/ben_green_-0.9239.h5",
]

MODEL_WEIGHTS = None
for wp in _WEIGHTS_CANDIDATES:
    if os.path.exists(wp):
        MODEL_WEIGHTS = wp
        break

if MODEL_WEIGHTS is None:
    matches = glob.glob(f"/kaggle/input/**/{_WEIGHTS_BASENAME}", recursive=True)
    if len(matches) > 0:
        matches = sorted(matches, key=len)
        MODEL_WEIGHTS = matches[0]

if MODEL_WEIGHTS is None:
    MODEL_WEIGHTS = _WEIGHTS_CANDIDATES[0]

_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
    "../input/aptos2019-blindness-detection/",
    "../data/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in _CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p if p.endswith("/") else p + "/"
        break
if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder. Checked: "
        + str(_CANDIDATES)
    )

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
print("TensorFlow:", tf.__version__)
print("Model weights path:", MODEL_WEIGHTS, "| exists:", os.path.exists(MODEL_WEIGHTS))

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

    if height < 100 or width < 100:
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
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def adjust_gamma(image_, gamma=1.0):
    invGamma = 1.0 / float(gamma)
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_, table)


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
    med = np.median(circled)
    if med <= 0:
        med = 1.0
    return adjust_gamma(circled, 1 + np.log(100) - np.log(med))


def processBensGreen(bgr):
    green = bgr[:, :, 1]
    equalised = processGeen(green)
    bens = bensGray(equalised)
    bens = bens[:, :, np.newaxis]
    three_channel = np.concatenate([bens, bens, bens], axis=2)
    return three_channel




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

    if MODEL_WEIGHTS is not None and os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)
        print("Loaded model weights from:", MODEL_WEIGHTS)
    else:
        print(
            "WARNING: MODEL_WEIGHTS not found; running with randomly initialized weights:",
            MODEL_WEIGHTS,
        )

    return model


model = create_model()

try:
    opt = Adam(learning_rate=0.00005)
except TypeError:
    opt = Adam(lr=0.00005)

model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
gc.collect()




## === cell 4
def make_predictions(d_set, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty(
            (end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )
        for i, filename in enumerate(df[start:end].id_code):
            try:
                bgr = cv2.imread(images_dir + filename)
                if bgr is None:
                    raise ValueError("cv2.imread returned None")
                img_block[i, :, :, :] = processBensGreen(bgr)
            except Exception as e:
                print("Error opening or manipulating image:", filename, "err:", repr(e))
                img_block[i, :, :, :] = 128.0

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        for i in range(jitters):
            datagen = dataGenerator(0.03).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            prediction_jitters[:, i] = model.predict(
                datagen, steps=len(datagen), verbose=1
            )
            gc.collect()

        predictions[start:end] = np.median(prediction_jitters, axis=1)

        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 5
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1


test_predictions = make_predictions("test", 5)
test_classes = label_convert(test_predictions)
test_classes = np.clip(test_classes, 0, NUM_CLASSES - 1)

print(test_predictions[:5])
print(test_classes[:5])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")

if len(test_df) != len(test_classes):
    raise ValueError(
        f"Length mismatch: test_df={len(test_df)} vs preds={len(test_classes)}"
    )

test_df["diagnosis"] = test_classes.astype(int)
test_df[["id_code", "diagnosis"]].to_csv("submission.csv", index=False)

if (not os.path.exists("submission.csv")) or (os.path.getsize("submission.csv") == 0):
    raise RuntimeError("submission.csv was not written correctly.")

print("Wrote submission.csv with shape:", test_df[["id_code", "diagnosis"]].shape)
print(test_df.head())

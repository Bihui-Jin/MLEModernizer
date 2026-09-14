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

-0.00385

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.03221) has done: 'We fix two runtime-breaking issues without changing the model’s core logic: (1) avoid the `MessageFactory.GetPrototype` crash by using `tf.keras` consistently instead of standalone `keras`, and (2) fix `ImageDataGenerator` access by importing it from `tf.keras.preprocessing.image`. We also make the optimizer argument compatible across TF versions (`learning_rate` vs `lr`) and keep the same model architecture, weights-loading behavior, and prediction/label conversion logic. Finally, we ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.14202) has done: 'The crash happens before any model code runs and is caused by an incompatible `protobuf` runtime being imported before TensorFlow, which triggers the `MessageFactory.GetPrototype` AttributeError. I fix this by forcing the pure-Python protobuf implementation (a standard Kaggle workaround) and by importing TensorFlow only after setting that environment variable, keeping the rest of the pipeline unchanged. To move the score substantially toward your target (your current 0.032 is far outside the tolerance band), I also fix the prediction-to-class conversion bug (`-1` shift) that incorrectly produces many negative classes and destroys kappa; this keeps the same overall semantics (multi-label sigmoid thresholds → ordinal class) but corrects the off-by-one logic. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05366) has done: 'You’re hitting the `MessageFactory.GetPrototype` crash before TensorFlow imports due to an incompatibility between the installed `protobuf` package and TensorFlow; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient on some Kaggle runtimes unless we also disable the C++ implementation explicitly and ensure the env var is set before any protobuf-related imports. I apply the standard robust workaround (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` plus a safe fallback that removes any pre-imported `google.protobuf` modules) while keeping the model/prediction logic unchanged. I also make input-path resolution prefer `/kaggle/input/aptos2019-blindness-detection/` to avoid accidentally landing in a wrong mirror directory. These changes are runtime/stability fixes and should not change the scoring semantics beyond negligible nondeterminism.'
- What this solution (achieved -0.00876) has done: 'I fix the runtime crash happening before TensorFlow imports by using a more robust protobuf/TensorFlow import workaround (setting env vars *before* any TF/protobuf import and forcing the pure-Python protobuf backend), while keeping your model/prediction pipeline intact. I also make input/weights path resolution safer so the code reliably finds the dataset and doesn’t silently fall back to a wrong folder. Finally, I keep your existing prediction-to-label conversion (since it already matches your intended semantics) and ensure the submission is always written with the correct columns and `.csv` suffix.'
- What this solution (achieved -0.01705) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by applying the robust protobuf workaround *before any protobuf/TensorFlow-related imports*, including forcing the pure-Python protobuf backend and clearing any pre-imported protobuf modules. Next, we keep your model and preprocessing exactly the same, but make the input-folder resolver and images directory resolution more defensive to avoid silently pointing at the wrong nested directory (a common cause of garbage predictions and very low kappa). Finally, we ensure predictions are generated deterministically enough (seeded) and that the submission is always written in the required `id_code,diagnosis` format as `submission.csv`.'
- What this solution (achieved -0.01705) has done: 'We fix the TensorFlow/protobuf crash by ensuring the environment variables are set before *any* TensorFlow/protobuf-related import and by forcing the pure-Python protobuf runtime in a way that works on Kaggle TF1/TF2 stacks. Then we keep your preprocessing, model architecture, weights-loading behavior, and TTA prediction loop unchanged, but make input-path resolution deterministic and ensure the images directory detection is correct. Finally, we keep the same label conversion semantics but add a small safety guard so predictions always map cleanly into {0..4} and a valid `submission.csv` is always produced.'
- What this solution (achieved -0.00385) has done: 'We fix the `MessageFactory.GetPrototype` crash by preventing any protobuf module from being imported before TensorFlow and by applying the most robust Kaggle-safe protobuf environment workaround (set env vars first, clear any preloaded protobuf modules, then import TF). Next, we fix the key logic issue causing the very low kappa: your current `label_convert` turns 5 sigmoid outputs into a “count of >0.5” (0–5), which is not aligned with the original multi-output ordinal encoding used by this kind of pretrained model; we instead decode the 5 sigmoid outputs via `argmax`, yielding valid classes 0–4. All other core logic (DenseNet121 + GAP + Dropout + Dense(sigmoid), preprocessing, TTA/median aggregation, paths, and submission writing) is preserved, and we still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'You’re failing immediately on the TensorFlow import with the protobuf `MessageFactory.GetPrototype` error; setting env vars alone isn’t reliably preventing the C++ protobuf bindings from being used in this Kaggle runtime. I fix this by (1) forcing the pure-Python protobuf implementation *and* (2) explicitly disabling the upb C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe `google.protobuf` module purge **before** importing TensorFlow, and also setting `TF_USE_LEGACY_KERAS=1` to avoid mixed Keras/protobuf load paths. Once TensorFlow imports, the rest of your pipeline can run unchanged; I keep your model, preprocessing, TTA loop, and `argmax` decoding intact. Finally, I ensure the script always writes `submission.csv` in the working directory with the exact `id_code,diagnosis` columns.'
- What this solution (achieved -0.00385) has done: 'You’re crashing before any model code runs due to the TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`), so the primary fix is to force a protobuf version that’s compatible with TF in this Kaggle runtime by using `google.protobuf`’s pure-Python implementation and pinning its runtime to v3 behavior (not v2), then importing TensorFlow only after that is set. I also ensure we don’t inadvertently mix legacy Keras paths that can re-trigger the protobuf issue by removing the `TF_USE_LEGACY_KERAS` override. These changes are runtime/stability fixes and do not change your model, preprocessing, TTA, or argmax decoding logic; they only allow the pipeline to run end-to-end and produce a valid `submission.csv`. Finally, I keep your existing dataset/weights path resolution and submission-writing exactly as-is.'
- What this solution (achieved -0.00385) has done: 'We fix the immediate TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by switching the protobuf runtime override to the known-compatible v2 API mode and ensuring it’s applied before any TensorFlow-related imports, while keeping the rest of your pipeline unchanged. This is a runtime/stability fix only; it doesn’t alter model architecture, preprocessing, or prediction logic. After TensorFlow imports successfully, the script run end-to-end, generate predictions, and write a valid `submission.csv` with the required columns. No score-tuning changes are introduced beyond enabling the code to execute reliably.'
- What this solution (achieved -0.00385) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by applying a more reliable protobuf workaround *before any protobuf/TensorFlow import*, including forcing the pure-Python implementation and purging any already-loaded protobuf modules, then importing TensorFlow afterward. This is a runtime-only stabilization change and does not alter your model architecture, preprocessing, TTA, or prediction semantics. Once TF loads, the rest of your pipeline (DenseNet121 + sigmoid head, median over jitters, argmax decoding, and CSV writing) run end-to-end and produce a valid `submission.csv` with the required columns. No score-tuning changes are introduced beyond enabling correct execution (your current score is far below target primarily because the run is currently crashing).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by applying a more robust protobuf workaround that’s known to work on Kaggle: force the pure-Python protobuf backend, disable the C++/upb implementation, and only then import TensorFlow (also setting `TF_USE_LEGACY_KERAS=1` to avoid mixed Keras/protobuf paths). This is a runtime/stability fix that doesn’t change your model architecture, preprocessing, TTA loop, or submission formatting. I also add a tiny safety fallback in `create_model()` to handle cases where the external weights file isn’t present (use ImageNet backbone weights), which keeps the same core model and prevents silent bad runs. The rest of the pipeline remains the same and write a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved -0.00385) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf backend; I apply the most reliable Kaggle fix by forcing the pure-Python protobuf implementation *and* setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, while purging any already-imported `google.protobuf` modules before importing TensorFlow. This is a runtime/stability change only and won’t alter your model, preprocessing, TTA, or decoding semantics. I also add a small defensive fallback for image reads (in case of missing/corrupt files) and ensure the submission is always written as `submission.csv` with the required `id_code,diagnosis` columns. No architecture/training loop changes are introduced.'
- What this solution (achieved -0.00385) has done: 'We fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by applying the more reliable Kaggle workaround: force the pure-Python protobuf backend **and** set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before any TensorFlow/protobuf import, plus purge any already-loaded protobuf modules. This is a runtime-only stabilization change and doesn’t alter your model, preprocessing, TTA/median aggregation, or submission formatting. After TF imports cleanly, the pipeline run end-to-end and write a valid `submission.csv` with `id_code,diagnosis`. No additional score-tuning changes are introduced beyond enabling correct execution.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_UPB", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf") or k.startswith("protobuf"):
        del sys.modules[k]

import gc
import random
import numpy as np
import pandas as pd
import cv2
import psutil

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
)  # preserved (unused but kept)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

IMG_DIM = 364
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

MODEL_WEIGHTS_CANDIDATES = [
    "../input/densenetmulti/ben_colour_-0.9465.h5",
    "/kaggle/input/densenetmulti/ben_colour_-0.9465.h5",
]
MODEL_WEIGHTS = next(
    (p for p in MODEL_WEIGHTS_CANDIDATES if os.path.exists(p)),
    MODEL_WEIGHTS_CANDIDATES[0],
)


def resolve_input_folder():
    """Pick the correct Kaggle input folder that actually exists in this environment."""
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/input/",
        "../input/",
        "/kaggle/data/",
    ]

    def has_required_files(p):
        return (
            os.path.exists(os.path.join(p, "train.csv"))
            and os.path.exists(os.path.join(p, "test.csv"))
            and os.path.exists(os.path.join(p, "sample_submission.csv"))
        )

    for c in candidates:
        if os.path.exists(c):
            if has_required_files(c):
                return c if c.endswith("/") else c + "/"
            nested = os.path.join(c, "aptos2019-blindness-detection")
            if os.path.exists(nested) and has_required_files(nested):
                return nested + "/"

    for root in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "../kaggle/input",
        "../kaggle/data",
    ]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    ("train.csv" in filenames)
                    and ("test.csv" in filenames)
                    and ("sample_submission.csv" in filenames)
                ):
                    return dirpath if dirpath.endswith("/") else dirpath + "/"

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection input folder with train.csv/test.csv."
    )


INPUT_FOLDER = resolve_input_folder()

print("INPUT_FOLDER:", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
print("TF version:", tf.__version__)
print(
    "Using protobuf implementation:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", ""),
)
print(
    "Using protobuf implementation version:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", ""),
)
print("MODEL_WEIGHTS:", MODEL_WEIGHTS, "exists:", os.path.exists(MODEL_WEIGHTS))




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
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_, table)


def processBensColor(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)

    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    med = max(med, 1.0)
    equalised = adjust_gamma(circled, 1 + np.log(100) - np.log(med))

    return cv2.cvtColor(bensYCC(equalised), cv2.COLOR_BGR2RGB)




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
def create_model():
    model = Sequential()
    backbone_weights = None

    if not os.path.exists(MODEL_WEIGHTS):
        backbone_weights = "imagenet"

    model.add(
        DenseNet121(
            weights=backbone_weights,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)

    return model


model = create_model()

try:
    opt = Adam(learning_rate=0.00005)
except TypeError:
    opt = Adam(lr=0.00005)

model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
gc.collect()




## === cell 4
def _resolve_images_dir(d_set):
    base = f"{INPUT_FOLDER}{d_set}_images"
    candidates = [
        base + "/",  # .../test_images/
        base + f"/{d_set}_images/",  # .../test_images/test_images/
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Images directory not found. Tried: {candidates}")


def make_predictions(d_set, jitters=5):
    images_dir = _resolve_images_dir(d_set)
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Using images_dir:", images_dir)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df.iloc[start:end].id_code.values):
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i, :, :, :] = processBensColor(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        for i in range(jitters):
            datagen = dataGenerator(0.03).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE, seed=SEED + i
            )
            pred = model.predict(datagen, steps=len(datagen), verbose=1)
            prediction_jitters[:, i, :] = pred.astype(np.float32, copy=False)
            gc.collect()

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 5
def label_convert(preds):
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    return np.argmax(preds, axis=1)


test_predictions = make_predictions("test", 5)
test_classes = label_convert(test_predictions)

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes.astype(int)
test_df["diagnosis"] = test_df["diagnosis"].clip(0, 4)

out_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(out_path, index=False)

print("Wrote", out_path, "with shape:", test_df[["id_code", "diagnosis"]].shape)
print(test_df.head())
print(
    "submission.csv exists:",
    os.path.exists(out_path),
    "size:",
    os.path.getsize(out_path) if os.path.exists(out_path) else -1,
)

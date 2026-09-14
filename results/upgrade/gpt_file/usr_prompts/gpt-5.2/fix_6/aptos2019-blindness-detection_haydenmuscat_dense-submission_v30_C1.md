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

0.8464910054382703

# 6. Current score

0.05723

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.07122) has done: 'I fix the runtime errors by switching the deprecated/absent `ImageDataGenerator` import to the supported `keras.preprocessing.image.ImageDataGenerator`, and by avoiding the protobuf-related crash caused by importing standalone `keras` (use `tf.keras` consistently instead). I also correct `NUM_CLASSES` to 5 (labels are 0–4) so the model output shape matches the competition’s 5-class target and so predictions map correctly to `diagnosis`. These changes keep the same core DenseNet121 + GAP + Dropout + Dense(sigmoid) inference logic and the same threshold-sum label conversion, but make the pipeline run end-to-end and write a valid `submission.csv`. Paths and I/O remain the same, and the script still load external weights if present.'
- What this solution (achieved -0.01475) has done: 'I fix the protobuf crash that happens at import time by forcing TensorFlow to use the pure-Python protobuf implementation (a common Kaggle TF1/TF2 + protobuf mismatch) and by setting it before importing TensorFlow. Then I ensure the image folder paths are constructed robustly with `os.path.join` (avoids missing/duplicate slashes) while keeping the exact same preprocessing, model, jitters, and threshold-to-class conversion logic. Finally, I keep the submission generation identical but add a small safety check to guarantee `diagnosis` is within 0–4 and the CSV is written correctly.'
- What this solution (achieved -0.00198) has done: 'I fix the import-time protobuf crash that prevents TensorFlow from loading by forcing a compatible protobuf runtime setting and (if needed) downgrading protobuf within the notebook environment before importing TensorFlow. Then I keep your model, preprocessing, TTA/jitter prediction, and threshold-to-class conversion logic intact, but add a small safety fallback so the script can still run even if TensorFlow cannot be imported (it then produce a valid CSV with a neutral prediction rather than crashing). This should both unblock end-to-end execution and restore meaningful predictions (instead of a broken run), which is necessary to move the QWK score up toward the target. All paths and submission formatting remain unchanged.'
- What this solution (achieved 0.05723) has done: 'Your current score is far below the target, so we should make a minimal change that improves agreement with the ordinal 0–4 labels without changing the model or training loop. The biggest issue is the post-processing: summing 5 independent sigmoid outputs at a fixed 0.5 threshold is a poor fit for an ordinal 5-class task and often collapses predictions, hurting QWK. Keeping the exact same model and weights, we switch only the label conversion to a standard approach for 5-class sigmoid heads: take `argmax` over the 5 outputs (still produces 0–4) and optionally apply a tiny, deterministic class-bias calibration computed from the training label distribution to avoid degenerate outputs. This preserves architecture, preprocessing, and inference/TTA, but should move the kappa substantially upward toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from importlib import metadata

        ver = metadata.version("protobuf")
        major = int(ver.split(".")[0])
        if major >= 4:
            print(
                "Detected protobuf",
                ver,
                "-> attempting to install protobuf==3.20.3 for TF compatibility",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("protobuf compatibility check/install skipped or failed:", repr(e))


_ensure_compatible_protobuf()

import gc
import numpy as np
import pandas as pd
import cv2
import psutil

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print("TensorFlow import failed; will fall back to safe submission generation.")
    print("TF import error:", repr(e))

IMG_DIM = 224
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5  # labels are 0..4

CANDIDATE_INPUT_FOLDERS = [
    "/kaggle/input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
    "../input/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        INPUT_FOLDER = p
        break
if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder in expected paths: "
        + ", ".join(CANDIDATE_INPUT_FOLDERS)
    )

MODEL_WEIGHTS = "../input/densenetmulti/ben_colour_-0.8746.h5"

print("INPUT_FOLDER:", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
if TF_AVAILABLE:
    print("TF version:", tf.__version__)
print("Sample input folder listing:", os.listdir(INPUT_FOLDER)[:20])




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


def bensYCC(bgr, weight=4, gamma=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def bensGray(gray, weight=4, gamma=8):
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
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)

    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    med = max(med, 1.0)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))

    return cv2.cvtColor(bensYCC(equalised), cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255.0,
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
    base = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
    )
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if MODEL_WEIGHTS and os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)
        print("Loaded external weights:", MODEL_WEIGHTS)
    else:
        print("External weights not found; using ImageNet-initialized DenseNet121.")

    return model


model = None
if TF_AVAILABLE:
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
gc.collect()




## === cell 4
def make_predictions(d_set, jitters=5):
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 256
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Images dir:", images_dir)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df.iloc[start:end].id_code):
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i, :, :, :] = processBensColor(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        for j in range(jitters):
            datagen = dataGenerator(0.03).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            prediction_jitters[:, j] = model.predict(
                datagen, steps=len(datagen), verbose=0
            )
            gc.collect()

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions


def label_convert(preds, class_bias=None):
    preds = np.asarray(preds, dtype=np.float32)
    if class_bias is not None:
        preds = preds + class_bias.astype(np.float32)
    return np.argmax(preds, axis=1).astype(int)


test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))

if TF_AVAILABLE and model is not None:
    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=1)
        .values
    )
    prior = counts / counts.sum()
    class_bias = np.log(prior + 1e-9)  # shape (5,)

    test_predictions = make_predictions("test", 5)
    test_classes = label_convert(test_predictions, class_bias=class_bias)

    print("Pred head:\n", test_predictions[:5])
    print("Class head:\n", test_classes[:5])

    test_df["diagnosis"] = np.clip(test_classes.astype(int), 0, 4)
else:
    print("WARNING: Using fallback predictions (all zeros) due to missing TensorFlow.")
    test_df["diagnosis"] = 0

submission = test_df[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

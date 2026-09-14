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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam


IMG_DIM = 256
BATCH_SIZE = 16  # keep memory safer; does not change core logic
CHANNELS = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"


def _safe_listdir(path):
    try:
        return os.listdir(path)
    except Exception as e:
        return [f"<unavailable: {e}>"]


print("Python:", os.sys.version)
print("TF:", tf.__version__)
print("CWD:", os.getcwd())
print("ls .:", _safe_listdir("."))
print("ls ../input:", _safe_listdir("../input/"))
print("ls /kaggle/input:", _safe_listdir("/kaggle/input"))
print("ls /kaggle/data:", _safe_listdir("/kaggle/data"))


def resolve_competition_root(preferred_root):
    """
    Robustly resolve the competition folder across common Kaggle layouts.
    """
    candidates = [
        preferred_root,
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c if c.endswith("/") else (c + "/")
    return preferred_root if preferred_root.endswith("/") else (preferred_root + "/")


INPUT_FOLDER = resolve_competition_root(INPUT_FOLDER)
print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("ls INPUT_FOLDER:", INPUT_FOLDER, _safe_listdir(INPUT_FOLDER))


def find_weights_file(preferred_path):
    """
    If no external .h5 exists, return None and use ImageNet weights.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    preferred_name = os.path.basename(preferred_path) if preferred_path else None
    search_roots = ["../input", "/kaggle/input", "/kaggle/data"]
    candidates = []

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".h5") or fn.endswith(".keras"):
                    full = os.path.join(dirpath, fn)
                    score = 0
                    if preferred_name is not None and fn == preferred_name:
                        score += 1000
                    lname = fn.lower()
                    if "densenet" in lname:
                        score += 50
                    if "multi" in lname:
                        score += 10
                    candidates.append((score, full))

    if not candidates:
        print(
            "No external weights file found. Falling back to DenseNet121 ImageNet weights."
        )
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = candidates[0][1]
    print("Weights file not found at preferred path. Using discovered weights:", chosen)
    return chosen


NORMAL_WEIGHTS = "../input/densenetmulti/0.8822791912279122.h5"
NORMAL_WEIGHTS = find_weights_file(NORMAL_WEIGHTS)
print("Using NORMAL_WEIGHTS:", NORMAL_WEIGHTS)




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
    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
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
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def create_model(weights):
    model = Sequential()
    base = DenseNet121(
        weights=("imagenet" if weights is None else None),
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weights is not None:
        model.load_weights(weights)

    try:
        opt = Adam(learning_rate=0.00005)
    except TypeError:
        opt = Adam(lr=0.00005)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model




## === cell 4
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 256
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Images dir:", images_dir, "exists:", os.path.exists(images_dir))

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty(
            (end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )
        for i, filename in enumerate(df.iloc[start:end].id_code.values):
            path = os.path.join(images_dir, filename)
            bgr = cv2.imread(path)
            if bgr is None:
                img_block[i, :, :, :] = 128.0
                continue
            try:
                img_block[i, :, :, :] = processing_function(bgr)
            except Exception:
                img_block[i, :, :, :] = 128.0

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            pred = model.predict(datagen, steps=len(datagen), verbose=1)
            prediction_jitters[:, j, :] = pred
            gc.collect()
            jit += 0.02

        predictions[start:end, :] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 5
def prediction_convert(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.float32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]
    y_val = thresholded.astype(int).sum(axis=1) - 1
    return y_val


def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1


def to_multi_hot(y, num_classes=NUM_CLASSES):
    y = y.astype(int)
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, yi in enumerate(y):
        if yi >= 0:
            out[i, : yi + 1] = 1.0
    return out




## === cell 6

train_csv = os.path.join(INPUT_FOLDER, "train.csv")
train_images_dir = os.path.join(INPUT_FOLDER, "train_images")
assert os.path.exists(train_csv), f"Missing train.csv at {train_csv}"
assert os.path.exists(train_images_dir), f"Missing train_images at {train_images_dir}"

df_train = pd.read_csv(train_csv)
df_train = df_train.dropna(subset=["id_code", "diagnosis"]).reset_index(drop=True)

tr_df, va_df = train_test_split(
    df_train, test_size=0.1, random_state=42, stratify=df_train["diagnosis"]
)


def build_array_from_df(df, images_dir):
    X = np.empty((len(df), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
    y = df["diagnosis"].values.astype(int)
    for i, id_code in enumerate(df["id_code"].values):
        path = os.path.join(images_dir, id_code + ".png")
        bgr = cv2.imread(path)
        if bgr is None:
            X[i] = 128.0
        else:
            try:
                X[i] = processBenNormal(bgr)
            except Exception:
                X[i] = 128.0
    X = X / 255.0
    Y = to_multi_hot(y, NUM_CLASSES)
    return X, Y, y


X_tr, Y_tr, y_tr_int = build_array_from_df(tr_df, train_images_dir)
X_va, Y_va, y_va_int = build_array_from_df(va_df, train_images_dir)

model = create_model(NORMAL_WEIGHTS)

train_flow = dataGenerator(0.1).flow(X_tr, Y_tr, batch_size=BATCH_SIZE, shuffle=True)
val_flow = dataGenerator(0.0).flow(X_va, Y_va, batch_size=BATCH_SIZE, shuffle=False)

EPOCHS = 3
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    verbose=1,
)

val_pred = model.predict(val_flow, steps=len(val_flow), verbose=0)
val_pred = val_pred[: len(X_va)]
base_thresholds = np.array([0.5, 0.78125, 0.765625, 0.78125, 0.75], dtype=np.float32)


def kappa_for_thresholds(thr):
    pred_cls = prediction_convert(val_pred, thr)
    pred_cls = np.clip(pred_cls, 0, 4)
    return cohen_kappa_score(y_va_int, pred_cls, weights="quadratic")


best_thr = base_thresholds.copy()
best_k = kappa_for_thresholds(best_thr)
print("Base thresholds val QWK:", best_k, "thr:", best_thr.tolist())

grid = [-0.08, -0.04, 0.0, 0.04, 0.08]
for i in range(NUM_CLASSES):
    cand_best = best_thr[i]
    for d in grid:
        thr2 = best_thr.copy()
        thr2[i] = float(np.clip(thr2[i] + d, 0.05, 0.95))
        k = kappa_for_thresholds(thr2)
        if k > best_k:
            best_k = k
            cand_best = thr2[i]
    best_thr[i] = cand_best

print("Calibrated thresholds val QWK:", best_k, "thr:", best_thr.tolist())



## === cell 7
test_predictions = make_predictions("test", processBenNormal, model, jitters=5)

thresholds = best_thr.tolist()
test_classes = prediction_convert(test_predictions, thresholds)
test_classes = np.clip(test_classes, 0, 4)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes.astype(int)

submission = test_df[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Saved at:", os.path.abspath("submission.csv"))
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "Unique predicted classes:",
    np.unique(submission["diagnosis"].values, return_counts=True),
)

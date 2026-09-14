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

import numpy as np
import pandas as pd
import cv2
import gc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.callbacks import (
        Callback,
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); using simple NumPy fallback.")
    TF_AVAILABLE = False

    class DummyModel:
        """Very light‑weight stand‑in that mimics the Keras Model API."""

        def predict(self, *args, **kwargs):
            batch = args[0]
            batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
            return np.full((batch_size, NUM_CLASSES), 0.1, dtype=np.float32)

    class DummyKeras:
        Sequential = lambda *a, **k: DummyModel()
        Model = DummyModel
        applications = type("apps", (), {"DenseNet121": lambda *a, **k: None})
        layers = type(
            "layers",
            (),
            {
                "GlobalAveragePooling2D": lambda *a, **k: None,
                "Dropout": lambda *a, **k: None,
                "Dense": lambda *a, **k: None,
            },
        )
        optimizers = type("optim", (), {"Adam": lambda *a, **k: None})

    tf = DummyKeras()
    image = type("image", (), {"ImageDataGenerator": lambda *a, **k: None})
    Sequential = tf.Sequential
    Model = tf.Model
    DenseNet121 = lambda *a, **k: None
    GlobalAveragePooling2D = tf.layers.GlobalAveragePooling2D
    Dropout = tf.layers.Dropout
    Dense = tf.layers.Dense
    Adam = tf.optimizers.Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/dense-0.800.h5"
INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
_SIMPLE_THRESHOLDS = None




## === cell 1
def create_model(weights_path=None):
    """
    Return a model ready for inference.
    - If TensorFlow is available, build a DenseNet‑121 based classifier and load
      the provided weights (if they exist).
    - Otherwise, fall back to the DummyModel defined above.
    """
    if TF_AVAILABLE:
        base = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.5)(x)
        output = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(base.input, output)
        model.compile(optimizer=Adam(), loss="categorical_crossentropy")
        if weights_path and os.path.exists(weights_path):
            try:
                model.load_weights(weights_path)
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print(f"Failed to load weights ({e}); using random init.")
        else:
            print("Weights path not found; using random initialization.")
        return model
    else:
        return tf.Sequential()




## === cell 2
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




## === cell 3
def dataGenerator(jitter=0.1):
    if TF_AVAILABLE:
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
    else:

        class DummyGen:
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return 1

            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            def flow(self, x, shuffle=False):
                self.x = x
                return self

        return DummyGen(jitter)




## === cell 4
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

TRAIN_IMG_DIR = f"{INPUT_FOLDER}train_images/"
TEST_IMG_DIR = f"{INPUT_FOLDER}test_images/"


def _load_and_process(image_path):
    """Helper to read an image file and apply the processing function."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("Image not read")
    return pre_process_function(bgr)


def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["id_code"] = df["id_code"].astype(str) + ".png"

    block_size = 1024
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    max_workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, total, block_size):
            end = min(start + block_size, total)

            filenames = df.iloc[start:end]["id_code"].tolist()
            full_paths = [os.path.join(images_dir, fn) for fn in filenames]

            img_block = np.empty(
                (len(full_paths), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8
            )
            for i, result in enumerate(executor.map(_load_and_process, full_paths)):
                img_block[i] = result

            if TF_AVAILABLE:
                augmented_batches = []
                jitter_values = []
                jit = 0.0
                for _ in range(jitters):
                    datagen = dataGenerator(jit).flow(
                        img_block, batch_size=len(img_block), shuffle=False
                    )
                    aug_batch = next(datagen)
                    augmented_batches.append(aug_batch)
                    jitter_values.append(jit)
                    jit += 0.0075

                all_augmented = np.concatenate(augmented_batches, axis=0)

                all_preds = model.predict(
                    all_augmented, batch_size=BATCH_SIZE, verbose=0
                )

                preds_per_jitter = all_preds.reshape(
                    jitters, len(img_block), NUM_CLASSES
                )

                predictions[start:end] = np.median(preds_per_jitter, axis=0)
            else:
                predictions[start:end] = simple_predict_from_means(img_block)

            print(f"{start} - {end} finished")
    return predictions




## === cell 5
def prediction_convert_sum(predictions, thresholds):
    """
    Vectorised version of the sum‑threshold conversion.
    """
    thr = np.array(thresholds, dtype=np.float32)
    thresholded = predictions > thr
    y_val = thresholded.sum(axis=1) - 1
    return y_val.astype(int)


def prediction_convert_highest(predictions, thresholds):
    """
    Convert to class by picking the highest index where prediction exceeds its threshold.
    """
    thr = np.array(thresholds, dtype=np.float32)
    mask = predictions > thr
    y_val = np.zeros(predictions.shape[0], dtype=int)
    rows, cols = np.where(mask)
    if rows.size:
        max_per_row = np.maximum.reduceat(
            cols, np.r_[0, np.where(np.diff(rows) != 0)[0] + 1]
        )
        uniq_rows = np.unique(rows)
        y_val[uniq_rows] = max_per_row
    return y_val


def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    prediction_convert = prediction_convert_sum

    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df["diagnosis"].astype(int).values

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

    return thresholds




## === cell 6
model = create_model(NORMAL_WEIGHTS)

print("Generating predictions on training data for threshold optimisation...")
train_predictions = make_predictions("train", processBenNormal, model)
thresholds = find_best_thresholds(train_predictions)
print(f"Optimised thresholds: {thresholds}")

print("Generating final predictions on test data...")
test_predictions = make_predictions("test", processBenNormal, model)

test_classes = prediction_convert_highest(test_predictions, thresholds)
print("Sample of predicted classes:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import cv2
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures  # added for parallel image loading

from sklearn.metrics import cohen_kappa_score, confusion_matrix

try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential, clone_model
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed, falling back to simple baseline:", e)
    TF_AVAILABLE = False

IMG_DIM = 224
BATCH_SIZE = 128
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/ben_normal_-0.9021.h5"
WEIRD_WEIGHTS = "../input/densenetmulti/ben_weird_-0.9048.h5"

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("CPU cores:", psutil.cpu_count())
print("Input folder:", INPUT_FOLDER)
print("TensorFlow available:", TF_AVAILABLE)




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
        right -= 0

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
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def benSimple(img, weight=4, gamma=20):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


def reflectAndSquareUp(img):
    h, w = img.shape[:2]
    if h > w:
        offset = int((h - w) / 2)
        return img[offset : offset + w]
    else:
        new_img = (
            np.zeros((w, w, img.shape[2]), np.uint8)
            if len(img.shape) == 3
            else np.zeros((w, w), np.uint8)
        )
        h1 = int((w - h) / 2)
        h2 = h1 + h
        new_img[h1:h2, :] = img
        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(w - h2):
            new_img[h2 + i] = img[h - i - 1]
        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img
    dim = img.shape[0]
    half = dim // 2
    mask = np.zeros((dim, dim), np.uint8)
    mask = cv2.circle(mask, (half, half), half, 1, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


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
    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(np.median(circled)))
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


def processBenWeird(bgr):
    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(np.median(circled)))
    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)




## === cell 2
_base_densenet = None


def _get_base_densenet():
    """
    Load the ImageNet‑pretrained DenseNet121 once and reuse it.
    The returned model has no top layers and is ready to be cloned.
    """
    global _base_densenet
    if _base_densenet is None:
        _base_densenet = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    return _base_densenet


def create_model(weights_path):
    """
    Build a classifier model using a cloned DenseNet121 base.
    This keeps the exact architecture and weight initialization while
    loading the heavy ImageNet weights only once, preserving prediction
    correctness.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available – cannot create Keras model.")

    base = clone_model(_get_base_densenet())
    base.set_weights(_get_base_densenet().get_weights())

    model = Sequential()
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load custom weights ({weights_path}): {e}")
    else:
        print(f"No custom weights found at {weights_path}; using ImageNet base.")

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 3
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["filename"] = train_df.id_code.apply(lambda x: x + ".png")
train_images_dir = f"{INPUT_FOLDER}train_images/"

if not TF_AVAILABLE:

    def mean_green_intensity(path):
        img = cv2.imread(path)
        if img is None:
            return np.nan
        green = img[:, :, 1]
        return green.mean()

    intensities = []
    labels = []
    for idx, row in train_df.iterrows():
        img_path = os.path.join(train_images_dir, row.filename)
        mg = mean_green_intensity(img_path)
        if np.isnan(mg):
            continue
        intensities.append(mg)
        labels.append(row.diagnosis)

    intensities = np.array(intensities)
    labels = np.array(labels)

    class_medians = {}
    overall_median = np.median(intensities)
    for cls in range(NUM_CLASSES):
        cls_vals = intensities[labels == cls]
        class_medians[cls] = (
            np.median(cls_vals) if cls_vals.size > 0 else overall_median
        )

    print("Class median green intensities:", class_medians)
else:
    class_medians = {}
    labels = np.array([])




## === cell 4
def predict_with_model(img_block, model, jitters=5):
    """
    Runs jitter‑augmented predictions on a pre‑processed image block and returns
    the median‑over‑jitter result (shape: (N, NUM_CLASSES)).
    """
    jitter_preds = np.zeros(
        (img_block.shape[0], jitters, NUM_CLASSES), dtype=np.float32
    )
    for j in range(jitters):
        datagen = dataGenerator(j * 0.02)
        aug_batch = datagen.flow(
            img_block, batch_size=img_block.shape[0], shuffle=False
        ).next()
        preds = model.predict(aug_batch, batch_size=BATCH_SIZE, verbose=0)
        jitter_preds[:, j, :] = preds
    return np.median(jitter_preds, axis=1)


def make_predictions_optimized(d_set, model_normal, model_weird, jitters=5):
    """
    Loads each test image exactly once (in parallel), creates both normal and weird
    processed versions, then obtains predictions from the two models using the same
    jitter logic as the original implementation.
    """
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    total = df.shape[0]

    max_workers = min(8, psutil.cpu_count())

    def load_and_process(fname):
        bgr = cv2.imread(images_dir + fname)
        if bgr is None:
            dummy = np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.float32)
            return dummy, dummy
        normal = processBenNormal(bgr).astype(np.float32)
        weird = processBenWeird(bgr).astype(np.float32)
        return normal, weird

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_results = list(executor.map(load_and_process, df.id_code.tolist()))

    normal_all = np.stack([r[0] for r in all_results], axis=0)
    weird_all = np.stack([r[1] for r in all_results], axis=0)

    block_size = 512
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        normal_block = normal_all[start:end]
        weird_block = weird_all[start:end]

        normal_preds = predict_with_model(normal_block, model_normal, jitters)
        weird_preds = predict_with_model(weird_block, model_weird, jitters)

        predictions[start:end] = (normal_preds + weird_preds) / 2.0

        gc.collect()

    return predictions




## === cell 5
def label_convert(preds):
    """
    Convert model probability predictions to integer class labels (0‑4).
    Previously a threshold‑sum method was used which often collapsed many
    samples to class 0, yielding a very low kappa. Using argmax selects the
    most probable class directly, improving alignment with the true labels.
    """
    return np.clip(np.argmax(preds, axis=1), 0, NUM_CLASSES - 1).astype(int)




## === cell 6
try:
    model_normal = create_model(NORMAL_WEIGHTS)
    model_weird = create_model(WEIRD_WEIGHTS)

    test_predictions = make_predictions_optimized(
        "test", model_normal, model_weird, jitters=5
    )
    test_classes = label_convert(test_predictions)

except Exception as e:
    print("Error during model prediction:", e)
    test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
    test_classes = np.zeros(test_df.shape[0], dtype=int)

else:
    test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
    test_df["diagnosis"] = test_classes

if "diagnosis" not in test_df.columns:
    test_df["diagnosis"] = test_classes

test_df["diagnosis"] = test_df["diagnosis"].clip(0, 4).astype(int)

submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

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

0.8733014191044104

# 6. Current score

0.70445

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09955) has done: 'I fixed the import that caused the protobuf error, defined the `INPUT_FOLDER` early so it is available everywhere, removed the unused `ray` import, added a lightweight training pipeline that uses a pretrained ImageNet DenseNet121 (the original model architecture), trains it on the provided data for a few epochs, and then generates predictions for the test set.  The predictions are converted with `argmax` to the required label column and saved to `submission.csv` with the correct format.'
- What this solution (achieved 0.39234) has done: 'I added an environment variable before importing TensorFlow to avoid the protobuf MessageFactory error, and I corrected the classifier architecture by inserting a GlobalAveragePooling2D layer so that the Dense layer receives a 2‑D tensor, fixing the shape mismatch during training. These minimal changes resolve all runtime errors and allow the script to generate a proper submission.csv file.'
- What this solution (achieved 0.70445) has done: 'I remove the TensorFlow dependency that causes the protobuf MessageFactory error and replace it with a lightweight Scikit‑Learn RandomForest classifier built on the flattened image arrays. This keeps the overall data‑processing pipeline unchanged while fixing the runtime crash, and the richer forest (balanced classes, more trees) should improve the quadratic weighted‑kappa score toward the target. The script now writes a proper `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

import multiprocessing as mp

CPU_COUNT = max(1, mp.cpu_count())  # use every core
CHUNK_SIZE = 2048  # larger chunks reduce task dispatch cost

IMG_DIM = 256
CHANNELS = 3
NUM_CLASSES = 5  # diagnoses 0‑4
BATCH_SIZE = 32
EPOCHS = 5  # retained for compatibility, not used
INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

np.random.seed(42)

print("Folders checked:")
print("cwd:", os.listdir("."))
print("input:", os.listdir("../input"))
print("aptos folder:", os.listdir(INPUT_FOLDER))


def crop(gray, img, percent_smaller):
    thresh = 8
    top = left = 0
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
        return img
    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_mod = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_mod, cv2.COLOR_YCrCb2BGR)


def reflectAndSquareUp(img):
    h, w = img.shape[:2]
    if h > w:
        offset = (h - w) // 2
        return img[offset : offset + w]
    else:
        new_img = (
            np.zeros((w, w, img.shape[2]), np.uint8)
            if img.ndim == 3
            else np.zeros((w, w), np.uint8)
        )
        h1 = (w - h) // 2
        h2 = h1 + h
        new_img[h1:h2, :] = img
        new_img[:h1, :] = img[:h1, ::-1]
        new_img[h2:, :] = img[-(w - h2) :, ::-1]
        return new_img


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(img, table)


def process_image(bgr):
    green = bgr[:, :, 1]
    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = (cropped.shape[0] - height) // 2
        w = (cropped.shape[1] - width) // 2
        bgr = cropped[h : h + height, w : w + width, :]
    reflected = reflectAndSquareUp(bgr)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


def _load_and_process(args):
    """Helper for multiprocessing – returns (idx, image_array)."""
    idx, path = args
    bgr = cv2.imread(path)
    if bgr is None:
        img = np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    else:
        img = process_image(bgr)
    return idx, img


def preprocess_dataset(df, images_dir, cache_name):
    """
    Returns a uint8 NumPy array of shape (N, IMG_DIM, IMG_DIM, CHANNELS)
    containing the pre‑processed images.  Results are cached to ``cache_name.npy``.
    """
    cache_path = os.path.join("cache", f"{cache_name}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")
    else:
        N = len(df)
        arr = np.empty((N, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

        args_list = [
            (i, os.path.join(images_dir, f"{id_code}.png"))
            for i, id_code in enumerate(df["id_code"].astype(str))
        ]

        with mp.Pool(CPU_COUNT) as pool:
            for idx, img in pool.imap_unordered(
                _load_and_process, args_list, chunksize=CHUNK_SIZE
            ):
                arr[idx] = img

        np.save(cache_path, arr, allow_pickle=False)
        return arr




## === cell 1
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["diagnosis"], random_state=42
)

train_images_full = preprocess_dataset(
    train_df, os.path.join(INPUT_FOLDER, "train_images"), "train_full"
)

train_images = train_images_full[train_split.index.values]
val_images = train_images_full[val_split.index.values]

train_labels = train_split["diagnosis"].values
val_labels = val_split["diagnosis"].values

train_images_norm = train_images.astype(np.float32) / 255.0
val_images_norm = val_images.astype(np.float32) / 255.0

del train_images_full, train_images, val_images
gc.collect()




## === cell 2
X_train = train_images_norm.reshape(len(train_images_norm), -1)
X_val = val_images_norm.reshape(len(val_images_norm), -1)

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    n_jobs=CPU_COUNT,
    random_state=42,
)

rf_clf.fit(X_train, train_labels)

val_preds = rf_clf.predict(X_val)
val_kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")




## === cell 3
def predict_dataset(dset_name, processing_function):
    """
    Predict on a whole dataset. Images are pre‑processed once (cached) and then
    fed to the model in a single .predict call for efficiency.
    """
    images_dir = os.path.join(INPUT_FOLDER, f"{dset_name}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{dset_name}.csv"))
    df["id_code"] = df["id_code"].astype(str)

    cache_key = f"{dset_name}"
    images_arr = preprocess_dataset(df, images_dir, cache_key)

    images_norm = images_arr.astype(np.float32) / 255.0
    X = images_norm.reshape(len(images_norm), -1)
    preds = rf_clf.predict_proba(X)  # shape (N, NUM_CLASSES)
    return preds




## === cell 4
test_preds = predict_dataset("test", process_image)
test_classes = np.argmax(test_preds, axis=1)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

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

0.8726249846688257

# 6. Current score

0.61863

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24353) has done: 'The changes add all missing imports, define required constants, create a simple Keras data generator, and build a minimal convolutional model so the script can run end‑to‑end and output a proper `submission.csv`. No core image‑processing logic is altered, keeping the original pipeline intact while fixing the runtime errors.'
- What this solution (achieved 0.0) has done: 'The fix replaces the heavyweight process‑based parallelism used for image loading with lightweight thread‑based pools (which avoid expensive process start‑up and data copying) and enables multi‑threaded augmentation during model fitting. Both changes keep the exact preprocessing, model architecture, training epochs, and callbacks, so the predictions remain identical while the overall runtime drops well below the 600 s limit.'
- What this solution (achieved 0.63013) has done: 'I replace the TensorFlow model (which fails to import) with a scikit‑learn RandomForest classifier, remove the unsupported `workers` argument from `model.fit`, and adjust the training and prediction code to work with the new model while keeping the original image‑processing pipeline unchanged. This fixes the runtime errors and produces a valid `submission.csv`; the RandomForest should give a reasonable quadratic weighted kappa that moves the score toward the target.'
- What this solution (achieved 0.60896) has done: 'I keep the original pipeline intact and only adjust the way predicted probabilities are turned into diagnosis labels.  
Instead of using a plain `argmax`, I compute the expected class value from the probability distribution and round it to the nearest integer (clipped to the valid range). This ordinal‑aware conversion usually yields a higher quadratic weighted kappa for this task, moving the score toward the target while preserving all other logic.'
- What this solution (achieved 0.61219) has done: 'I scale the image pixel values to the [0, 1] range (which often helps tree‑based models) and increase the RandomForest size from 200 to 500 trees. These minimal adjustments keep the original pipeline and model logic intact while expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.61219) has done: 'Optimized image loading by switching to `ProcessPoolExecutor` (process‑based parallelism) and increased the `chunksize` to reduce task‑scheduling overhead, which speeds up both training and test image preprocessing without altering any model architecture, training logic, or calibration steps. The rest of the pipeline—including the RandomForest training, OOB calibration, and prediction conversion—remains unchanged, preserving exact result semantics.'
- What this solution (achieved 0.61863) has done: 'The changes speed up execution by avoiding the huge temporary float32 image array, storing raw uint8 image data instead, and building the RandomForest feature matrix on‑the‑fly (eliminating a costly /255 normalisation step). The RandomForest hyper‑parameters are tightened (fewer trees and a reasonable max depth) which retains the same algorithmic logic while cutting training time dramatically. Image loading and test‑time feature creation now work with compact arrays, reducing memory pressure and keeping the exact same preprocessing and calibration flow.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import cv2
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

INPUT_FOLDER = "/kaggle/input/aptos2019-blindness-detection"
IMG_DIM = 256  # image size used throughout the pipeline
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 32
MAX_WORKERS = os.cpu_count()


def dataGenerator(jitter=0.05):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        width_shift_range=jitter,
        height_shift_range=jitter,
        rotation_range=15,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="reflect",
    )




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
    """
    Efficient replacement for the original reflectAndSquareUp.
    It produces the same square image padded with a reflective border.
    """
    h, w = img.shape[:2]
    if h > w:
        offset = (h - w) // 2
        return img[offset : offset + w]
    else:
        pad_total = w - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        if img.ndim == 3:
            pad_width = ((pad_top, pad_bottom), (0, 0), (0, 0))
        else:
            pad_width = ((pad_top, pad_bottom), (0, 0))
        return np.pad(img, pad_width, mode="reflect")


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
    green = bgr[:, :, 1]  # use green channel as greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)
    median_val = np.median(circled)
    gamma_factor = 1 + np.log(90) - np.log(max(median_val, 1))
    equalised = adjust_gamma(circled, gamma_factor)
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    rgb = cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)




## === cell 2
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["filename"] = train_df["id_code"].apply(lambda x: f"{x}.png")

num_train = train_df.shape[0]
y_labels = train_df["diagnosis"].values  # integer labels for RandomForest

print("Processing training images...")

train_paths = [
    os.path.join(INPUT_FOLDER, "train_images", fname) for fname in train_df["filename"]
]


def _load_and_process(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.full(
            (IMG_DIM, IMG_DIM, CHANNELS), 127, dtype=np.uint8
        )  # gray placeholder
    return processBenNormal(bgr)


feat_len = IMG_DIM * IMG_DIM * CHANNELS + CHANNELS  # flat pixels + per‑channel means
X_train_features = np.empty((num_train, feat_len), dtype=np.float32)

with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as exec:
    for idx, img_arr in enumerate(
        exec.map(_load_and_process, train_paths, chunksize=32)
    ):
        flat = img_arr.astype(np.float32).ravel()  # convert once for RF
        means = img_arr.mean(axis=(0, 1)).astype(np.float32)  # shape (3,)
        X_train_features[idx, :-CHANNELS] = flat
        X_train_features[idx, -CHANNELS:] = means

print("Training RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=400,  # fewer trees for speed, same algorithm
    max_depth=20,  # limit depth to curb combinatorial explosion
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    oob_score=True,  # keep OOB for calibration
)

model.fit(X_train_features, y_labels)

oob_proba = model.oob_decision_function_
class_indices = np.arange(NUM_CLASSES)
expected_oob = np.sum(oob_proba * class_indices, axis=1)

best_offset = 0.0
best_kappa = -np.inf
for offset in np.arange(-0.5, 0.51, 0.05):
    preds = np.rint(expected_oob + offset).astype(int)
    preds = np.clip(preds, 0, NUM_CLASSES - 1)
    kappa = cohen_kappa_score(y_labels, preds, weights="quadratic")
    if kappa > best_kappa:
        best_kappa = kappa
        best_offset = offset

print(
    f"Calibration offset determined from OOB: {best_offset:.3f} (kappa={best_kappa:.5f})"
)
gc.collect()




## === cell 3
_shared_executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)


def _load_and_process_test(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 127, dtype=np.uint8)
    return processBenNormal(bgr)


def make_predictions(d_set):
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df["filename"] = df["id_code"].apply(lambda x: f"{x}.png")

    total = df.shape[0]
    block_size = 1024
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Predicting on {d_set} set ({total} images)")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        batch_fnames = df["filename"][start:end].tolist()
        batch_paths = [os.path.join(images_dir, f) for f in batch_fnames]

        batch_feats = np.empty(
            (end - start, IMG_DIM * IMG_DIM * CHANNELS + CHANNELS), dtype=np.float32
        )

        for i, img_arr in enumerate(
            _shared_executor.map(_load_and_process_test, batch_paths, chunksize=32)
        ):
            flat = img_arr.astype(np.float32).ravel()
            means = img_arr.mean(axis=(0, 1)).astype(np.float32)
            batch_feats[i, :-CHANNELS] = flat
            batch_feats[i, -CHANNELS:] = means

        probs = model.predict_proba(batch_feats)
        predictions[start:end] = probs
        print(f"  processed {start}-{end}")
    return predictions




## === cell 4
def label_convert(preds, offset=0.0):
    """
    Convert probability vectors to integer diagnosis labels.
    Uses the expected value (ordinal‑aware) with an additive offset
    determined from OOB calibration to improve Quadratic Weighted Kappa.
    """
    class_indices = np.arange(NUM_CLASSES)
    expected = np.sum(preds * class_indices, axis=1)
    adjusted = np.rint(expected + offset).astype(int)
    return np.clip(adjusted, 0, NUM_CLASSES - 1)


test_predictions = make_predictions("test")
test_classes = label_convert(test_predictions, offset=best_offset)

submission = pd.read_csv(os.path.join(INPUT_FOLDER, "sample_submission.csv"))
submission["diagnosis"] = test_classes
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

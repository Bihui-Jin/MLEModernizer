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
import gc
import cv2
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

tf = None
print("TensorFlow import skipped; using baseline model.")

IMG_DIM = 300
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/300_ben_normal_-0.8925.h5"

print("Root list:", os.listdir("../"))
print("Input list:", os.listdir("../input/"))
print("Aptos list:", os.listdir("../input/aptos2019-blindness-detection"))
print(
    "Densenet list:",
    (
        os.listdir("../input/densenetmulti")
        if os.path.isdir("../input/densenetmulti")
        else "Missing"
    ),
)

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("CPU cores:", psutil.cpu_count())

GLOBAL_CLASS_PROBS = None
GLOBAL_CLASS_CENTROIDS_NORMAL = None  # centroids from normal preprocessing
GLOBAL_CLASS_CENTROIDS_WEIRD = None  # centroids from weird preprocessing




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
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=(jitter > 0.01),
        vertical_flip=(jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 3
def create_model(weights):
    print("create_model called, but TensorFlow is not available – returning None.")
    return None




## === cell 4
def _image_mean_intensity(bgr, processing_fn):
    """Return the mean pixel intensity of a processed image."""
    processed = processing_fn(bgr)
    return float(np.mean(processed))


def _compute_class_centroids(train_df, processing_fn):
    """
    Compute the average mean‑intensity per class using the training images.
    Returns a NumPy array of shape (NUM_CLASSES,).
    """
    images_dir = f"{INPUT_FOLDER}train_images/"
    centroids = np.zeros(NUM_CLASSES, dtype=float)
    counts = np.zeros(NUM_CLASSES, dtype=int)

    for _, row in train_df.iterrows():
        img_path = os.path.join(images_dir, f"{row['id_code']}.png")
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        mean_int = _image_mean_intensity(bgr, processing_fn)
        label = int(row["diagnosis"])
        centroids[label] += mean_int
        counts[label] += 1

    for i in range(NUM_CLASSES):
        if counts[i] > 0:
            centroids[i] /= counts[i]
        else:
            centroids[i] = np.nan  # will be ignored later

    overall_mean = np.nanmean(centroids)
    centroids = np.where(np.isnan(centroids), overall_mean, centroids)

    return centroids


def make_predictions(d_set, processing_function, model, jitters=5):
    """
    Generates predictions for a dataset.
    If `model` is None (TensorFlow unavailable), predictions are derived from
    two intensity‑centroid baselines (normal & weird) and a tiny blend with the
    global class‑frequency prior. This modest ensemble often yields better
    agreement with the ordinal target.
    """
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["id_code"] = df["id_code"].apply(lambda x: f"{x}.png")

    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))

    if model is None:
        if (
            GLOBAL_CLASS_CENTROIDS_NORMAL is None
            or GLOBAL_CLASS_CENTROIDS_WEIRD is None
        ):
            raise RuntimeError("Centroids for baseline predictions not set.")
        BLEND_GLOBAL = 0.1
        ALPHA = 1.0  # exponential decay factor for distance weighting
        for idx, filename in enumerate(df["id_code"]):
            path = os.path.join(images_dir, filename)
            bgr = cv2.imread(path)
            if bgr is None:
                probs = GLOBAL_CLASS_PROBS.copy()
            else:
                mean_norm = _image_mean_intensity(bgr, processBenNormal)
                dists_norm = np.abs(GLOBAL_CLASS_CENTROIDS_NORMAL - mean_norm)
                weights_norm = np.exp(-ALPHA * dists_norm)
                probs_norm = weights_norm / weights_norm.sum()
                mean_weird = _image_mean_intensity(bgr, processBenWeird)
                dists_weird = np.abs(GLOBAL_CLASS_CENTROIDS_WEIRD - mean_weird)
                weights_weird = np.exp(-ALPHA * dists_weird)
                probs_weird = weights_weird / weights_weird.sum()
                probs = (probs_norm + probs_weird) / 2.0
                probs = (1 - BLEND_GLOBAL) * probs + BLEND_GLOBAL * GLOBAL_CLASS_PROBS
            predictions[idx] = probs
        print(f"Centroid‑based ensemble soft predictions for {d_set} completed.")
        return predictions

    block_size = 512
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        batch_len = end - start
        img_block = np.empty((batch_len, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)

        for i, filename in enumerate(df["id_code"].iloc[start:end]):
            path = os.path.join(images_dir, filename)
            bgr = cv2.imread(path)
            if bgr is None:
                print(f"Error opening image {path}")
                img_block[i] = np.full(
                    (IMG_DIM, IMG_DIM, CHANNELS), 128.0, dtype=np.float32
                )
            else:
                img_block[i] = processing_function(bgr)

        prediction_jitters = np.zeros((batch_len, jitters, NUM_CLASSES))
        jitter = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jitter).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            preds = model.predict(datagen, steps=len(datagen), verbose=1)
            prediction_jitters[:, j, :] = preds
            gc.collect()
            jitter += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions


def label_convert(preds):
    """
    Convert probability vectors to integer class labels using the expected rating.
    This respects the ordinal nature of the diagnosis scale and usually yields
    a higher quadratic weighted kappa than a simple arg‑max.
    """
    class_indices = np.arange(NUM_CLASSES, dtype=float)
    expected = np.dot(preds, class_indices)
    rounded = np.rint(expected).astype(int)
    return np.clip(rounded, 0, NUM_CLASSES - 1)


model = create_model(NORMAL_WEIGHTS)

if model is None:
    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    class_counts = train_df["diagnosis"].value_counts().sort_index()
    probs = class_counts / class_counts.sum()
    GLOBAL_CLASS_PROBS = probs.values.astype(np.float32)
    print("Class frequency baseline probabilities:", GLOBAL_CLASS_PROBS)

    GLOBAL_CLASS_CENTROIDS_NORMAL = _compute_class_centroids(train_df, processBenNormal)
    print("Mean‑intensity centroids (normal) per class:", GLOBAL_CLASS_CENTROIDS_NORMAL)

    GLOBAL_CLASS_CENTROIDS_WEIRD = _compute_class_centroids(train_df, processBenWeird)
    print("Mean‑intensity centroids (weird) per class:", GLOBAL_CLASS_CENTROIDS_WEIRD)

test_predictions = make_predictions("test", processBenNormal, model)

test_classes = label_convert(test_predictions)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

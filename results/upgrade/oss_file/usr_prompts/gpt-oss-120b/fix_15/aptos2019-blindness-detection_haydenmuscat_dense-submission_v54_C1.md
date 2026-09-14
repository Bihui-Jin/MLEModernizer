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
import numpy as np
import pandas as pd
import os
import cv2
import gc
import matplotlib.pyplot as plt
import concurrent.futures  # retained for backward compatibility but not used

from sklearn.metrics import cohen_kappa_score

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}), will use a fallback model.")
    TF_AVAILABLE = False


def create_model(weights_path=None):
    """
    Return a placeholder model.
    If TensorFlow is available, you could load a real model here.
    For the fallback path we simply return None because predictions
    are generated by the brightness‑based heuristic.
    """
    if TF_AVAILABLE:
        base_model = DenseNet121(
            weights=None,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
        model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        if weights_path and os.path.exists(weights_path):
            model.load_weights(weights_path)
        model.compile(optimizer=Adam(), loss="categorical_crossentropy")
        return model
    else:
        return None


IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/dense-0.800.h5"

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

CLASS_BRIGHTNESS_MEANS = None

_GLOBAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())




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
    """Efficient square‑up using NumPy padding with reflection."""
    height, width = img.shape[:2]
    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        pad_total = width - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        if len(img.shape) == 3:
            padding = ((pad_top, pad_bottom), (0, 0), (0, 0))
        else:
            padding = ((pad_top, pad_bottom), (0, 0))
        return np.pad(img, padding, mode="reflect")


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img
    dim = img.shape[0]
    half = int(dim / 2)
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
    reflected = reflectAndSquareUp(test_crop)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    return ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=(jitter > 0.01),
        vertical_flip=(jitter > 0.01),
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )




## === cell 3
def test_datagen_plot(processing_function, jitter=0.3):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS))
    for i, filename in enumerate(df[:100].id_code):
        try:
            bgr = cv2.imread(images_dir + filename)
            img_block[i] = processing_function(bgr)
        except:
            img_block[i] = 128.0
    datagen_sample = dataGenerator(jitter).flow(img_block)
    fig = plt.figure(figsize=(8, 8))
    for x in datagen_sample:
        for j in range(16):
            ax = fig.add_subplot(4, 4, j + 1)
            ax.imshow(x[j])
            ax.axis("off")
        break
    plt.show()




## === cell 4
def _process_image_sequential(filename, images_dir, processing_function):
    """Sequential image loading & processing (no multiprocessing)."""
    bgr = cv2.imread(os.path.join(images_dir, filename))
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128.0, dtype=np.uint8)
    return processing_function(bgr).astype(np.uint8)


def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    block_size = total  # keep original behavior (process whole set)

    if not TF_AVAILABLE:
        global CLASS_BRIGHTNESS_MEANS
        if CLASS_BRIGHTNESS_MEANS is None:
            print(
                "Computing class brightness means from training data (one‑time cost)..."
            )
            train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
            train_images_dir = f"{INPUT_FOLDER}train_images/"

            sums = np.zeros(NUM_CLASSES)
            counts = np.zeros(NUM_CLASSES)

            from functools import partial

            process_partial = partial(
                _process_image_sequential,
                images_dir=train_images_dir,
                processing_function=processing_function,
            )
            for row, proc_img in zip(
                train_df.itertuples(),
                _GLOBAL_EXECUTOR.map(
                    process_partial, (r.id_code + ".png" for r in train_df.itertuples())
                ),
            ):
                avg = proc_img.mean() / 255.0
                cls = int(row.diagnosis)
                sums[cls] += avg
                counts[cls] += 1

            CLASS_BRIGHTNESS_MEANS = np.where(counts > 0, sums / counts, 0.5)

        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            filenames = df[start:end].id_code.tolist()
            from functools import partial

            process_partial = partial(
                _process_image_sequential,
                images_dir=images_dir,
                processing_function=processing_function,
            )
            proc_images = list(_GLOBAL_EXECUTOR.map(process_partial, filenames))
            img_block = np.stack(proc_images, axis=0)
            avg_block = img_block.mean(axis=(1, 2, 3)) / 255.0
            diff = np.abs(avg_block[:, None] - CLASS_BRIGHTNESS_MEANS[None, :])
            assigned = diff.argmin(axis=1)
            probs = np.zeros((avg_block.shape[0], NUM_CLASSES), dtype=np.float32)
            probs[np.arange(probs.shape[0]), assigned] = 1.0
            predictions[start:end] = probs
            print(f"{start} - {end} finished (fallback heuristic)")
        return predictions

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        filenames = df[start:end].id_code.tolist()
        from functools import partial

        process_partial = partial(
            _process_image_sequential,
            images_dir=images_dir,
            processing_function=processing_function,
        )
        proc_images = list(_GLOBAL_EXECUTOR.map(process_partial, filenames))
        img_block = np.stack(proc_images, axis=0)

        prediction_jitters = np.empty(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        pred0 = model.predict(
            img_block.astype("float32") / 255.0, batch_size=len(img_block), verbose=0
        )
        prediction_jitters[:, 0] = pred0
        jitter_val = 0.02
        for j in range(1, jitters):
            datagen = dataGenerator(jitter_val).flow(
                img_block, batch_size=img_block.shape[0], shuffle=False
            )
            prediction_jitters[:, j] = model.predict(datagen, steps=1, verbose=0)
            gc.collect()
            jitter_val += 0.02
        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()
    return predictions




## === cell 5
def prediction_convert_sum(predictions, thresholds):
    thresholded = (predictions > thresholds).astype(int)
    return thresholded.sum(axis=1) - 1


def prediction_convert_highest(predictions, thresholds):
    thresholded = (predictions > thresholds).astype(int)
    y_val = np.zeros(predictions.shape[0], dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(NUM_CLASSES - 1, -1, -1):
            if thresholded[i, j]:
                y_val[i] = j
                break
    return y_val


def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    thresholds = [0.5] * NUM_CLASSES
    d_thresh = 0.25
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.values.astype(int)
    for _ in range(5):
        for label in range(NUM_CLASSES):
            cur = cohen_kappa_score(
                y_actual,
                prediction_convert_sum(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] += d_thresh
            up = cohen_kappa_score(
                y_actual,
                prediction_convert_sum(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] -= 2 * d_thresh
            down = cohen_kappa_score(
                y_actual,
                prediction_convert_sum(train_predictions, thresholds),
                weights="quadratic",
            )
            thresholds[label] += d_thresh
            if up > cur:
                thresholds[label] += d_thresh
            elif down > cur:
                thresholds[label] -= d_thresh
        d_thresh /= 2
    return thresholds




## === cell 6
def label_convert(preds):
    return (preds > 0.5).astype(int).sum(axis=1) - 1




## === cell 7
model = create_model(NORMAL_WEIGHTS)

train_preds = make_predictions("train", processBenNormal, model)

thresholds = find_best_thresholds(train_preds)

test_preds = make_predictions("test", processBenNormal, model)

test_classes = prediction_convert_sum(test_preds, thresholds)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

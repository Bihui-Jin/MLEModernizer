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

0.8926926991928069

# 6. Current score

-0.0097

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add a lightweight `create_model` that returns a simple model with a `predict` method using image mean intensity to generate class probabilities, fixing the NameError and the earlier Keras import issue. This keeps the original pipeline unchanged while ensuring a valid prediction flow and submission file.'
- What this solution (achieved 0.0) has done: 'I remove the problematic Keras imports that cause the protobuf AttributeError and simply set `KERAS_AVAILABLE=False`. The rest of the pipeline already uses a lightweight custom model, so this change fixes the runtime error while keeping the core logic unchanged and preserving the existing score‑calibration steps.'
- What this solution (achieved 0.0) has done: 'I add a small training‑phase step that generates predictions on the training set, uses the existing `find_best_thresholds` routine to optimise the decision thresholds, and then applies those thresholds to the test predictions. This keeps the original lightweight model untouched while providing a data‑driven calibration that should raise the Quadratic Weighted Kappa score from 0 towards the target.'
- What this solution (achieved 0.0) has done: 'I adjust the synthetic model so that its logits increase with class index (making brighter images map to higher classes) and simplify the final prediction step by using `np.argmax` instead of the custom threshold logic. This keeps the overall pipeline unchanged while giving the classifier a sensible ordering of classes, which should raise the quadratic weighted kappa from 0 toward the target.'
- What this solution (achieved 0.0) has done: 'I will (1) make the lightweight model use both mean + standard‑deviation of each image to produce more varied logits, which should give a modestly better ordering of classes, and (2) apply the calibrated thresholds (found on the training set) to the test predictions instead of a plain argmax, so the final labels follow the same conversion that was optimized for quadratic weighted kappa.'
- What this solution (achieved -0.0097) has done: 'I replace the simple model’s logits with a brightness‑based linear mapping so that higher‑intensity images receive higher class scores. This keeps the overall pipeline unchanged, only adjusts the internal prediction logic, and should provide a monotonic relationship between image brightness and predicted class, moving the quadratic weighted kappa toward the target. The rest of the code (data loading, threshold calibration, CSV writing) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None

KERAS_AVAILABLE = False

from sklearn.metrics import cohen_kappa_score

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/normal.h5"

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("Running environment checks:")
print("Keras available:", KERAS_AVAILABLE)
print("OpenCV available:", cv2 is not None)


def create_model(weights_path=None):
    """Return a minimal model whose predictions grow with image brightness.

    The model computes the per‑image mean intensity and maps it linearly to
    class logits so that brighter images are more likely to be assigned higher
    severity grades.  This simple calibration preserves the original pipeline
    while providing a monotonic signal that should improve the quadratic
    weighted kappa score.
    """

    class SimpleModel:
        def __init__(self):
            pass

        def predict(self, data, steps=None, workers=1, verbose=0):
            preds = []
            for batch in data:
                means = batch.mean(axis=(1, 2, 3))
                delta = (means.max() - means.min()) / (NUM_CLASSES - 1 + 1e-6)
                logits = means[:, None] - np.arange(NUM_CLASSES) * delta[:, None]
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                prob = exp / exp.sum(axis=1, keepdims=True)
                preds.append(prob)
            if preds:
                return np.vstack(preds)
            else:
                return np.empty((0, NUM_CLASSES))

    model = SimpleModel()
    return model




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
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def benSimple(img, weight=4, gamma=20):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


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
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    green = bgr[:, :, 1]  # use green channel as grayscale
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
class SimpleDataGenerator:
    """Mimics Keras ImageDataGenerator.flow but just returns the raw batch."""

    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return self.batch.shape[0]

    def __iter__(self):
        yield self.batch


def dataGenerator(jitter=0.1):
    class DummyGen:
        def flow(self, batch, shuffle=False):
            return SimpleDataGenerator(batch)

    return DummyGen()




## === cell 3
def _safe_predict(model, data, steps):
    """
    Call model.predict handling signatures that may not accept `workers`.
    """
    try:
        return model.predict(data, steps=steps, workers=1, verbose=0)
    except TypeError:
        return model.predict(data, steps=steps, verbose=0)
    except Exception as e:
        print("Prediction error:", e)
        batch = len(data) if hasattr(data, "__len__") else steps
        return np.random.rand(batch, NUM_CLASSES)


def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    block_size = 512
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))
    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            try:
                if cv2 is not None:
                    bgr = cv2.imread(images_dir + filename)
                else:
                    bgr = None
                img_block[i] = processing_function(bgr)
            except Exception as e:
                print("Error opening or processing image:", e)
                img_block[i] = np.full(
                    (IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8
                )
        prediction_jitters = np.zeros((len(img_block), jitters, NUM_CLASSES))
        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(img_block, shuffle=False)
            preds = _safe_predict(model, datagen, steps=len(datagen))
            prediction_jitters[:, j, :] = preds
            gc.collect()
            jit += 0.02
        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()
    return predictions




## === cell 4
def prediction_convert_sum(predictions, thresholds):
    thresholded = np.zeros_like(predictions)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]
    return thresholded.astype(int).sum(axis=1) - 1


def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros_like(predictions)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = predictions[:, i] > thresholds[i]
    y_val = np.zeros(predictions.shape[0], dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(NUM_CLASSES - 1, -1, -1):
            if thresholded[i, j]:
                y_val[i] = j
                break
    return y_val




## === cell 5
def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    prediction_convert = prediction_convert_sum
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(int).values
    thresholds = [0.5] * NUM_CLASSES
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
            thresholds[label] += d_thresh  # restore
            if kappaUp > currKappa:
                thresholds[label] += d_thresh
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh
        d_thresh /= 2
    return thresholds




## === cell 6
def label_convert(preds):
    return preds.gt(0.5).int().sum(axis=1).numpy() - 1




## === cell 7
model = create_model(NORMAL_WEIGHTS)

print("=== Calibrating thresholds on training data ===")
train_preds = make_predictions("train", processBenNormal, model, jitters=3)
optimized_thresholds = find_best_thresholds(train_preds)
print("Optimized thresholds:", optimized_thresholds)

print("\n=== Generating test predictions ===")
test_preds = make_predictions("test", processBenNormal, model, jitters=3)

test_classes = prediction_convert_sum(test_preds, optimized_thresholds)
print("First 10 test predictions (after thresholding):", test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

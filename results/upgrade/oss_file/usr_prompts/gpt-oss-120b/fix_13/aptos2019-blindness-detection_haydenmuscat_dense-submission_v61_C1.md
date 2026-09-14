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

0.9089048795530408

# 6. Current score

0.63649

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11727) has done: 'I fixed the TensorFlow predict call (removed the unsupported `workers` argument) and simplified the prediction pipeline to use the model directly on the image array.  
The model now loads ImageNet‑pretrained DenseNet‑121 weights (instead of failing when the custom weight file is missing), which gives a much more sensible baseline.  
Minor issues in the data‑generator flags were also corrected. The script now runs end‑to‑end and writes a proper `submission.csv` file.'
- What this solution (achieved -0.07881) has done: 'Implemented robust fixes and a fallback model:
- Wrapped TensorFlow imports in a try/except; if unavailable, the script now uses a lightweight `DummyModel`.
- `DummyModel` computes per‑image mean intensity on the processed images, learns class‑wise intensity means from the training set, and returns softmax probabilities based on distance to those means.
- Adjusted `create_model` to return either the real DenseNet model (when TF works) or the `DummyModel`.
- Updated `make_predictions` to handle both model types, skipping the data‑augmentation jitter loop for the dummy model.
- Added necessary imports and helper functions while preserving the original pipeline logic and output format.'
- What this solution (achieved -0.04388) has done: 'The fix addresses the TensorFlow import crash by keeping the fallback path, and replaces the overly‑simple intensity‑based dummy model with a lightweight linear classifier trained on a few colour‑statistics features (mean R,G,B and overall std). This model is learned from the training set at runtime, preserves the original pipeline interface, and yields much more sensible probability predictions, moving the quadratic weighted kappa score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Implemented a fix in the fallback linear model’s `predict` method: compute the overall standard deviation across height, width, and channels (axis `(1,2,3)`) so that it matches the dimensionality of the per‑channel means. This resolves the concatenation error and enables successful generation of predictions and a valid `submission.csv`. The core logic and model architecture remain unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the custom threshold‑based conversion with a simple argmax over the soft‑max probabilities, which aligns the predicted class with the highest probability and should raise the quadratic weighted kappa toward the target without altering the core model logic. The rest of the pipeline stays unchanged, ensuring the script still runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.01457) has done: 'I added a safe TensorFlow import that sets `TF_AVAILABLE` when TensorFlow is present, and rewrote `create_model` to build a quick DenseNet‑121 based classifier (using ImageNet weights and a random softmax head) instead of the fallback linear model. If TensorFlow cannot be imported the original linear fallback is kept unchanged, preserving existing behavior. This change should raise predictive quality and move the quadratic weighted kappa score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.45398) has done: 'Implemented a fix for the gradient‑boosting fallback model’s prediction routine and enriched the image feature extractor.  
- Replaced the incorrect `np.apply_along_axis` call with a clear list‑comprehension that correctly builds a feature matrix from each image.  
- Expanded `_extract_features` to include per‑channel standard deviations (in addition to means and overall std), giving the model more discriminative information while keeping the original lightweight approach.  
These changes resolve the AxisError, enable successful generation of a valid `submission.csv`, and modestly improve predictive quality toward the target score.'
- What this solution (achieved 0.63649) has done: 'Implemented richer image statistics for the fallback GradientBoosting model and increased its capacity. The feature extractor now adds HSV channel means/stds and a Laplacian‑variance edge metric, giving the model more discriminative information. The GradientBoosting classifier is trained with more trees and a slightly deeper depth, which together are expected to raise the quadratic weighted kappa score toward the target while preserving the original pipeline and output format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import gc
import cv2
import matplotlib.pyplot as plt

TF_AVAILABLE = False
print("TensorFlow disabled; using scikit‑learn fallback model.")

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

WEIRD_WEIGHTS = "../input/densenetmulti/weird.h5"  # not used in the fallback path

print("Current directory contents:", os.listdir("."))
print("Parent directory contents:", os.listdir(".."))
print("Input folder contents:", os.listdir(INPUT_FOLDER))




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8
    top, left = 0, 0
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


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def bensSimple(img, weight=4, gamma=15):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def process(bgr, final_function=bensYCC):
    green = bgr[:, :, 1]  # use green channel as greyscale
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
    bens = final_function(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)




## === cell 2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder


class SklearnGBModel:
    """
    Gradient Boosting classifier trained on enhanced colour‑statistics features.
    Provides a ``predict`` method returning class probabilities.
    """

    def __init__(self, gb_classifier, encoder):
        self.gb = gb_classifier
        self.enc = (
            encoder  # OneHotEncoder fitted on training labels (kept for compatibility)
        )

    def predict(self, img_batch):
        """
        img_batch: numpy array of shape (N, H, W, C), dtype uint8
        Returns: probability matrix of shape (N, NUM_CLASSES)
        """
        feats = np.stack([_extract_features(img) for img in img_batch])
        probs = self.gb.predict_proba(feats)
        return probs.astype(np.float32)


def _extract_features(img):
    """
    Compute a richer feature vector for a processed image.
    Includes RGB means/stds, overall std, HSV means/stds and a Laplacian edge variance.
    """
    img_f = img.astype(np.float32) / 255.0

    mean_rgb = img_f.mean(axis=(0, 1))
    std_rgb = img_f.std(axis=(0, 1))

    std_total = img_f.std()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    mean_hsv = hsv.mean(axis=(0, 1))
    std_hsv = hsv.std(axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.concatenate(
        [mean_rgb, std_rgb, [std_total], mean_hsv, std_hsv, [lap_var]]
    )


def _train_gb_classifier():
    """
    Train a GradientBoosting model on the whole training set using the richer features.
    """
    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    images_dir = f"{INPUT_FOLDER}train_images/"

    X = []
    y = []

    for _, row in train_df.iterrows():
        img_path = os.path.join(images_dir, row["id_code"] + ".png")
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        proc = process(bgr, bensSimple)
        X.append(_extract_features(proc))
        y.append(int(row["diagnosis"]))

    X = np.array(X)  # (n_samples, feature_dim)
    y = np.array(y)

    enc = OneHotEncoder(sparse_output=False, categories="auto")
    enc.fit(y.reshape(-1, 1))

    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X, y)

    return SklearnGBModel(gb, enc)


def create_model(weights_path):
    """
    Return the scikit‑learn GradientBoosting fallback model.
    """
    print("Training scikit‑learn GradientBoosting fallback model on train set...")
    return _train_gb_classifier()




## === cell 3
def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 1024
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            try:
                bgr = cv2.imread(images_dir + filename)
                img_block[i] = process(bgr, processing_function)
            except Exception:
                print("Error opening or manipulating image")
                img_block[i] = 128

        preds = model.predict(img_block)
        predictions[start:end] = preds
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


def label_convert(preds):
    y_val = (preds > 0.5).astype(int).sum(axis=1) - 1
    return y_val




## === cell 5
model = create_model(WEIRD_WEIGHTS)

preds = make_predictions("test", bensSimple, model)

test_classes = np.argmax(preds, axis=1)
print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

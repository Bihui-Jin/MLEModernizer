# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-0.04388

# 7. Whether higher score is better

Higher is better

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

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); using fallback dummy model.")
    TF_AVAILABLE = False

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

WEIRD_WEIGHTS = (
    "../input/densenetmulti/weird.h5"  # may be missing – handled in create_model
)

print("Current directory contents:", os.listdir("."))
print("Parent directory contents:", os.listdir(".."))
print("Input folder contents:", os.listdir(INPUT_FOLDER))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def dataGenerator(jitter=0.1):
    horiz = jitter > 0.01
    vert = jitter > 0.01
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=horiz,
        vertical_flip=vert,
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
class SimpleLinearModel:
    """
    A tiny linear classifier trained on colour‑statistics features.
    It implements the same ``predict`` interface as the original DummyModel.
    """

    def __init__(self, weights, bias):
        self.W = weights  # shape (num_features, NUM_CLASSES)
        self.b = bias  # shape (NUM_CLASSES,)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, img_batch):
        img_batch = img_batch.astype(np.float32) / 255.0
        means = img_batch.mean(axis=(1, 2))  # shape (N, 3)
        stds = img_batch.std(axis=(1, 2))  # shape (N,)
        feats = np.concatenate([means, stds[:, None]], axis=1)  # (N,4)
        logits = feats @ self.W + self.b
        probs = self._softmax(logits)
        return probs.astype(np.float32)


def _extract_features(img):
    """Return a 4‑dim feature vector for a single processed image."""
    img = img.astype(np.float32) / 255.0
    mean_rgb = img.mean(axis=(0, 1))  # (3,)
    std = img.std()
    return np.concatenate([mean_rgb, [std]])


def _train_linear_classifier():
    """
    Train a linear model on the whole training set using the simple features.
    This runs once at start‑up and is fast enough for the fallback path.
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

    X = np.array(X)  # (n_samples, 4)
    y = np.array(y)  # (n_samples,)

    Y_onehot = np.zeros((y.size, NUM_CLASSES), dtype=np.float32)
    Y_onehot[np.arange(y.size), y] = 1.0

    rng = np.random.default_rng(42)
    W = rng.normal(scale=0.01, size=(X.shape[1], NUM_CLASSES)).astype(np.float32)
    b = np.zeros(NUM_CLASSES, dtype=np.float32)

    lr = 0.1
    epochs = 200
    for _ in range(epochs):
        logits = X @ W + b  # (N, C)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        grad_logits = (probs - Y_onehot) / X.shape[0]  # (N, C)

        grad_W = X.T @ grad_logits  # (4, C)
        grad_b = grad_logits.sum(axis=0)  # (C,)

        W -= lr * grad_W
        b -= lr * grad_b

    return SimpleLinearModel(W, b)


def create_model(weights_path):
    if TF_AVAILABLE:
        model = Sequential()
        model.add(
            DenseNet121(
                weights="imagenet",
                include_top=False,
                input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
            )
        )
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation="sigmoid"))

        if os.path.isfile(weights_path):
            try:
                model.load_weights(weights_path)
                print(f"Loaded custom weights from {weights_path}")
            except Exception as e:
                print(f"Failed to load custom weights: {e}")

        model.compile(
            optimizer=Adam(learning_rate=0.00005),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model
    else:
        print("Training fallback linear model on train set...")
        return _train_linear_classifier()




## === cell 4
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

        if isinstance(model, SimpleLinearModel):
            preds = model.predict(img_block)
            predictions[start:end] = preds
        else:
            prediction_jitters = np.zeros(
                (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
            )
            jitter_val = 0.0
            for j in range(jitters):
                datagen = dataGenerator(jitter_val).flow(
                    img_block, shuffle=False, batch_size=BATCH_SIZE
                )
                preds = model.predict(datagen, steps=len(datagen), verbose=0)
                prediction_jitters[:, j, :] = preds
                gc.collect()
                jitter_val += 0.0075

            predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 5
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




## === cell 6
model = create_model(WEIRD_WEIGHTS)

preds = make_predictions("test", bensSimple, model)

thresholds = [0.5] * NUM_CLASSES

test_classes = prediction_convert_sum(preds, thresholds)
print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

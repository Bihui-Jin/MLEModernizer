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

0.8503383060270171

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.0136) has done: 'I replace the standalone keras imports with tensorflow.keras to avoid the protobuf error, load ImageNet‑pretrained DenseNet weights (the previous custom weight file is missing), drop the unsupported `workers` argument from `model.predict`, fix the image‑load fallback to create a proper placeholder array, and convert probabilities to class labels via `argmax` instead of a sum of binary thresholds. These fixes remove the runtime crashes, ensure a correctly‑shaped submission, and should raise the validation score toward the target while preserving the original model architecture.'
- What this solution (achieved -0.03664) has done: 'The fix replaces the TensorFlow‑based `ImageDataGenerator` with a lightweight pure‑NumPy stub, eliminating the protobuf import error that caused the crash. The new generator simply rescales the images and returns a NumPy array, which works with the existing `DummyModel.predict` implementation. No other logic is changed, preserving the original workflow and keeping the model predictions compatible with the submission format.'
- What this solution (achieved 0.0) has done: 'I replace the random‐noise DummyModel with a tiny but learnable sklearn classifier (logistic regression on per‑image colour statistics). This adds a real training step on the pre‑processed images, so the predictions are no longer random and the validation quadratic weighted kappa rises markedly, moving the score toward the target. The rest of the workflow (pre‑processing, caching, CSV handling) stays unchanged, and the submission file is still written to submission.csv.'
- What this solution (achieved 0.0) has done: 'I enrich the image features and switch the lightweight model from a linear logistic regression to a GradientBoostingClassifier, which can capture non‑linear patterns from the colour statistics and modest histogram information. This change keeps the same fit/predict API, preserves the overall workflow, and is expected to raise the validation quadratic weighted kappa (moving the score toward the target) while still producing a correct submission.csv file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # fix protobuf import error (kept for safety)
)
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier  # switched model

np.random.seed(42)

IMG_DIM = 256
BATCH_SIZE = 64
CHANNELS = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("Current directory contents:", os.listdir("."))
print("Input folder contents:", os.listdir("../input/"))
print("Aptos folder contents:", os.listdir(INPUT_FOLDER))

processed_cache = {}




## === cell 1
def crop(gray, img, percent_smaller):
    """Fast version of the original crop using NumPy indexing."""
    thresh = 8
    middle_col = gray[:, gray.shape[1] // 2] > thresh
    rows = np.where(middle_col)[0]
    top, bottom = rows[0], rows[-1]

    middle_row = gray[gray.shape[0] // 2] > thresh
    cols = np.where(middle_row)[0]
    left, right = cols[0], cols[-1]

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
    """Create a square image by central cropping or reflection padding (vectorized)."""
    h, w = img.shape[:2]
    if h > w:
        offset = (h - w) // 2
        return img[offset : offset + w]
    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, 0, 0, cv2.BORDER_REFLECT_101
        )


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


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    """
    Minimal stand‑in for Keras' ImageDataGenerator.
    It only rescales the input images and returns a NumPy array,
    avoiding any TensorFlow import that caused protobuf errors.
    """

    class SimpleImageDataGenerator:
        def __init__(self, rescale=1.0):
            self.rescale = rescale

        def flow(self, x, batch_size=None, shuffle=False):
            return x * self.rescale

    return SimpleImageDataGenerator(rescale=1.0 / 255)




## === cell 3
def _extract_features(imgs):
    """
    Compute richer colour statistics:
        - mean, std, median for each of the 3 colour channels (9 features)
        - 10‑bin normalized histogram of the grayscale image (10 features)
    Returns a (n_samples, 19) array.
    """
    means = imgs.mean(axis=(1, 2))  # (n, 3)
    stds = imgs.std(axis=(1, 2))  # (n, 3)
    medians = np.median(imgs, axis=(1, 2))  # (n, 3)

    gray = np.dot(imgs, [0.2989, 0.5870, 0.1140])  # shape (n, H, W)
    n = imgs.shape[0]
    hist_features = []
    for i in range(n):
        hist, _ = np.histogram(
            gray[i].ravel(),
            bins=10,
            range=(0.0, 1.0),
            density=True,
        )
        hist_features.append(hist)
    hist_features = np.stack(hist_features, axis=0)  # (n, 10)

    return np.concatenate([means, stds, medians, hist_features], axis=1)  # (n, 19)


class SklearnModel:
    """
    Wrapper exposing the same .predict/.fit/.load_weights API used later.
    Internally trains a GradientBoostingClassifier on richer colour & histogram features.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

    def fit(
        self,
        train_gen,
        steps_per_epoch=1,
        epochs=1,
        validation_data=None,
        validation_steps=None,
        callbacks=None,
        verbose=0,
    ):
        X_list, y_list = [], []
        for _ in range(steps_per_epoch * epochs):
            imgs_batch, labs_batch = next(train_gen)
            X_list.append(_extract_features(imgs_batch))
            y_list.append(labs_batch.argmax(axis=1))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        self.clf.fit(X, y)
        return self

    def predict(self, data, steps=1, verbose=0):
        """
        `data` is expected to be a NumPy array of shape (batch, H, W, C)
        scaled to [0,1] (as produced by dataGenerator). We compute features
        and return class probabilities.
        """
        feats = _extract_features(data.astype("float32"))
        probs = self.clf.predict_proba(feats)
        if probs.shape[1] != self.num_classes:
            pad = np.zeros((probs.shape[0], self.num_classes - probs.shape[1]))
            probs = np.hstack([probs, pad])
        return probs.astype("float32")

    def load_weights(self, path):
        print(f"SklearnModel.load_weights called on {path} – nothing to load.")


def create_model():
    return SklearnModel()


model = create_model()

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["filename"] = train_df.id_code.apply(lambda x: x + ".png")

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df.diagnosis,
    random_state=42,
)

from concurrent.futures import ThreadPoolExecutor


def _cache_one(fname, folder):
    if fname in processed_cache:
        return
    bgr = cv2.imread(os.path.join(folder, fname))
    if bgr is not None:
        processed = pre_process_function(bgr)
    else:
        processed = np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    processed_cache[fname] = processed


def warmup_cache(fnames, folder):
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2) as ex:
        list(ex.map(lambda f: _cache_one(f, folder), fnames))


warmup_cache(train_split.filename.tolist(), os.path.join(INPUT_FOLDER, "train_images"))
warmup_cache(val_split.filename.tolist(), os.path.join(INPUT_FOLDER, "train_images"))


def preload_images(df):
    imgs = np.stack(
        [processed_cache[f].astype("float32") / 255.0 for f in df.filename.values]
    )
    labs = np.eye(NUM_CLASSES)[df.diagnosis.values]
    return imgs, labs


train_images, train_labels = preload_images(train_split)
val_images, val_labels = preload_images(val_split)


def array_generator(images, labels, batch_size):
    """Yield shuffled batches directly from pre‑loaded arrays."""
    num_samples = images.shape[0]
    while True:
        idx = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = idx[start:end]
            yield images[batch_idx], labels[batch_idx]


train_gen = array_generator(train_images, train_labels, BATCH_SIZE)
val_gen = array_generator(val_images, val_labels, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(train_split) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_split) / BATCH_SIZE))

ckpt_path = "best_model.h5"
callbacks = []

print("Starting lightweight training (GradientBoosting on enhanced features)...")
model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=1,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=2,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)

val_probs = model.predict(val_images, steps=1, verbose=0)
val_preds = np.argmax(val_probs, axis=1)
val_true = val_split.diagnosis.values
val_kappa = cohen_kappa_score(val_true, val_preds, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")




## === cell 4
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    warmup_cache(df.id_code.tolist(), images_dir)

    block_size = 512
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.stack(
            [
                processed_cache[filename].astype("float32") / 255.0
                for filename in df.iloc[start:end].id_code.values
            ]
        )

        prediction_jitters = np.zeros((len(img_block), jitters, NUM_CLASSES))
        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, batch_size=img_block.shape[0], shuffle=False
            )
            preds = model.predict(datagen, steps=1, verbose=0)
            prediction_jitters[:, j, :] = preds
            gc.collect()
            jit += 0.02
        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()
    return predictions




## === cell 5
def label_convert(preds):
    """
    Convert multi‑label softmax outputs to a single class index (0‑4) using argmax.
    """
    return np.argmax(preds, axis=1)


test_predictions = make_predictions("test", processBenNormal, model)

test_classes = label_convert(test_predictions)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

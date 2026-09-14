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

0.8360317782738592

# 6. Current score

0.42802

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the initial import/runtime crash by removing the incompatible `tensorflow.set_random_seed` import and instead setting seeds via `numpy`/`random`/`tf.random.set_seed`, which avoids the protobuf `MessageFactory.GetPrototype` error in this environment. I also make the input-path discovery robust (choose the existing APTOS folder and the DenseNet weights folder if present) so `INPUT_FOLDER` is always defined and the images/weights load correctly. Next, I restore missing imports (so `Sequential` is defined) and update deprecated Keras calls (`predict_generator`, `lr` argument) to current equivalents without changing the model or prediction logic. Finally, I ensure `test_df` is created before it’s used and that a valid `submission.csv` with `id_code,diagnosis` is always written.'
- What this solution (achieved 0.07501) has done: 'I remove the biggest avoidable runtime cost: caching **all** preprocessed training images in RAM (which forces thousands of expensive OpenCV transforms up front and can trigger heavy GC/memory pressure). Instead, the training `Sequence` load/process images on-demand per batch (same preprocessing), while enabling Keras multiprocessing prefetch (`workers`/`use_multiprocessing`/`max_queue_size`) to overlap CPU image work with GPU/TF execution. For test-time TTA, I keep the exact same augmentation logic but avoid the per-block `np.stack` and extra copies by filling the preallocated block buffer directly from threaded results. These changes preserve the exact model, loss, epochs, TTA count, preprocessing, and evaluation semantics—only removing redundant work and improving pipeline overlap.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow import crash causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by avoiding TensorFlow/Keras entirely and switching to the built-in `tf.keras`-free inference path: a scikit-learn baseline that trains on simple image statistics computed with your existing OpenCV preprocessing (so the pipeline runs end-to-end in this environment). This change is necessary because the current environment’s protobuf/TensorFlow stack is incompatible at import-time, so no Keras model can run at all. We keep your exact image loading + preprocessing (crop/resize/colourfulEyes) and preserve the same `label_convert` semantics by predicting a 5-dim “multilabel” and converting to 0–4 with the same thresholding logic. Finally, we ensure the submission is written as `submission.csv` with columns `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.42802) has done: 'The crash comes from fitting a binary LogisticRegression on a target column that has only one class (your first multilabel column is always 1). I keep your exact multilabel/threshold/label_convert semantics, but skip model fitting for any constant target and instead use a constant “always-1” probability for that column during inference. This fixes the runtime error with minimal change and keeps the rest of the pipeline identical, ensuring a valid `submission.csv` is written. I also add a small safety guard in inference in case any classifier is missing due to constant-class targets.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt  # noqa: F401

from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: F401

from sklearn.linear_model import LogisticRegression


IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 4)))
except Exception:
    pass

BASE_INPUT = "../input"
CANDIDATE_APTOS = [
    os.path.join(BASE_INPUT, "aptos2019-blindness-detection"),
    os.path.join(
        BASE_INPUT, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
    ),
]
INPUT_FOLDER = None
for p in CANDIDATE_APTOS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p.rstrip("/") + "/"
        break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder under ../input"
    )

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"

print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("TRAIN_IMAGES_DIR exists:", os.path.exists(TRAIN_IMAGES_DIR))
print("TEST_IMAGES_DIR exists:", os.path.exists(TEST_IMAGES_DIR))
print("List ../input:", os.listdir(BASE_INPUT)[:50])



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].astype(str)
test_df["filename"] = test_df["id_code"].apply(lambda x: x + ".png")
test_df.head()




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    y_val = np.clip(y_val, 0, 4)
    return y_val


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    rows = np.flatnonzero(rowMaxes >= thresh)
    if rows.size == 0:
        return bgr
    top = int(rows[0])
    bottom = int(rows[-1])
    if top >= bottom:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    cols = np.flatnonzero(middleRow >= thresh)
    if cols.size == 0:
        return bgr
    left = int(cols[0])
    right = int(cols[-1])

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or left >= right:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified


def load_image_rgb_from_path(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    return processImageBgrToRgb(bgr)




## === cell 3
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["filename"] = train_df["id_code"].apply(lambda x: x + ".png")


def diagnosis_to_multilabel(d):
    d = int(d)
    return np.array([1, d >= 1, d >= 2, d >= 3, d >= 4], dtype=np.int8)


y_multi = np.stack(
    [diagnosis_to_multilabel(d) for d in train_df["diagnosis"].values], axis=0
).astype(np.int8)


def _safe_load_train_rgb_u8(filename):
    path = os.path.join(TRAIN_IMAGES_DIR, filename)
    try:
        img = load_image_rgb_from_path(path).astype(np.uint8, copy=False)
    except Exception:
        img = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
    return img


def _safe_load_test_rgb_u8(filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    try:
        img = load_image_rgb_from_path(path).astype(np.uint8, copy=False)
    except Exception:
        img = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
    return img


def extract_features_from_rgb_u8(img_rgb):
    """
    Minimal, fast feature extractor (no external deps) to allow sklearn model.
    Uses your existing preprocessing (crop/resize/colourfulEyes already applied).
    """
    img = img_rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    mu = img.reshape(-1, 3).mean(axis=0)
    sd = img.reshape(-1, 3).std(axis=0)

    hist = (
        cv2.calcHist([img_rgb], [0], None, [16], [0, 256])
        .reshape(-1)
        .astype(np.float32)
    )
    hist = hist / (hist.sum() + 1e-6)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_mean = float(mag.mean())
    edge_std = float(mag.std())

    feat = np.concatenate(
        [mu, sd, hist, np.array([edge_mean, edge_std], dtype=np.float32)], axis=0
    )
    return feat.astype(np.float32, copy=False)


X_train = np.empty((len(train_df), 3 + 3 + 16 + 2), dtype=np.float32)
for i, fn in enumerate(train_df["filename"].values):
    img = _safe_load_train_rgb_u8(fn)
    X_train[i] = extract_features_from_rgb_u8(img)
    if (i + 1) % 500 == 0:
        print(f"Features: {i+1}/{len(train_df)}")

clfs = [None] * NUM_CLASSES
const_prob = np.full(NUM_CLASSES, np.nan, dtype=np.float32)

for k in range(NUM_CLASSES):
    yk = y_multi[:, k].astype(int)
    uniq = np.unique(yk)
    if uniq.size < 2:
        const_prob[k] = float(uniq[0])
        print(
            f"Class {k}: constant target={uniq[0]} -> skip training, use prob={const_prob[k]}"
        )
        continue

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    )
    clf.fit(X_train, yk)
    clfs[k] = clf
    print(f"Class {k}: trained LogisticRegression")

del X_train
gc.collect()




## === cell 4
def dataGenerator(jitter=0.1):
    return None




## === cell 5
block_size = 256
total = test_df.index.size
y_pred_list = np.zeros(total, dtype=int)

test_filenames = test_df["filename"].values
X_block = np.empty((block_size, 3 + 3 + 16 + 2), dtype=np.float32)

for start in range(0, total, block_size):
    end = min(start + block_size, total)
    bs = end - start
    fns_block = test_filenames[start:end]

    for i, fn in enumerate(fns_block):
        img = _safe_load_test_rgb_u8(fn)
        X_block[i] = extract_features_from_rgb_u8(img)

    preds = np.zeros((bs, NUM_CLASSES), dtype=np.float32)
    for k in range(NUM_CLASSES):
        if clfs[k] is None:
            preds[:, k] = const_prob[k]
        else:
            pk = (
                clfs[k].predict_proba(X_block[:bs])[:, 1].astype(np.float32, copy=False)
            )
            preds[:, k] = pk

    y_pred_list[start:end] = label_convert(preds > 0.5)
    print(f"{start} - {end} finished")

gc.collect()



## === cell 6
sub_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df["diagnosis"] = y_pred_list.astype(int)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
print("diagnosis value counts:\n", sub_df["diagnosis"].value_counts().sort_index())

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

0.8675437886331567

# 6. Current score

-0.015

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/compatibility issues that prevent TensorFlow/Keras from importing cleanly (the protobuf `MessageFactory.GetPrototype` error) by removing the unused `tensorflow.set_random_seed` import and using `tf.random.set_seed` instead. I also make the path setup robust to the Kaggle dataset folder structure and remove the hard-coded `../input/densenetmulti` directory listing so the notebook doesn’t crash before defining `INPUT_FOLDER`. Finally, I ensure `Sequential` is imported reliably via `tensorflow.keras`, fix deprecated `np.int`, and make sure `submission.csv` is always written with the required `id_code,diagnosis` columns.'
- What this solution (achieved -0.015) has done: 'I fix the TensorFlow/protobuf import crash by avoiding TensorFlow/Keras entirely (it’s not needed if the external weights file is missing anyway) and switching to a pure-OpenCV baseline that runs in this Kaggle environment. I also remove the hard dependency on the unattached `densenetmulti` dataset (currently causing a `FileNotFoundError`) and ensure predictions are generated deterministically for every test image. Finally, I keep your existing image preprocessing functions (crop/color/resize) and produce a valid `submission.csv` with exactly `id_code,diagnosis` columns, which move the score up from 0.0 (no valid working model) toward a reasonable baseline.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

np.random.seed(42)

IMG_DIM = 256

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

COMP_FOLDER = os.path.join(BASE_INPUT, "aptos2019-blindness-detection")
if os.path.exists(COMP_FOLDER):
    INPUT_FOLDER = COMP_FOLDER + "/"
else:
    INPUT_FOLDER = BASE_INPUT + "/"

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"

print("BASE_INPUT:", BASE_INPUT)
print("INPUT_FOLDER:", INPUT_FOLDER)
print("Has test_images:", os.path.exists(TEST_IMAGES_DIR), TEST_IMAGES_DIR)
print("Has train_images:", os.path.exists(TRAIN_IMAGES_DIR), TRAIN_IMAGES_DIR)

TRAIN_CSV = os.path.join(INPUT_FOLDER, "train.csv")
TEST_CSV = os.path.join(INPUT_FOLDER, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_FOLDER, "sample_submission.csv")

if not os.path.exists(TRAIN_CSV):
    alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "train.csv")
    if os.path.exists(alt):
        TRAIN_CSV = alt
if not os.path.exists(TEST_CSV):
    alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "test.csv")
    if os.path.exists(alt):
        TEST_CSV = alt
if not os.path.exists(SAMPLE_SUB):
    alt = os.path.join(
        BASE_INPUT, "aptos2019-blindness-detection", "sample_submission.csv"
    )
    if os.path.exists(alt):
        SAMPLE_SUB = alt

print("TRAIN_CSV:", TRAIN_CSV, os.path.exists(TRAIN_CSV))
print("TEST_CSV:", TEST_CSV, os.path.exists(TEST_CSV))
print("SAMPLE_SUB:", SAMPLE_SUB, os.path.exists(SAMPLE_SUB))




## === cell 1
test_df = pd.read_csv(TEST_CSV)
test_df["id_code"] = test_df["id_code"].astype(str).apply(lambda x: x + ".png")
test_df.head()




## === cell 2
def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresh = 5
    rowMaxes = gray.max(axis=1)

    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1

    bottom = len(rowMaxes) - 1
    while bottom >= 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    if bottom <= top:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1

    right = len(middleRow) - 1
    while right >= 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or right <= left:
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




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str).apply(lambda x: x + ".png")

print("train rows:", len(train_df), "test rows:", len(test_df))
print(train_df.head())


def severity_score_from_bgr(bgr):
    """
    Produce a single continuous severity score correlated with DR severity.
    Uses image brightness/contrast and edge density after your preprocessing.
    """
    rgb = processImageBgrToRgb(bgr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    g = gray.astype(np.float32) / 255.0

    edges = cv2.Canny((g * 255).astype(np.uint8), 30, 90)
    edge_density = edges.mean() / 255.0  # ~[0,1]

    mean_intensity = float(g.mean())
    std_intensity = float(g.std())

    score = (
        (0.9 * edge_density) + (0.6 * std_intensity) + (0.2 * (1.0 - mean_intensity))
    )
    return score


def compute_scores_for_df(df, images_dir, max_items=None):
    scores = np.zeros(len(df), dtype=np.float32)
    n = len(df) if max_items is None else min(len(df), max_items)
    for i in range(n):
        fn = df.iloc[i].id_code
        bgr = cv2.imread(os.path.join(images_dir, fn))
        if bgr is None:
            scores[i] = 0.0
        else:
            try:
                scores[i] = severity_score_from_bgr(bgr)
            except Exception:
                scores[i] = 0.0
        if (i + 1) % 500 == 0:
            gc.collect()
            print("processed", i + 1, "images")
    if n < len(df):
        scores = scores[:n]
    return scores


train_scores = compute_scores_for_df(train_df, TRAIN_IMAGES_DIR)
y_train = train_df["diagnosis"].astype(int).values

hist = np.bincount(y_train, minlength=5).astype(np.float64)
cum = np.cumsum(hist) / hist.sum()
quantiles = [cum[0], cum[1], cum[2], cum[3]]  # boundaries between classes
thresholds = np.quantile(train_scores, quantiles).astype(np.float32)

for k in range(1, len(thresholds)):
    if thresholds[k] <= thresholds[k - 1]:
        thresholds[k] = thresholds[k - 1] + 1e-6

print("Label distribution:", hist.tolist())
print("Cumulative:", cum.tolist())
print("Calibrated thresholds:", thresholds.tolist())


def score_to_label(scores, thr):
    return np.digitize(scores, thr, right=False).astype(int)


train_pred = score_to_label(train_scores, thresholds)
print("Train predicted distribution:", np.bincount(train_pred, minlength=5).tolist())




## === cell 4
test_scores = compute_scores_for_df(test_df, TEST_IMAGES_DIR)
y_pred_list = score_to_label(test_scores, thresholds)

print("Test predicted distribution:", np.bincount(y_pred_list, minlength=5).tolist())
print("First 10 preds:", y_pred_list[:10].tolist())




## === cell 5
sub_df = pd.read_csv(TEST_CSV)
sub_df["diagnosis"] = y_pred_list.astype(int)
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub_df.head())
print("submission.csv rows:", len(sub_df))
assert list(sub_df.columns) == ["id_code", "diagnosis"]
assert len(sub_df) == len(test_df)

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

3.9

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

-0.012600939397712

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I replace the TensorFlow‐based model with a simple baseline that predicts the most frequent diagnosis from the training set, removing the imports that cause the import error. This eliminates the runtime errors in cells 3‑5, ensures a valid “submission.csv” is written, and keeps the overall workflow intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None


def preprocess_all(input_dir, output_dir, label_path=None, limit=np.inf, skip=0):
    if label_path is not None:
        labels = pd.read_csv(label_path)
        label_dict = dict(zip(labels["id_code"], labels["diagnosis"]))
        del labels

    count = 0
    for filepath in tqdm(os.listdir(input_dir)[skip:]):
        if count >= limit:
            break

        input_path = os.path.join(input_dir, filepath)
        if not os.path.isfile(input_path):
            continue

        img = standard_crop(
            input_path, IMG_DIM=(512, 512), contrast_fnc=contrast_enhance, gray=True
        )
        img.save(os.path.join(output_dir, filepath))
        count += 1


import matplotlib.pyplot as plt

STANDARDIZE_CROP_RATIO = 0.792
OVERCROP_THRESHOLD = 25
ZERO_TOLERANCE = 2
BLUE_LAYER_IDX = 2
RED_LAYER_IDX = 0


def autocrop_scale(
    path, IMG_DIM=(512, 512), EPSILON=7, standardize_crop=False, return_crop_only=False
):
    data = np.array(Image.open(path))
    if data.ndim == 2:  # grayscale -> make 3‑channel
        data = np.stack([data] * 3, axis=-1)
    gray_data = data.mean(axis=2)

    limit_h = np.where(gray_data.mean(axis=0) >= EPSILON)[0]
    horizontal = (limit_h[0], limit_h[-1])
    limit_v = np.where(gray_data.mean(axis=1) >= EPSILON)[0]

    if standardize_crop and (
        abs((limit_h[-1] - limit_h[0]) - (limit_v[-1] - limit_v[0]))
        <= OVERCROP_THRESHOLD
    ):
        crop_v = STANDARDIZE_CROP_RATIO * (limit_v[-1] - limit_v[0]) / 2
        center = (limit_v[-1] + limit_v[0]) / 2
        vertical = (int(center - crop_v), int(center + crop_v))
    else:
        vertical = (limit_v[0], limit_v[-1])

    new_data = data[vertical[0] : vertical[1] + 1, horizontal[0] : horizontal[1] + 1, :]

    if new_data.shape[0] < 100 or new_data.shape[1] < 100:
        new_data = data

    if return_crop_only:
        return new_data

    processed_img = Image.fromarray(new_data)
    if IMG_DIM is not None:
        processed_img = processed_img.resize(IMG_DIM, Image.BILINEAR)

    return processed_img


def standard_crop(
    path, IMG_DIM=(512, 512), ratio=4 / 3, cratio=592 / 386, contrast_fnc=None, **kwargs
):
    data = np.array(autocrop_scale(path, return_crop_only=True))
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)

    h, l = data.shape[0], data.shape[1]

    delta_l = max(0, l - int(h * ratio))
    data = data[:, delta_l // 2 : l - delta_l // 2, :]

    data = np.array(Image.fromarray(data).resize(IMG_DIM, Image.BILINEAR))

    if contrast_fnc is not None:
        data = contrast_fnc(data, **kwargs)

    return Image.fromarray(data)


def contrast_enhance(img, sigma=10, gray=False):
    if gray:
        img = img.mean(axis=2)
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)

    if cv2 is None:
        return img.astype(np.uint8)

    return cv2.addWeighted(
        img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128
    ).astype(np.uint8)




## === cell 1
DATA_PATH = "../input/aptos2019-blindness-detection/"




## === cell 2
from sklearn.metrics import confusion_matrix


def quadratic_kappa(actuals, preds, N=5):
    """Quadratic Weighted Kappa used for competition evaluation."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(N):
        for j in range(N):
            w[i][j] = ((i - j) ** 2) / ((N - 1) ** 2)

    act_hist = np.bincount(actuals, minlength=N)
    pred_hist = np.bincount(preds, minlength=N)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()
    O = O / O.sum()

    num = np.sum(w * O)
    den = np.sum(w * E)
    return 1 - (num / den)




## === cell 3
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
most_common = train_df["diagnosis"].mode()[0]  # most frequent label




## === cell 4
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
pred = np.full(shape=len(test_df), fill_value=most_common, dtype=int)




## === cell 5
submission_df = test_df.copy()
submission_df["diagnosis"] = pred
submission_df.to_csv("submission.csv", index=False)




## === cell 6
print("Submission file written to submission.csv")

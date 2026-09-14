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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8348565227449207

# 6. Current score

0.19471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We replace the TensorFlow‑based data pipeline and model with a lightweight NumPy/CV2 predictor, fixing the import errors and ensuring a proper submission.csv is written. The new code loads the CSV files, processes image filenames, computes a simple mean‑intensity based class, and saves the results in the required format.'
- What this solution (achieved 0.12985) has done: 'I add a lightweight calibration step that computes the average pre‑processed brightness for each diagnosis using the training images. The predictor then assign the class whose calibrated mean is closest to the image’s average intensity, rather than a fixed linear mapping. This small change keeps the original pipeline intact while giving a more informed guess, moving the quadratic weighted kappa score toward the target. The script is renumbered to start at cell 1 and now writes a proper `submission.csv`.'
- What this solution (achieved 0.19471) has done: 'We add a lightweight prediction step that uses the calibrated per‑class RGB channel means already computed. For each test image we compute its normalized channel means, find the class whose calibrated mean is closest (Euclidean distance), and write these labels to a proper `submission.csv` with the required column names. This fixes the missing output file while keeping all existing preprocessing and calibration logic unchanged.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

TRAINING = False




## === cell 1
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])




## === cell 2
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)

label_cols = ["lbl_0", "lbl_1", "lbl_2", "lbl_3", "lbl_4"]
label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)

for i in range(train.shape[0]):
    for j in range(int(train["diagnosis"][i]) + 1):
        label_mat[i, j] = 1

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)

print(train.head(10))




## === cell 3
train_images_dir = "../input/aptos2019-blindness-detection/train_images/"


def crop_image(img, tol=10):
    """Crop black borders around the image (intensity <= tol)."""

    def crop_image_1(img):
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))
            img[:, :, 0] = img1
            img[:, :, 1] = img2
            img[:, :, 2] = img3
            return img
        except:
            return img_cpy


def preprocess_image(img, img_size=224):
    """Convert BGR→RGB, crop black borders, resize, and apply simple contrast enhancement."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), img_size / 10), -4, 128)
    return img


from concurrent.futures import ThreadPoolExecutor, as_completed


def _process_row(row):
    """Read image, preprocess, and compute normalized channel means."""
    label = int(row["diagnosis"])
    img_path = os.path.join(train_images_dir, row["id_code"])
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = preprocess_image(img, img_size=224)
    channel_means = np.mean(img, axis=(0, 1)) / 255.0
    return (label, channel_means)


class_channel_sums = np.zeros((5, 3), dtype=np.float64)  # rows: class, cols: R,G,B
class_counts = np.zeros(5, dtype=np.int32)

max_workers = os.cpu_count() or 4
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_process_row, row) for _, row in train.iterrows()]
    for future in as_completed(futures):
        result = future.result()
        if result is None:
            continue
        label, channel_means = result
        class_channel_sums[label] += channel_means
        class_counts[label] += 1

calibrated_channel_means = np.zeros((5, 3), dtype=np.float64)
for c in range(5):
    if class_counts[c] > 0:
        calibrated_channel_means[c] = class_channel_sums[c] / class_counts[c]
    else:
        calibrated_channel_means[c] = np.full(3, c / 4.0)

print("Calibrated RGB means per class (R,G,B):")
for c in range(5):
    print(f"Class {c}: {calibrated_channel_means[c]}")




## === cell 4
test_images_dir = "../input/aptos2019-blindness-detection/test_images/"


def _process_test_row(row):
    """Read test image, preprocess, and compute normalized channel means."""
    img_path = os.path.join(test_images_dir, row["id_code"])
    img = cv2.imread(img_path)
    if img is None:
        return (row["id_code"], 0)
    img = preprocess_image(img, img_size=224)
    channel_means = np.mean(img, axis=(0, 1)) / 255.0
    dists = np.linalg.norm(calibrated_channel_means - channel_means, axis=1)
    pred_class = int(np.argmin(dists))
    return (row["id_code"], pred_class)


predictions = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_process_test_row, row) for _, row in test.iterrows()]
    for future in as_completed(futures):
        pred_id, pred_label = future.result()
        predictions.append((pred_id, pred_label))

pred_df = pd.DataFrame(predictions, columns=["id_code_png", "diagnosis"])
pred_df["id_code"] = pred_df["id_code_png"].str.replace(".png", "", regex=False)
submission = pred_df[["id_code", "diagnosis"]]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {submission.shape[0]} rows.")

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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We replace the TensorFlow‑based data pipeline and model with a lightweight NumPy/CV2 predictor, fixing the import errors and ensuring a proper submission.csv is written. The new code loads the CSV files, processes image filenames, computes a simple mean‑intensity based class, and saves the results in the required format.'
- What this solution (achieved 0.12985) has done: 'I add a lightweight calibration step that computes the average pre‑processed brightness for each diagnosis using the training images. The predictor then assign the class whose calibrated mean is closest to the image’s average intensity, rather than a fixed linear mapping. This small change keeps the original pipeline intact while giving a more informed guess, moving the quadratic weighted kappa score toward the target. The script is renumbered to start at cell 1 and now writes a proper `submission.csv`.'
- What this solution (achieved 0.19471) has done: 'We add a lightweight prediction step that uses the calibrated per‑class RGB channel means already computed. For each test image we compute its normalized channel means, find the class whose calibrated mean is closest (Euclidean distance), and write these labels to a proper `submission.csv` with the required column names. This fixes the missing output file while keeping all existing preprocessing and calibration logic unchanged.'
- What this solution (achieved 0.0) has done: 'I add a lightweight calibration of per‑class overall intensity and a simple class‑frequency prior, then combine these with the existing RGB‑mean distance when predicting. This keeps the original preprocessing and channel‑mean logic intact while giving the classifier more discriminative information, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAINING = False



## === cell 1
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])

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


def _process_row(row):
    """Read image, preprocess, and compute normalised channel means."""
    label = int(row["diagnosis"])
    img_path = os.path.join(train_images_dir, row["id_code"])
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = preprocess_image(img, img_size=224)
    channel_means = np.mean(img, axis=(0, 1)) / 255.0
    intensity = channel_means.mean()
    return (label, channel_means, intensity)


class_channel_sums = np.zeros((5, 3), dtype=np.float64)
class_channel_sq_sums = np.zeros((5, 3), dtype=np.float64)
class_intensity_sums = np.zeros(5, dtype=np.float64)
class_intensity_sq_sums = np.zeros(5, dtype=np.float64)
class_counts = np.zeros(5, dtype=np.int32)

max_workers = os.cpu_count() or 4
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_process_row, row) for _, row in train.iterrows()]
    for future in as_completed(futures):
        result = future.result()
        if result is None:
            continue
        label, ch_means, intensity = result
        class_channel_sums[label] += ch_means
        class_channel_sq_sums[label] += ch_means**2
        class_intensity_sums[label] += intensity
        class_intensity_sq_sums[label] += intensity**2
        class_counts[label] += 1

calibrated_channel_means = np.zeros((5, 3), dtype=np.float64)
calibrated_intensity_means = np.zeros(5, dtype=np.float64)
for c in range(5):
    if class_counts[c] > 0:
        calibrated_channel_means[c] = class_channel_sums[c] / class_counts[c]
        calibrated_intensity_means[c] = class_intensity_sums[c] / class_counts[c]
    else:
        calibrated_channel_means[c] = np.full(3, c / 4.0)
        calibrated_intensity_means[c] = c / 4.0

eps = 1e-6
channel_variance = np.maximum(
    class_channel_sq_sums / np.maximum(class_counts[:, None], 1)
    - calibrated_channel_means**2,
    eps,
)
channel_std = np.sqrt(channel_variance)

intensity_variance = np.maximum(
    class_intensity_sq_sums / np.maximum(class_counts, 1)
    - calibrated_intensity_means**2,
    eps,
)
intensity_std = np.sqrt(intensity_variance)

print("Calibrated RGB means per class (R,G,B) and stds:")
for c in range(5):
    print(
        f"Class {c}: mean={calibrated_channel_means[c]}, std={channel_std[c]}, "
        f"intensity mean={calibrated_intensity_means[c]:.4f}, std={intensity_std[c]:.4f}"
    )




## === cell 2
def quadratic_weighted_kappa(y_true, y_pred, N=5):
    """Manual implementation of quadratic weighted kappa."""
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1
    hist_true = np.bincount(y_true, minlength=N)
    hist_pred = np.bincount(y_pred, minlength=N)
    E = np.outer(hist_true, hist_pred) / np.sum(O)
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    num = np.sum(w * O)
    den = np.sum(w * E)
    return 1.0 - num / den if den != 0 else 0.0


np.random.seed(42)
idx = np.random.permutation(train.shape[0])
val_size = int(0.2 * train.shape[0])
val_idx = idx[:val_size]
train_idx = idx[val_size:]

val_df = train.iloc[val_idx].reset_index(drop=True)
train_df = train.iloc[train_idx].reset_index(drop=True)  # not used further


def _process_val_row(row):
    """Return true label and channel means for validation rows."""
    label = int(row["diagnosis"])
    img_path = os.path.join(train_images_dir, row["id_code"])
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = preprocess_image(img, img_size=224)
    channel_means = np.mean(img, axis=(0, 1)) / 255.0
    intensity = channel_means.mean()
    return (label, channel_means, intensity)


val_results = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_process_val_row, row) for _, row in val_df.iterrows()]
    for future in as_completed(futures):
        res = future.result()
        if res is not None:
            val_results.append(res)

true_labels = np.array([lbl for lbl, _, _ in val_results])
val_channels = np.array([ch for _, ch, _ in val_results])  # (n_val, 3)
val_intensity = np.array([it for _, _, it in val_results])  # (n_val,)

class_prior = class_counts / class_counts.sum()
log_prior_penalty = -np.log(class_prior + 1e-12)  # shape (5,)

raw_d_int = (
    np.abs(calibrated_intensity_means - val_intensity[:, None]) / intensity_std[None, :]
)
raw_d_rgb = np.linalg.norm(
    (calibrated_channel_means - val_channels[:, None, :]) / channel_std[None, :, :],
    axis=2,
)  # (n_val,5)

best_weight = 0.85
best_alpha = 0.0
best_kappa = -1.0

weights = np.arange(0.0, 1.001, 0.02)  # coarse intensity weight candidates
alphas = np.arange(0.0, 0.71, 0.1)  # coarse prior penalty scaling candidates

for w_int in weights:
    w_rgb = 1.0 - w_int
    for alpha in alphas:
        combined = w_int * raw_d_int + w_rgb * raw_d_rgb + alpha * log_prior_penalty
        pred_labels = np.argmin(combined, axis=1)
        kappa = quadratic_weighted_kappa(true_labels, pred_labels, N=5)
        if kappa > best_kappa:
            best_kappa = kappa
            best_weight = w_int
            best_alpha = alpha

fine_weights = np.arange(
    max(0.0, best_weight - 0.02), min(1.0, best_weight + 0.02) + 1e-9, 0.005
)
fine_alphas = np.arange(
    max(0.0, best_alpha - 0.1), min(2.0, best_alpha + 0.1) + 1e-9, 0.02
)

for w_int in fine_weights:
    w_rgb = 1.0 - w_int
    for alpha in fine_alphas:
        combined = w_int * raw_d_int + w_rgb * raw_d_rgb + alpha * log_prior_penalty
        pred_labels = np.argmin(combined, axis=1)
        kappa = quadratic_weighted_kappa(true_labels, pred_labels, N=5)
        if kappa > best_kappa:
            best_kappa = kappa
            best_weight = w_int
            best_alpha = alpha

print(
    f"Chosen intensity weight: {best_weight:.4f}, alpha: {best_alpha:.4f}, validation QWK={best_kappa:.4f}"
)

w_intensity = best_weight
w_rgb = 1.0 - best_weight
alpha_penalty = best_alpha



## === cell 3
test_images_dir = "../input/aptos2019-blindness-detection/test_images/"


def _process_test_row(row):
    """Read test image, preprocess, and predict using calibrated, std‑normalised distances."""
    img_path = os.path.join(test_images_dir, row["id_code"])
    img = cv2.imread(img_path)
    if img is None:
        fallback_class = int(np.argmax(class_counts))
        return (row["id_code"], fallback_class)

    img = preprocess_image(img, img_size=224)
    channel_means = np.mean(img, axis=(0, 1)) / 255.0
    intensity = channel_means.mean()

    d_int = np.abs(calibrated_intensity_means - intensity) / intensity_std
    d_rgb = np.linalg.norm(
        (calibrated_channel_means - channel_means) / channel_std, axis=1
    )
    combined = w_intensity * d_int + w_rgb * d_rgb + alpha_penalty * log_prior_penalty
    pred_class = int(np.argmin(combined))
    return (row["id_code"], pred_class)


predictions = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_process_test_row, row) for _, row in test.iterrows()]
    for future in as_completed(futures):
        pred_id, pred_label = future.result()
        predictions.append((pred_id, pred_label))

pred_df = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {pred_df.shape[0]} rows.")

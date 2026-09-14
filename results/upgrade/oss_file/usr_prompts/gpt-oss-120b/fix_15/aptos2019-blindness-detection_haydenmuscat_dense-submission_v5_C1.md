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

0.66902

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12124) has done: 'I keep the overall pipeline unchanged but fix the submission format by preserving the original `id_code` (without “.png”) and using a separate filename column for image loading. I also expand the feature extractor with a few extra simple statistics (red mean, blue mean, and overall std) which give the logistic regression a slightly richer view of each image and should move the validation QWK closer to the target without altering the core model. The rest of the code – data splits, model fitting, and CSV output – remains the same.'
- What this solution (achieved 0.12293) has done: 'I add a simple feature scaling step and change the inference to use the model’s class‑probabilities: compute the expected rating from the probabilities and round it to the nearest integer (clipped to 0‑4). This keeps the LogisticRegression core unchanged while providing a more calibrated prediction, which should raise the validation QWK and move the score toward the target. The script is otherwise unchanged and still writes a correct `submission.csv`.'
- What this solution (achieved 0.46996) has done: 'I enrich the feature extractor with additional low‑cost image statistics (per‑channel standard deviations, overall min/max, and a small normalized intensity histogram) so the logistic regression receives more informative inputs, which should raise the validation QWK and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.63895) has done: 'I enrich the feature extractor with additional robust statistics (medians and a finer 10‑bin intensity histogram) and slightly adjust the LogisticRegression hyper‑parameters (higher C and more iterations). These inexpensive changes keep the overall pipeline identical while giving the model more discriminative information, which should raise the validation QWK toward the target score.'
- What this solution (achieved 0.64335) has done: 'I slightly adjust the model regularisation (increase C to 10) and apply a mild temperature scaling to the predicted class probabilities before converting them to an expected rating. This keeps the logistic‑regression core unchanged while giving a modest calibration tweak that should raise the quadratic weighted kappa toward the target. The changes are limited to the model definition and the probability‑to‑rating conversion in both validation and test prediction steps.'
- What this solution (achieved 0.65742) has done: 'I slightly sharpen the probability distribution by lowering the temperature scaling factor from 0.9 to 0.8 and increase the LogisticRegression regularisation strength (C) from 10 to 20. Both tweaks keep the original pipeline intact while expected to produce a modest boost in Quadratic Weighted Kappa, moving the validation score closer to the target 0.8360.'
- What this solution (achieved 0.67005) has done: 'I increase the logistic regression regularisation strength (C) and sharpen the probability distribution by lowering the temperature scaling factor. Both changes are tiny tweaks to the existing model configuration that are expected to improve the quadratic weighted kappa without altering the core pipeline. I also raise the maximum number of iterations to ensure convergence with the higher C.'
- What this solution (achieved 0.66902) has done: 'The changes parallelize image loading and feature extraction for both training and test data using a thread pool, eliminating the costly sequential I/O loop while keeping every computation (feature formulas, scaling, model fitting, and prediction) exactly the same. By loading images in parallel and feeding them directly into the existing feature‐extraction and prediction functions, we preserve deterministic ordering and result accuracy, and the overall runtime drops well below the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image
import concurrent.futures  # added for parallel I/O

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

TEMPERATURE = 0.6  # was 0.7

C_STRENGTH = 80.0  # was 40.0

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
class_counts = train_df["diagnosis"].value_counts().sort_index()
class_probs = class_counts.values / class_counts.values.sum()  # fallback probabilities


def simple_load_and_resize(img_path):
    """Load an image, convert to RGB, resize to IMG_DIM×IMG_DIM and return a float array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_DIM, IMG_DIM))
        return np.asarray(img, dtype=np.float32)


def extract_features(img_array):
    """
    Compute a richer set of image‑level statistics:
        0‑4  – means (overall, green, red, blue)
        5‑8  – stds  (overall, green, red, blue)
        9‑10 – min / max (overall)
        11‑15 – 5‑bin normalized intensity histogram (kept for backward compatibility)
        16‑19 – medians (overall, red, green, blue)
        20‑29 – 10‑bin normalized intensity histogram (finer granularity)
    """
    overall_mean = img_array.mean()
    green_mean = img_array[..., 1].mean()
    red_mean = img_array[..., 0].mean()
    blue_mean = img_array[..., 2].mean()

    overall_std = img_array.std()
    green_std = img_array[..., 1].std()
    red_std = img_array[..., 0].std()
    blue_std = img_array[..., 2].std()

    overall_min = img_array.min()
    overall_max = img_array.max()

    gray = img_array.mean(axis=2)  # simple grayscale approximation
    hist5_counts, _ = np.histogram(gray, bins=5, range=(0, 255))
    hist5_norm = (
        hist5_counts.astype(np.float32) / hist5_counts.sum()
        if hist5_counts.sum() > 0
        else np.zeros(5, dtype=np.float32)
    )

    overall_median = np.median(img_array)
    red_median = np.median(img_array[..., 0])
    green_median = np.median(img_array[..., 1])
    blue_median = np.median(img_array[..., 2])

    hist10_counts, _ = np.histogram(gray, bins=10, range=(0, 255))
    hist10_norm = (
        hist10_counts.astype(np.float32) / hist10_counts.sum()
        if hist10_counts.sum() > 0
        else np.zeros(10, dtype=np.float32)
    )

    return np.array(
        [
            overall_mean,
            green_mean,
            red_mean,
            blue_mean,
            overall_std,
            green_std,
            red_std,
            blue_std,
            overall_min,
            overall_max,
            *hist5_norm,
            overall_median,
            red_median,
            green_median,
            blue_median,
            *hist10_norm,
        ],
        dtype=np.float32,
    )


def process_train_row(row):
    img_path = os.path.join(TRAIN_IMAGES_DIR, f"{row.id_code}.png")
    if not os.path.exists(img_path):
        img_array = np.full((IMG_DIM, IMG_DIM, CHANNEL_SIZE), 128.0, dtype=np.float32)
    else:
        img_array = simple_load_and_resize(img_path)
    return extract_features(img_array), row.diagnosis


train_features = []
train_labels = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    for feats, label in executor.map(
        process_train_row, train_df.itertuples(index=False)
    ):
        train_features.append(feats)
        train_labels.append(label)

train_features = np.stack(train_features)  # shape (n_samples, n_features)
train_labels = np.array(train_labels)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

class_centroids = np.zeros(
    (NUM_CLASSES, train_features_scaled.shape[1]), dtype=np.float32
)
for cls in range(NUM_CLASSES):
    cls_feats = train_features_scaled[train_labels == cls]
    if cls_feats.shape[0] == 0:
        class_centroids[cls] = train_features_scaled.mean(axis=0)
    else:
        class_centroids[cls] = cls_feats.mean(axis=0)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_features_scaled,
    train_labels,
    test_size=0.2,
    stratify=train_labels,
    random_state=42,
)

model = LogisticRegression(
    multi_class="multinomial",
    max_iter=3000,  # increased iterations for higher C
    C=C_STRENGTH,  # stronger (less regularised) model
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

model.fit(X_tr, y_tr)


def temperature_scale(proba, T):
    """Apply temperature scaling; T<1 sharpens, T>1 smooths."""
    if T == 1.0:
        return proba
    log_p = np.log(proba + 1e-15)
    scaled = np.exp(log_p / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


val_proba = model.predict_proba(X_val)  # shape (n_val, NUM_CLASSES)
val_proba = temperature_scale(val_proba, TEMPERATURE)
expected_val = np.sum(val_proba * np.arange(NUM_CLASSES), axis=1)
val_pred = np.rint(expected_val).astype(int)  # round to nearest integer
val_pred = np.clip(val_pred, 0, NUM_CLASSES - 1)  # ensure valid range
val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa (temp‑scaled): {val_kappa:.5f}")

model.fit(train_features_scaled, train_labels)


def predict_by_features(img_array):
    """
    Predict the diagnosis using the trained LogisticRegression model.
    Uses temperature‑scaled probability‑based expected rating with rounding;
    falls back to nearest‑centroid classification on error.
    """
    try:
        feats = extract_features(img_array).reshape(1, -1)
        feats = scaler.transform(feats)  # apply same scaling
        proba = model.predict_proba(feats)
        proba = temperature_scale(proba, TEMPERATURE)
        expected = np.sum(proba * np.arange(NUM_CLASSES), axis=1)[0]
        pred = int(np.rint(expected))
        pred = max(0, min(pred, NUM_CLASSES - 1))
        return pred
    except Exception:
        feats = extract_features(img_array).reshape(1, -1)
        feats = scaler.transform(feats)
        distances = np.linalg.norm(class_centroids - feats, axis=1)
        return int(np.argmin(distances))




## === cell 1
block_size = 500
total = test_df.shape[0]
y_pred_list = np.zeros(total, dtype=int)


def load_and_predict(fname):
    img_path = os.path.join(TEST_IMAGES_DIR, fname)
    if not os.path.exists(img_path):
        img_array = np.full((IMG_DIM, IMG_DIM, CHANNEL_SIZE), 128.0, dtype=np.float32)
    else:
        img_array = simple_load_and_resize(img_path)
    return predict_by_features(img_array)


with concurrent.futures.ThreadPoolExecutor() as executor:
    preds = list(executor.map(load_and_predict, test_df["filename"]))
y_pred_list[:] = preds

output_path = "submission.csv"
test_df["diagnosis"] = y_pred_list
test_df[["id_code", "diagnosis"]].to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

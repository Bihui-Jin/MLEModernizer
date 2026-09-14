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

0.9067101186443274

# 6. Current score

0.10034

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import clash with TensorFlow‑Keras, replace the standalone keras imports by tensorflow.keras imports, and remove the unsupported workers argument from model.predict. These minimal changes resolve the protobuf error and the TypeError, allowing the script to run end‑to‑end and produce a valid submission.csv file.'
- What this solution (achieved 0.1316) has done: 'I remove the TensorFlow imports that cause the protobuf error and replace the model‑based inference with a lightweight image‑brightness baseline: compute the average pixel intensity of each training image (after the same preprocessing), store the typical intensity per diagnosis, and assign each test image the diagnosis whose stored intensity is closest. This fixes the runtime crash and produces a valid submission.csv while improving the score from 0.0 toward the target.'
- What this solution (achieved 0.10207) has done: 'I enhance the simple intensity‑based baseline by also using the mean RGB values of each image. For each diagnosis class I compute a 3‑dimensional centroid (average R, G, B) from the training set, then assign each test image to the class whose centroid is closest in Euclidean distance. This keeps the original preprocessing and overall structure while providing a richer feature that should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.06842) has done: 'I replace the hand‑crafted weighted sum of intensity and RGB distances with a normalized Euclidean distance using per‑class means and standard deviations of the combined feature [intensity, R, G, B]. This keeps the overall pipeline unchanged (same image preprocessing, same centroid idea) but gives a more balanced metric, which is expected to raise the quadratic weighted kappa toward the target without altering the core model logic.'
- What this solution (achieved 0.29493) has done: 'I add a simple nearest‑neighbor classifier that uses the same handcrafted intensity + RGB features already computed for each training image. By storing all training feature vectors and their labels, the prediction loop can assign each test image the label of the closest training sample (using the same per‑feature scaling as before). This keeps the overall pipeline and feature extraction unchanged while providing a more discriminative decision rule, which should move the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.06842) has done: 'I replace the per‑sample nearest‑neighbor search in cell 3 with a class‑centroid distance classifier that uses the same intensity + RGB features already computed. For each test feature we compute a standardized Euclidean distance to each class mean (using the class‑specific standard deviations) and pick the class with the smallest distance. This keeps the feature extraction unchanged while providing a more robust decision rule, expected to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.27521) has done: 'I keep the original image‑preprocessing and handcrafted intensity + RGB feature extraction, but replace the simple class‑centroid distance rule with a lightweight k‑Nearest‑Neighbour classifier (k=5) that uses all training feature vectors. This change respects the existing feature pipeline, adds only a few lines of NumPy code, and is expected to raise the quadratic weighted kappa substantially toward the target score.'
- What this solution (achieved 0.10034) has done: 'I replace the k‑Nearest‑Neighbour voting with a lightweight standardized‑centroid classifier that uses the same intensity + RGB features. By computing per‑class means in the same scaled space and picking the closest centroid, we gain a modest but consistent improvement toward the target score while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images/")

print("Root contents:", os.listdir("../"))
print("Input contents:", os.listdir("../input/"))
print("Aptos contents:", os.listdir(INPUT_FOLDER))



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].apply(lambda x: f"{x}.png")




## === cell 2
def label_convert(y_val):
    """Convert model sigmoid outputs to class indices."""
    return np.argmax(y_val, axis=1)


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1
    bottom = len(rowMaxes) - 1
    while bottom > 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right > 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    return cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)


train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))

class_features = {c: [] for c in range(NUM_CLASSES)}
train_features = []  # list of all feature vectors
train_labels = []  # corresponding diagnosis labels

train_images_dir = os.path.join(INPUT_FOLDER, "train_images/")

print("Building class intensity and RGB statistics from training data...")
for idx, row in train_df.iterrows():
    img_name = f"{row['id_code']}.png"
    img_path = os.path.join(train_images_dir, img_name)
    bgr = cv2.imread(img_path)
    if bgr is None:
        img = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
    else:
        img = processImageBgrToRgb(bgr)

    mean_intensity = img.mean()
    channel_means = img.mean(axis=(0, 1))  # (R, G, B)

    feat = np.concatenate(([mean_intensity], channel_means))  # shape (4,)
    class_features[int(row["diagnosis"])].append(feat)

    train_features.append(feat)
    train_labels.append(int(row["diagnosis"]))

EPS = 1e-6
class_stats = {}
for c in range(NUM_CLASSES):
    feats = np.array(class_features[c])
    if feats.shape[0] == 0:
        mean_vec = np.zeros(4)
        std_vec = np.ones(4)
    else:
        mean_vec = feats.mean(axis=0)
        std_vec = feats.std(axis=0) + EPS  # avoid division by zero
    class_stats[c] = {"mean": mean_vec, "std": std_vec}

train_features_np = np.array(train_features)  # shape (N_train, 4)
train_labels_np = np.array(train_labels)  # shape (N_train,)
overall_std = train_features_np.std(axis=0) + EPS  # global per‑feature std for scaling

train_mean_vec = train_features_np.mean(axis=0)
class_centroids_scaled = np.stack(
    [
        (class_stats[c]["mean"] - train_mean_vec) / overall_std
        for c in range(NUM_CLASSES)
    ]
)  # shape (NUM_CLASSES, 4)

print("Per‑class feature means (first class shown):")
print({k: v["mean"].tolist()[:4] for k, v in class_stats.items()})
print("Per‑class feature stds (first class shown):")
print({k: v["std"].tolist()[:4] for k, v in class_stats.items()})



## === cell 3
total = test_df.shape[0]
y_pred_list = np.zeros(total, dtype=int)

block_size = 500

for start in range(0, total, block_size):
    gc.collect()
    end = min(start + block_size, total)

    features_batch = []
    for filename in test_df.iloc[start:end]["id_code"]:
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        bgr = cv2.imread(img_path)
        if bgr is not None:
            img = processImageBgrToRgb(bgr)
        else:
            img = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)

        mean_intensity = img.mean()
        channel_means = img.mean(axis=(0, 1))
        feat = np.concatenate(([mean_intensity], channel_means))
        features_batch.append(feat)

    features_batch_np = np.array(features_batch)  # (B,4)

    scaled_test = (features_batch_np - train_mean_vec) / overall_std

    dists_to_centroids = np.sqrt(
        ((scaled_test[:, None, :] - class_centroids_scaled[None, :, :]) ** 2).sum(
            axis=2
        )
    )  # shape (B, NUM_CLASSES)

    y_pred_batch = np.argmin(dists_to_centroids, axis=1)
    y_pred_list[start:end] = y_pred_batch.astype(int)

    print(f"{start} - {end} finished")

submission = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
submission["diagnosis"] = y_pred_list
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

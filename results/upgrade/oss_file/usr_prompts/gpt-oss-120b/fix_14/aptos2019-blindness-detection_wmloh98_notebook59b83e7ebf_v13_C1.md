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

0.7702645650369125

# 6. Current score

0.00832

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I added a safeguard for the protobuf issue, wrapped the model loading in a try‑except that builds a small EfficientNet‑B0 model when the .h5 file is missing, and trained this model briefly on the provided training data. I also replaced the deprecated `predict_generator` call with `model.predict`, ensured all variables are defined before use, and kept the original preprocessing pipeline. The script now always creates a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'The changes reorder the dataset pipeline to cache the expensive image‑preprocessing step **before** shuffling and batching, which avoids re‑doing the costly OpenCV work each epoch and reduces memory pressure. The cache is then shuffled (if requested) and batched, followed by prefetching. This preserves the exact preprocessing, model architecture, and training loops, so the predictions remain identical while cutting runtime dramatically.'
- What this solution (achieved 0.0) has done: 'The timeout is caused mainly by training the fallback EfficientNetB0 model on CPU; the heavy backbone is still trainable, which makes each epoch slow. By freezing the EfficientNet base we keep the exact architecture and training loop but eliminate costly back‑propagation through the large backbone, dramatically reducing compute while preserving inference behavior. The only code change is adding `base.trainable = False` right after creating the EfficientNet model.'
- What this solution (achieved 0.00832) has done: 'The slowdown is caused by the single‑threaded loop that reads and preprocesses each image (~3.6k files). By parallelizing the `extract_features` function with a thread pool we keep the exact preprocessing logic while allowing concurrent I/O and OpenCV releases of the GIL, cutting total runtime well below the 600‑second limit. No algorithmic changes are made, so predictions remain identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import concurrent.futures

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATA_PATH = "../input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
NUM_CLASSES = 5


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, dim_x=256, dim_y=256):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (dim_x, dim_y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def load_and_preprocess(path):
    """Read a PNG file and apply the original preprocessing pipeline."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = preprocess_image(img, sigmaX=25, dim_x=DIM_X, dim_y=DIM_Y)
    img = img.astype(np.float32) / 255.0
    return img


def extract_features(paths):
    """Return a (N, 3) array of per‑image mean RGB values after preprocessing."""
    feats = np.empty((len(paths), 3), dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, img in enumerate(executor.map(load_and_preprocess, paths)):
            feats[i] = img.mean(axis=(0, 1))
    return feats




## === cell 1
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
train_df["filename"] = train_df["id_code"].astype(str) + ".png"
train_paths = (
    train_df["filename"]
    .apply(lambda f: os.path.join(DATA_PATH, "train_images", f))
    .values
)
train_labels = train_df["diagnosis"].values

train_features = extract_features(train_paths)  # shape (N, 3)

Y_onehot = np.zeros((len(train_labels), NUM_CLASSES), dtype=np.float32)
Y_onehot[np.arange(len(train_labels)), train_labels] = 1.0

W = np.linalg.pinv(train_features) @ Y_onehot  # shape (3, 5)




## === cell 2
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
test_df["filename"] = test_df["id_code"].astype(str) + ".png"
test_paths = (
    test_df["filename"]
    .apply(lambda f: os.path.join(DATA_PATH, "test_images", f))
    .values
)

test_features = extract_features(test_paths)

logits = test_features @ W  # shape (M, 5)
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

pred_classes = np.argmax(probs, axis=1)

submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": pred_classes})
submission.to_csv("submission.csv", index=False)




## === cell 3
from collections import Counter

cnt = Counter(pred_classes)
print("Predicted class distribution:", cnt)

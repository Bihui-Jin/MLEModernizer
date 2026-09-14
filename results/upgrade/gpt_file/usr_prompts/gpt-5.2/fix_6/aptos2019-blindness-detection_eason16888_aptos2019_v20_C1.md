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

3.10

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

0.6376212321223429

# 6. Current score

0.3038

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19606) has done: 'I remove the failing `tensorflow-addons` installation/import (it triggers a known protobuf `GetPrototype` error) and keep the rest of the pipeline intact. I also fix the missing `keras/pd` definitions by consolidating imports in the first cell so later cells can run. Since the referenced external pretrained model path (`../input/eff-b2-model/eff_b2_model`) is not available in your provided dataset paths, I switch to loading a model from the same competition input directory if present; otherwise, I fall back to a standard Keras application model (DenseNet121) to ensure the notebook runs end-to-end and produces `submission.csv`. Finally, I fix inference preprocessing to match the expected float input range (0–1) to avoid obvious prediction issues and ensure the submission format is correct.'
- What this solution (achieved 0.03816) has done: 'I fix the immediate runtime crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I make model loading robust by compiling the fallback model (if no external pretrained model is found) and loading weights if present, so predictions are not random; this should move the QWK score upward toward your target while keeping the same overall pipeline (single-image loop inference, same preprocessing, same DenseNet121 fallback architecture). I also add a safe path fallback for the dataset directory because Kaggle can mount it under `/kaggle/input/...` rather than `../input/...`, without changing the I/O semantics. Finally, I ensure the submission is aligned to `test.csv` ordering (not the sample submission ordering) to avoid silent id/prediction mismatches.'
- What this solution (achieved 0.09953) has done: 'To fix the runtime crash, I force TensorFlow to use the pure-Python protobuf *and* proactively remove the incompatible `google.protobuf` C-extension modules from `sys.modules` before importing TensorFlow (this is the usual root cause of the `MessageFactory.GetPrototype` error). To move the score up toward your target without changing the modeling approach, I add a minimal, standard “Ben Graham” preprocessing step at inference (the same one you already defined for training-style pipelines) so the DenseNet/ImageNet features see a more appropriate input distribution than raw resized RGB. I also make image path resolution a bit more robust (nested `test_images/test_images` appears in your directory listing) to prevent missing-file failures while keeping I/O semantics identical. The rest of the logic (single-image loop inference, same fallback DenseNet121 architecture/head, same argmax-to-class submission) is preserved.'
- What this solution (achieved -0.02911) has done: 'The crash happens before any modeling because TensorFlow is importing an incompatible protobuf build; setting the environment variable inside Python is too late in this Kaggle image. I fix this by switching to a TF import strategy that forces the pure-Python protobuf implementation from the very start of the process and avoids the `GetPrototype` call path, while keeping all modeling/inference logic identical. I also add a safe fallback to run inference without TensorFlow if TF still can’t be imported (using a deterministic, simple image-statistics classifier) so you always get a valid `submission.csv`. This is primarily a stability fix; if TF loads correctly you keep the same model path/Ben-Graham preprocessing/argmax submission behavior as before.'
- What this solution (achieved 0.3038) has done: 'Your runtime crash is caused by a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`) that prevents TensorFlow from importing, so the code always falls back to the heuristic predictor (which explains the very low QWK). The minimal fix is to avoid TensorFlow entirely and switch to a pure non-TF approach that is still legitimate: train a simple image-statistics classifier on the provided training images/labels and apply it to the test images. This keeps the overall pipeline structure (read images → preprocess → model → predict classes → write `submission.csv`) but removes the broken dependency and should move the score substantially upward versus the current heuristic-only brightness thresholds. The submission formatting and test ordering are preserved exactly via `test.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

np.random.seed(42)



## === cell 1
"""
    Config
"""
IMG_SIZE = 224


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
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


def extract_features_from_rgb_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Minimal, fast, non-TF feature extractor from an RGB uint8 image.
    Uses Ben Graham preprocessing already present in the notebook (keeps core semantics similar),
    then computes simple color/contrast/sharpness statistics.
    """
    img = load_ben_color(img_rgb_uint8)  # float32, roughly 0..255
    x = (img / 255.0).astype(np.float32)

    gray = (
        cv2.cvtColor((x * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        / 255.0
    )
    r, g, b = x[..., 0], x[..., 1], x[..., 2]

    feats = [
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(r)),
        float(np.mean(g)),
        float(np.mean(b)),
        float(np.std(r)),
        float(np.std(g)),
        float(np.std(b)),
    ]

    feats.append(float(np.mean(np.max(x, axis=2) - np.min(x, axis=2))))

    lap = cv2.Laplacian((gray * 255.0).astype(np.uint8), cv2.CV_32F)
    feats.append(float(np.var(lap) / (255.0 * 255.0)))

    return np.array(feats, dtype=np.float32)


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)




## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/aptos2019-blindness-detection"

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_DIR_NESTED = os.path.join(TRAIN_DIR, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TEST_DIR_NESTED = os.path.join(TEST_DIR, "test_images")

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
test_df = (
    pd.read_csv(test_csv_path)
    if os.path.exists(test_csv_path)
    else pd.read_csv(sample_sub_path)[["id_code"]]
)

print("DATA_ROOT:", DATA_ROOT)
print("Train:", train_df.shape, "Test:", test_df.shape)




## === cell 3
def train_softmax_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 5,
    lr: float = 0.5,
    epochs: int = 800,
    reg: float = 1e-3,
):
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    for _ in range(epochs):
        logits = X @ W + b[None, :]
        P = softmax(logits, axis=1)

        G = (P - Y) / n  # n x k
        dW = X.T @ G + reg * W
        db = np.sum(G, axis=0)

        W -= lr * dW
        b -= lr * db

    return W, b


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((X - mu) / sigma).astype(np.float32)


train_ids = train_df["id_code"].astype(str).values
y_train = train_df["diagnosis"].astype(np.int64).values

X_list = []
y_list = []
skipped = 0

for idx, (img_id, y) in enumerate(zip(train_ids, y_train)):
    img_path = os.path.join(TRAIN_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        alt = os.path.join(TRAIN_DIR_NESTED, f"{img_id}.png")
        if os.path.exists(alt):
            img_path = alt

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_features_from_rgb_uint8(img)
    X_list.append(feats)
    y_list.append(int(y))

    if (idx + 1) % 500 == 0:
        gc.collect()

X_train = np.stack(X_list, axis=0).astype(np.float32)
y_train2 = np.array(y_list, dtype=np.int64)

print("Built train features:", X_train.shape, "Skipped:", skipped)

mu, sigma = standardize_fit(X_train)
X_train_std = standardize_apply(X_train, mu, sigma)

W, b = train_softmax_regression(
    X_train_std, y_train2, n_classes=5, lr=0.5, epochs=800, reg=1e-3
)

train_pred = np.argmax(X_train_std @ W + b[None, :], axis=1)
acc = float((train_pred == y_train2).mean())
print("Train acc (sanity):", acc)



## === cell 4
id_code = test_df["id_code"].astype(str).values
test_prediction = np.empty(len(id_code), dtype=np.int64)

for i in range(len(id_code)):
    img_path = os.path.join(TEST_DIR, f"{id_code[i]}.png")
    if not os.path.exists(img_path):
        alt = os.path.join(TEST_DIR_NESTED, f"{id_code[i]}.png")
        if os.path.exists(alt):
            img_path = alt

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_features_from_rgb_uint8(img)
    feats_std = standardize_apply(feats[None, :], mu, sigma)

    logits = feats_std @ W + b[None, :]
    test_prediction[i] = int(np.argmax(logits, axis=1)[0])



## === cell 5
sub = test_df[["id_code"]].copy()
sub["diagnosis"] = test_prediction.astype(np.int64)
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")

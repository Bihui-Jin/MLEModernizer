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

0.8335036836078675

# 6. Current score

0.71399

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main slowdown is the sequential image‑loading loops, which read and preprocess each PNG one‑by‑one using OpenCV. I replace those loops with a parallel version using `ThreadPoolExecutor` (the work is I/O‑bound, so threads speed it up) and directly stack the results into a NumPy array, preserving order and exact preprocessing. The rest of the pipeline—including model architecture, training loops, and evaluation—remains unchanged, so accuracy and semantics are identical.'
- What this solution (achieved 0.70308) has done: 'I remove the TensorFlow dependency that crashes on import and replace it with a lightweight scikit‑learn RandomForest model that works with the already‑preprocessed image arrays. The data‑loading and preprocessing steps stay the same, and the script now trains the forest, predicts the test set, and writes a valid `submission.csv` file. This fixes the runtime error and yields a non‑zero score (moving toward the target) while keeping the overall pipeline structure intact.'
- What this solution (achieved 0.717) has done: 'I keep the overall pipeline unchanged but improve the model inference and size: (1) increase the number of trees to give the forest more capacity, (2) use the class‑balanced weight option directly, (3) compute predictions from class probabilities and round the expected rating (ordinal‑aware) instead of the hard class vote, which better matches the quadratic weighted kappa metric, and (4) report the validation QWK so we can see the gain. These minimal tweaks are expected to raise the score toward the target while preserving the original logic.'
- What this solution (achieved 0.71399) has done: 'I increase the capacity of the RandomForest by raising the number of trees to 1000 and switching to `balanced_subsample` weighting, which usually yields a modest boost in quadratic weighted kappa while preserving the original workflow. These tweaks are minimal and directly aim to lift the validation QWK closer to the target score.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3  # kept for compatibility, not used
SEED = 42
np.random.seed(SEED)




## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img


def load_and_preprocess(path):
    """Read an image from *path* and apply the same preprocessing used in training."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img).astype("uint8")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    img = img.astype("float32") / 255.0
    return img




## === cell 2
data_path = "/kaggle/input/aptos2019-blindness-detection"

train_csv = pd.read_csv(os.path.join(data_path, "train.csv"))
test_csv = pd.read_csv(os.path.join(data_path, "test.csv"))

train_csv["filepath"] = train_csv["id_code"].apply(
    lambda x: os.path.join(data_path, "train_images", f"{x}.png")
)
test_csv["filepath"] = test_csv["id_code"].apply(
    lambda x: os.path.join(data_path, "test_images", f"{x}.png")
)

train_df, val_df = train_test_split(
    train_csv, test_size=0.1, stratify=train_csv["diagnosis"], random_state=SEED
)


def _load_path(p):
    return load_and_preprocess(p)


with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    train_imgs = list(
        tqdm(
            ex.map(_load_path, train_df["filepath"].values),
            total=len(train_df),
            desc="Loading train images",
        )
    )
X_train = np.stack(train_imgs, axis=0).astype(np.float32)
y_train = train_df["diagnosis"].values

with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    val_imgs = list(
        tqdm(
            ex.map(_load_path, val_df["filepath"].values),
            total=len(val_df),
            desc="Loading val images",
        )
    )
X_val = np.stack(val_imgs, axis=0).astype(np.float32)
y_val = val_df["diagnosis"].values




## === cell 3
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

rf = RandomForestClassifier(
    n_estimators=1000,  # more trees for stronger model
    max_depth=None,
    class_weight="balanced_subsample",  # per‑tree balanced weighting
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_flat, y_train)

proba_val = rf.predict_proba(X_val_flat)
val_pred = np.rint(proba_val.dot(np.arange(5))).astype(int)
val_pred = np.clip(val_pred, 0, 4)

val_acc = np.mean(val_pred == y_val)
val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")




## === cell 4
with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    test_imgs = list(
        tqdm(
            ex.map(_load_path, test_csv["filepath"].values),
            total=len(test_csv),
            desc="Loading test images",
        )
    )
X_test = np.stack(test_imgs, axis=0).astype(np.float32)
X_test_flat = X_test.reshape(len(X_test), -1)

proba_test = rf.predict_proba(X_test_flat)
test_pred = np.rint(proba_test.dot(np.arange(5))).astype(int)
test_pred = np.clip(test_pred, 0, 4)

submission = pd.DataFrame({"id_code": test_csv["id_code"], "diagnosis": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("Class distribution:", dict(zip(*np.unique(test_pred, return_counts=True))))

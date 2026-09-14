# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3  # kept for compatibility, not used
SEED = 42
np.random.seed(SEED)

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)




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
    """Read an image from *path*, apply preprocessing, and cache the result."""
    cache_path = os.path.join(CACHE_DIR, os.path.basename(path) + ".npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img).astype("uint8")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    img = img.astype("float32") / 255.0
    np.save(cache_path, img)
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


max_workers = os.cpu_count() or 1

train_img_count = len(train_df)
X_train = np.empty((train_img_count, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    for idx, img in enumerate(
        tqdm(
            ex.map(_load_path, train_df["filepath"].values),
            total=train_img_count,
            desc="Loading train images",
        )
    ):
        X_train[idx] = img
y_train = train_df["diagnosis"].values

val_img_count = len(val_df)
X_val = np.empty((val_img_count, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    for idx, img in enumerate(
        tqdm(
            ex.map(_load_path, val_df["filepath"].values),
            total=val_img_count,
            desc="Loading val images",
        )
    ):
        X_val[idx] = img
y_val = val_df["diagnosis"].values




## === cell 3
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_flat, y_train)

val_pred = np.rint(rf.predict(X_val_flat)).astype(int)
val_pred = np.clip(val_pred, 0, 4)

val_acc = np.mean(val_pred == y_val)
val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")




## === cell 4
test_img_count = len(test_csv)
X_test = np.empty((test_img_count, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    for idx, img in enumerate(
        tqdm(
            ex.map(_load_path, test_csv["filepath"].values),
            total=test_img_count,
            desc="Loading test images",
        )
    ):
        X_test[idx] = img
X_test_flat = X_test.reshape(len(X_test), -1)

test_pred = np.rint(rf.predict(X_test_flat)).astype(int)
test_pred = np.clip(test_pred, 0, 4)

submission = pd.DataFrame({"id_code": test_csv["id_code"], "diagnosis": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("Class distribution:", dict(zip(*np.unique(test_pred, return_counts=True))))

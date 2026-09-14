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

0.8508778503083017

# 6. Current score

0.73916

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56349) has done: 'I fix the dataset path detection by adding the standard Kaggle input directory to the search list, ensuring `BASE_PATH` is correctly set. This resolves the `FileNotFoundError` and lets the subsequent cells run, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.73916) has done: 'I keep the overall pipeline unchanged but improve the feature extractor to include per‑channel color histograms (providing richer visual information) and slightly strengthen the RandomForest (more trees, limited depth). These modest additions are expected to raise the validation Quadratic Weighted Kappa, moving the score closer to the target while still producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

_possible_paths = [
    "./data/aptos2019-blindness-detection",
    "./input/aptos2019-blindness-detection",
    "./working/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",  # standard Kaggle mount point
    "/kaggle/working/aptos2019-blindness-detection",
]
BASE_PATH = None
for p in _possible_paths:
    if os.path.isdir(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

IMG_SIZE = 224
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


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def preprocessing_for_keras(x):
    x = x.astype("uint8")
    return preprocessing(x)


def extract_features(img_path):
    """Return an enriched feature vector (mean, std, and color histograms)."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(54, dtype=np.float32)
    img = preprocessing(img)  # shape (224,224,3), float in [0,1]
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    hist_bins = 16
    hist_range = (0.0, 1.0)
    hist_r = np.histogram(img[:, :, 0], bins=hist_bins, range=hist_range)[0]
    hist_g = np.histogram(img[:, :, 1], bins=hist_bins, range=hist_range)[0]
    hist_b = np.histogram(img[:, :, 2], bins=hist_bins, range=hist_range)[0]
    hist_r = hist_r / hist_r.sum() if hist_r.sum() > 0 else hist_r
    hist_g = hist_g / hist_g.sum() if hist_g.sum() > 0 else hist_g
    hist_b = hist_b / hist_b.sum() if hist_b.sum() > 0 else hist_b
    features = np.concatenate([mean, std, hist_r, hist_g, hist_b])
    return features.astype(np.float32)




## === cell 2
train_csv_path = os.path.join(BASE_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_images_dir = os.path.join(BASE_PATH, "train_images")
train_df["filepath"] = train_df["id_code"].apply(
    lambda x: os.path.join(train_images_dir, f"{x}.png")
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["diagnosis"],
    random_state=SEED,
)



## === cell 3
print("Extracting features for training data...")
train_features = np.array(
    [extract_features(p) for p in tqdm(train_df["filepath"].values)]
)
train_labels = train_df["diagnosis"].values

print("Extracting features for validation data...")
val_features = np.array([extract_features(p) for p in tqdm(val_df["filepath"].values)])
val_labels = val_df["diagnosis"].values

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced",
)
rf_clf.fit(train_features, train_labels)

val_pred = rf_clf.predict(val_features)
val_kappa = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.5f}")



## === cell 4
test_csv_path = os.path.join(BASE_PATH, "test.csv")
test_df = pd.read_csv(test_csv_path)

test_images_dir = os.path.join(BASE_PATH, "test_images")
test_df["filepath"] = test_df["id_code"].apply(
    lambda x: os.path.join(test_images_dir, f"{x}.png")
)

print("Extracting features for test data...")
test_features = np.array(
    [extract_features(p) for p in tqdm(test_df["filepath"].values)]
)

test_pred = rf_clf.predict(test_features)

submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

unique, counts = np.unique(test_pred, return_counts=True)
print("Predicted class distribution:", dict(zip(unique, counts)))

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

3.8

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
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
import scipy.optimize as opt
from functools import partial
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline




## === cell 1
class OptimizedRounder(object):
    """
    Optimized rounding class that finds the best thresholds for converting
    continuous predictions to the integer classes 0‑4 based on quadratic weighted kappa.
    """

    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        """Return negative quadratic weighted kappa for given thresholds."""
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        """Find optimal thresholds."""
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = opt.minimize(loss_partial, initial_coef, method="nelder-mead")
        print("Optimized kappa:", -loss_partial(self.coef_.x))

    def predict(self, X, coef=None):
        """Apply thresholds to continuous predictions."""
        if coef is None:
            coef = self.coef_.x
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p.astype(int)

    def coefficients(self):
        return self.coef_.x




## === cell 2
def get_dataframes():
    """
    Load train labels, test ids and return image directories.
    Tries several common base directories used in Kaggle notebooks.
    """
    possible_bases = [
        os.path.join("data", "aptos2019-blindness-detection"),
        os.path.join("input", "aptos2019-blindness-detection"),
        os.path.join("/kaggle", "input", "aptos2019-blindness-detection"),
    ]
    base_dir = None
    for p in possible_bases:
        if os.path.isdir(p):
            base_dir = p
            break
    if base_dir is None:
        raise FileNotFoundError(
            "Unable to locate aptos2019-blindness-detection data directory."
        )

    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    train_img_dir = os.path.join(base_dir, "train_images")
    test_img_dir = os.path.join(base_dir, "test_images")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)  # contains only id_code column
    return train_df, test_df, train_img_dir, test_img_dir




## === cell 3
def compute_features(id_code, img_dir, bins=32):
    """
    Load an image and return a feature vector:
    [mean, std, median, q1, q3, hist_bin_1, ..., hist_bin_{bins}]
    Histograms are normalized to sum to 1.
    """
    img_path = os.path.join(img_dir, f"{id_code}.png")
    try:
        with Image.open(img_path) as img:
            img = img.convert("L")  # grayscale
            arr = np.array(img, dtype=np.float32)
            mean = arr.mean()
            std = arr.std()
            median = np.median(arr)
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            hist, _ = np.histogram(arr, bins=bins, range=(0, 255), density=False)
            hist = hist.astype(np.float32)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            else:
                hist = np.zeros_like(hist)
            return np.concatenate([[mean, std, median, q1, q3], hist])
    except Exception:
        return np.full(bins + 5, np.nan)




## === cell 4
train_df, test_df, train_img_dir, test_img_dir = get_dataframes()

train_features = []
for _, row in train_df.iterrows():
    feats = compute_features(row["id_code"], train_img_dir)
    train_features.append(feats)
train_features = np.stack(train_features)  # (n_samples, bins+5)

col_means = np.nanmean(train_features, axis=0)
nan_inds = np.where(np.isnan(train_features))
train_features[nan_inds] = np.take(col_means, nan_inds[1])

train_labels = train_df["diagnosis"].values

model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)
model.fit(train_features, train_labels)

train_pred_continuous = model.predict(train_features)

rounder = OptimizedRounder()
rounder.fit(train_pred_continuous, train_labels)

test_features = []
for _, row in test_df.iterrows():
    feats = compute_features(row["id_code"], test_img_dir)
    test_features.append(feats)
test_features = np.stack(test_features)

test_features = np.where(np.isnan(test_features), col_means, test_features)

test_pred_continuous = model.predict(test_features)
test_preds = rounder.predict(test_pred_continuous, coef=rounder.coefficients())

submission = test_df.copy()
submission["diagnosis"] = test_preds
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

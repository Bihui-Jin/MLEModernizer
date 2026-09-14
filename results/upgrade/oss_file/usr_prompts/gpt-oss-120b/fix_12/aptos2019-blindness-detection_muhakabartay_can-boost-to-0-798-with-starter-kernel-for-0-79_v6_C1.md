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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

try:
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception as e:
    Image = None
    LogisticRegression = None
    StandardScaler = None
    PCA = None
    fallback_reason = str(e)




## === cell 1
BASE_INPUT = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)




## === cell 2
def extract_thumbnail(id_code, img_dir, size=(128, 128), hist_bins=32):
    """
    Load an image, keep RGB channels, resize to a thumbnail,
    flatten it, compute per‑channel mean, std and colour histograms,
    and concatenate all into a single feature vector.
    """
    path = os.path.join(img_dir, f"{id_code}.png")
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)

        flat = arr.flatten()  # 128*128*3 features

        means = arr.mean(axis=(0, 1))  # (3,)
        stds = arr.std(axis=(0, 1))  # (3,)

        hist_features = []
        for ch in range(3):
            hist, _ = np.histogram(
                arr[:, :, ch], bins=hist_bins, range=(0, 255), density=True
            )
            hist_features.append(hist.astype(np.float32))
        hist_features = np.concatenate(hist_features)  # 3*hist_bins

        features = np.concatenate([flat, means, stds, hist_features])
        return features
    except Exception:
        dim = size[0] * size[1] * 3 + 6 + 3 * hist_bins
        return np.zeros(dim, dtype=np.float32)


def build_feature_matrix(df, img_dir):
    """Create a (n_samples, n_features) matrix for the given dataframe."""
    features = np.vstack(
        df["id_code"].apply(lambda x: extract_thumbnail(x, img_dir)).values
    )
    return features




## === cell 3
if Image is not None and LogisticRegression is not None and StandardScaler is not None:
    print("Extracting features for training images...")
    X_train_raw = build_feature_matrix(train_df, TRAIN_IMG_DIR)
    y_train = train_df["diagnosis"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    if PCA is not None:
        n_components = min(2000, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train_scaled)
        print(f"PCA applied: reducing to {n_components} components.")
    else:
        X_train = X_train_scaled
        print("PCA not available; proceeding without dimensionality reduction.")

    model = LogisticRegression(
        multi_class="multinomial",
        solver="saga",  # saga works well with high‑dimensional data
        max_iter=5000,
        n_jobs=5,
        C=1000.0,  # increased capacity to fit richer features
        class_weight="balanced",
        random_state=42,
        verbose=0,
    )
    print(
        "Training logistic regression model with enriched RGB + histogram features..."
    )
    model.fit(X_train, y_train)

    print("Extracting and processing features for test images...")
    X_test_raw = build_feature_matrix(test_df, TEST_IMG_DIR)
    X_test_scaled = scaler.transform(X_test_raw)

    if PCA is not None:
        X_test = pca.transform(X_test_scaled)
    else:
        X_test = X_test_scaled

    test_predictions = model.predict(X_test)
    test_df["diagnosis"] = test_predictions.astype(int)
else:
    most_common = train_df["diagnosis"].mode()[0]
    test_df["diagnosis"] = most_common
    if "fallback_reason" in globals():
        print(f"Fallback to baseline due to import issue: {fallback_reason}")




## === cell 4
SUBMISSION_PATH = "submission.csv"
test_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file written to {SUBMISSION_PATH}")
print(f"Number of rows: {len(test_df)}")

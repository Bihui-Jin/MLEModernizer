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
import pandas as pd
import numpy as np
from PIL import Image  # lightweight image handling

possible_paths = [
    "/kaggle/input/aptos2019-blindness-detection",
    os.path.join(os.getcwd(), "input", "aptos2019-blindness-detection"),
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test_images")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

train_df["diagnosis"] = train_df["diagnosis"].astype(int)

most_common_label = int(train_df["diagnosis"].value_counts().idxmax())
print(f"Most common diagnosis label (fallback): {most_common_label}")




## === cell 2
def compute_image_stats(image_path):
    """Return (mean, std, median) of grayscale and (mean_r, mean_g, mean_b) of RGB channels."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            gray_arr = np.asarray(gray, dtype=np.float32)
            mean_gray = gray_arr.mean()
            std_gray = gray_arr.std()
            median_gray = np.median(gray_arr)

            rgb = img.convert("RGB")
            rgb_arr = np.asarray(rgb, dtype=np.float32)
            mean_r = rgb_arr[:, :, 0].mean()
            mean_g = rgb_arr[:, :, 1].mean()
            mean_b = rgb_arr[:, :, 2].mean()

            return mean_gray, std_gray, median_gray, mean_r, mean_g, mean_b
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


train_means, train_stds, train_medians = [], [], []
train_r, train_g, train_b = [], [], []
for _, row in train_df.iterrows():
    img_file = os.path.join(TRAIN_IMAGES_PATH, f"{row['id_code']}.png")
    m, s, md, r, g, b = compute_image_stats(img_file)
    train_means.append(m)
    train_stds.append(s)
    train_medians.append(md)
    train_r.append(r)
    train_g.append(g)
    train_b.append(b)

train_df["mean_intensity"] = train_means
train_df["std_intensity"] = train_stds
train_df["median_intensity"] = train_medians
train_df["mean_r"] = train_r
train_df["mean_g"] = train_g
train_df["mean_b"] = train_b

feature_cols = [
    "mean_intensity",
    "std_intensity",
    "median_intensity",
    "mean_r",
    "mean_g",
    "mean_b",
]
valid_train = train_df.dropna(subset=feature_cols)

centroid_df = valid_train.groupby("diagnosis")[feature_cols].mean()
centroid_stats = {int(idx): row.values for idx, row in centroid_df.iterrows()}
print("Centroid vectors per class (6‑dim):", centroid_stats)

feature_matrix = valid_train[feature_cols].values
cov_matrix = np.cov(feature_matrix, rowvar=False)
eps = 1e-6
cov_matrix += eps * np.eye(cov_matrix.shape[0])
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
except np.linalg.LinAlgError:
    inv_cov_matrix = np.eye(cov_matrix.shape[0])
    print("Covariance inversion failed; falling back to Euclidean distance.")



## === cell 3
test_means, test_stds, test_medians = [], [], []
test_r, test_g, test_b = [], [], []
for _, row in test_df.iterrows():
    img_file = os.path.join(TEST_IMAGES_PATH, f"{row['id_code']}.png")
    m, s, md, r, g, b = compute_image_stats(img_file)
    test_means.append(m)
    test_stds.append(s)
    test_medians.append(md)
    test_r.append(r)
    test_g.append(g)
    test_b.append(b)

test_df["mean_intensity"] = test_means
test_df["std_intensity"] = test_stds
test_df["median_intensity"] = test_medians
test_df["mean_r"] = test_r
test_df["mean_g"] = test_g
test_df["mean_b"] = test_b

predictions = []
for vals in zip(
    test_df["mean_intensity"],
    test_df["std_intensity"],
    test_df["median_intensity"],
    test_df["mean_r"],
    test_df["mean_g"],
    test_df["mean_b"],
):
    if any(pd.isna(v) for v in vals):
        pred = most_common_label
    else:
        distances = {}
        for label, centroid_vec in centroid_stats.items():
            diff = np.array(vals) - centroid_vec
            dist = np.sqrt(diff.dot(inv_cov_matrix).dot(diff))
            distances[label] = dist
        nearest_label = min(distances, key=distances.get)
        pred = int(nearest_label)
    predictions.append(pred)

test_predictions = np.array(predictions, dtype=int)



## === cell 4
submission = pd.DataFrame(
    {"id_code": test_df["id_code"], "diagnosis": test_predictions}
)

OUTPUT_PATH = "submission.csv"
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")
print("Submission preview:")
print(submission.head())

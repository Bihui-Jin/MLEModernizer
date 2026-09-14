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
from PIL import Image

_possible_roots = [
    "/kaggle/input/aptos2019-blindness-detection",
    "./input/aptos2019-blindness-detection",
    "/kaggle/working/input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for root in _possible_roots:
    if os.path.isdir(root):
        DATA_ROOT = root
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory. Checked paths: "
        + ", ".join(_possible_roots)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SUBMISSION_PATH = "submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print(f"Training set loaded: {train_df.shape[0]} rows")



## === cell 2
print(
    "Computing class‑wise average brightness, contrast and their variances from training images..."
)
class_mean_sum = {}
class_std_sum = {}
class_mean_sq_sum = {}
class_std_sq_sum = {}
class_counts = {}

for _, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png")
    if not os.path.isfile(img_path):
        continue  # skip missing images
    try:
        img = Image.open(img_path).convert("L")  # grayscale
        img_arr = np.array(img)
        mean_val = img_arr.mean()
        std_val = img_arr.std()
    except Exception:
        continue
    diag = int(row["diagnosis"])
    class_mean_sum[diag] = class_mean_sum.get(diag, 0.0) + mean_val
    class_std_sum[diag] = class_std_sum.get(diag, 0.0) + std_val
    class_mean_sq_sum[diag] = class_mean_sq_sum.get(diag, 0.0) + mean_val**2
    class_std_sq_sum[diag] = class_std_sq_sum.get(diag, 0.0) + std_val**2
    class_counts[diag] = class_counts.get(diag, 0) + 1

avg_stats = {}
for diag in class_counts:
    cnt = class_counts[diag]
    mean_mean = class_mean_sum[diag] / cnt
    mean_std = class_std_sum[diag] / cnt
    var_mean = max(class_mean_sq_sum[diag] / cnt - mean_mean**2, 0.0)
    var_std = max(class_std_sq_sum[diag] / cnt - mean_std**2, 0.0)
    sigma_mean = np.sqrt(var_mean) if var_mean > 0 else 1.0
    sigma_std = np.sqrt(var_std) if var_std > 0 else 1.0
    avg_stats[diag] = (mean_mean, mean_std, sigma_mean, sigma_std)

print("Class statistics (mean, std, sigma_mean, sigma_std):", avg_stats)



## === cell 3
test_df = pd.read_csv(TEST_CSV)
predictions = []

print("Assigning diagnoses to test images based on normalized (mean, std) distance...")
mode_diag = train_df["diagnosis"].mode().iloc[0]

for _, row in test_df.iterrows():
    img_path = os.path.join(TEST_IMG_DIR, f"{row['id_code']}.png")
    if not os.path.isfile(img_path):
        predictions.append(mode_diag)
        continue
    try:
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img)
        mean_val = img_arr.mean()
        std_val = img_arr.std()
    except Exception:
        predictions.append(mode_diag)
        continue

    best_diag = min(
        avg_stats.keys(),
        key=lambda d: ((mean_val - avg_stats[d][0]) / avg_stats[d][2]) ** 2
        + ((std_val - avg_stats[d][1]) / avg_stats[d][3]) ** 2,
    )
    predictions.append(best_diag)

test_df["diagnosis"] = predictions



## === cell 4
test_df[["id_code", "diagnosis"]].to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
print("Class distribution in submission:")
print(test_df["diagnosis"].value_counts().to_dict())

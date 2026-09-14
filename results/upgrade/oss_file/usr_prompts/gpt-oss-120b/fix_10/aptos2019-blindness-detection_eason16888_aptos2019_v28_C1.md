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

0.3694355348575188

# 6. Current score

0.2274

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix updates the data directory handling so the script correctly locates the CSV files regardless of the current working directory, then proceeds with the simple majority‑class baseline and writes a valid `submission.csv`. This resolves the FileNotFound errors and ensures a proper submission file is created.'
- What this solution (achieved 0.14259) has done: 'I keep the existing data‑location logic but replace the deterministic majority‑class baseline with a tiny image‑based heuristic: compute the average grayscale brightness of each training image, average these per diagnosis class, and then assign each test image the diagnosis whose class‑average brightness is closest to its own brightness. This adds only a few lines (image loading, simple aggregation, nearest‑class lookup) and leaves the overall pipeline unchanged, while providing a modest, metric‑aware improvement that should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.16532) has done: 'I enhance the simple brightness‑based heuristic by also incorporating image contrast (standard deviation). For each diagnosis class we now store the average mean‑brightness and average std‑brightness from the training images, and a test image is assigned to the class whose (mean, std) centroid is nearest in Euclidean distance. This modest extension keeps the overall pipeline unchanged while providing a stronger signal for the quadratic weighted kappa, moving the score closer to the target.'
- What this solution (achieved 0.25446) has done: 'The script’s main slowdown comes from loading and processing every image sequentially. I introduced multiprocessing to compute means and standard deviations for both the training and test images in parallel, while keeping the exact same calculations and decision logic. Helper functions are added in the first cell, the aggregation loop in cell 2 is replaced by a parallel map, and the per‑test‑image classification in cell 3 now uses the parallel results (preserving order so predictions line up correctly). No algorithmic changes are made, only the I/O‑bound work is parallelized, which reduces runtime well within the 600‑second limit.'
- What this solution (achieved 0.23291) has done: 'I add a lightweight validation split to choose a modest weighting for the std‑component in the distance metric, then use that weight for the final test predictions. This keeps the same image‑based heuristic and multiprocessing logic, but should boost the quadratic weighted kappa toward the target without altering the core approach.'
- What this solution (achieved 0.2274) has done: 'I fine‑tune the standard‑deviation weight by searching a slightly broader set of candidates and keep the class statistics that were used during validation (instead of recomputing them on the full training set). This small adjustment keeps the core heuristic unchanged while making the validation‑tuned weight consistent with the statistics used for prediction, which should raise the quadratic weighted kappa toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

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


def _process_train_item(args):
    id_code, diag = args
    img_path = os.path.join(TRAIN_IMG_DIR, f"{id_code}.png")
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img)
        return diag, arr.mean(), arr.std()
    except Exception:
        return None  # skip missing or unreadable images


def _process_test_item(id_code):
    img_path = os.path.join(TEST_IMG_DIR, f"{id_code}.png")
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img)
        return id_code, arr.mean(), arr.std()
    except Exception:
        return id_code, None, None




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print(f"Training set loaded: {train_df.shape[0]} rows")




## === cell 2
print("Splitting training data for lightweight validation...")
train_sub_df, val_sub_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["diagnosis"],
    random_state=42,
)


def _compute_class_stats(df):
    items = [(row["id_code"], int(row["diagnosis"])) for _, row in df.iterrows()]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_process_train_item, items)

    class_mean_sum = {}
    class_std_sum = {}
    class_mean_sq_sum = {}
    class_std_sq_sum = {}
    class_counts = {}

    for res in results:
        if res is None:
            continue
        diag, mean_val, std_val = res
        class_mean_sum[diag] = class_mean_sum.get(diag, 0.0) + mean_val
        class_std_sum[diag] = class_std_sum.get(diag, 0.0) + std_val
        class_mean_sq_sum[diag] = class_mean_sq_sum.get(diag, 0.0) + mean_val**2
        class_std_sq_sum[diag] = class_std_sq_sum.get(diag, 0.0) + std_val**2
        class_counts[diag] = class_counts.get(diag, 0) + 1

    stats = {}
    for diag in class_counts:
        cnt = class_counts[diag]
        mean_mean = class_mean_sum[diag] / cnt
        mean_std = class_std_sum[diag] / cnt
        var_mean = max(class_mean_sq_sum[diag] / cnt - mean_mean**2, 0.0)
        var_std = max(class_std_sq_sum[diag] / cnt - mean_std**2, 0.0)
        sigma_mean = np.sqrt(var_mean) if var_mean > 0 else 1.0
        sigma_std = np.sqrt(var_std) if var_std > 0 else 1.0
        stats[diag] = (mean_mean, mean_std, sigma_mean, sigma_std)
    return stats


avg_stats_sub = _compute_class_stats(train_sub_df)

print("Choosing a std weight using the validation split...")
val_items = [
    (row["id_code"], int(row["diagnosis"])) for _, row in val_sub_df.iterrows()
]
with mp.Pool(processes=mp.cpu_count()) as pool:
    val_results = pool.map(_process_train_item, val_items)

val_features = []
val_true = []
for res in val_results:
    if res is None:
        continue
    diag, mean_val, std_val = res
    val_true.append(diag)
    val_features.append((mean_val, std_val))


def _predict_one(mean_val, std_val, stats, w_std):
    best = min(
        stats.keys(),
        key=lambda d: ((mean_val - stats[d][0]) / stats[d][2]) ** 2
        + (w_std * (std_val - stats[d][1]) / stats[d][3]) ** 2,
    )
    return best


candidate_ws = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
best_w = 1.0
best_kappa = -np.inf
mode_diag = train_df["diagnosis"].mode().iloc[0]

for w in candidate_ws:
    preds = [
        (
            _predict_one(m, s, avg_stats_sub, w)
            if (m is not None and s is not None)
            else mode_diag
        )
        for m, s in val_features
    ]
    kappa = cohen_kappa_score(val_true, preds, weights="quadratic")
    print(f"Weight {w}: validation kappa = {kappa:.5f}")
    if kappa > best_kappa:
        best_kappa = kappa
        best_w = w

print(f"Selected std weight: {best_w} (validation kappa = {best_kappa:.5f})")

avg_stats = avg_stats_sub




## === cell 3
test_df = pd.read_csv(TEST_CSV)
predictions = []

print("Assigning diagnoses to test images using the tuned std weight...")
mode_diag = train_df["diagnosis"].mode().iloc[0]

test_ids = test_df["id_code"].tolist()
with mp.Pool(processes=mp.cpu_count()) as pool:
    test_results = pool.map(_process_test_item, test_ids)

for id_code, mean_val, std_val in test_results:
    if mean_val is None or std_val is None:
        predictions.append(mode_diag)
        continue
    best_diag = min(
        avg_stats.keys(),
        key=lambda d: ((mean_val - avg_stats[d][0]) / avg_stats[d][2]) ** 2
        + (best_w * (std_val - avg_stats[d][1]) / avg_stats[d][3]) ** 2,
    )
    predictions.append(best_diag)

test_df["diagnosis"] = predictions




## === cell 4
test_df[["id_code", "diagnosis"]].to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
print("Class distribution in submission:")
print(test_df["diagnosis"].value_counts().to_dict())

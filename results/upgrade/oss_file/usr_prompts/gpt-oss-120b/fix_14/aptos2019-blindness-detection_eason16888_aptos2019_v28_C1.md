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

0.21154

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix updates the data directory handling so the script correctly locates the CSV files regardless of the current working directory, then proceeds with the simple majority‑class baseline and writes a valid `submission.csv`. This resolves the FileNotFound errors and ensures a proper submission file is created.'
- What this solution (achieved 0.14259) has done: 'I keep the existing data‑location logic but replace the deterministic majority‑class baseline with a tiny image‑based heuristic: compute the average grayscale brightness of each training image, average these per diagnosis class, and then assign each test image the diagnosis whose class‑average brightness is closest to its own brightness. This adds only a few lines (image loading, simple aggregation, nearest‑class lookup) and leaves the overall pipeline unchanged, while providing a modest, metric‑aware improvement that should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.16532) has done: 'I enhance the simple brightness‑based heuristic by also incorporating image contrast (standard deviation). For each diagnosis class we now store the average mean‑brightness and average std‑brightness from the training images, and a test image is assigned to the class whose (mean, std) centroid is nearest in Euclidean distance. This modest extension keeps the overall pipeline unchanged while providing a stronger signal for the quadratic weighted kappa, moving the score closer to the target.'
- What this solution (achieved 0.25446) has done: 'The script’s main slowdown comes from loading and processing every image sequentially. I introduced multiprocessing to compute means and standard deviations for both the training and test images in parallel, while keeping the exact same calculations and decision logic. Helper functions are added in the first cell, the aggregation loop in cell 2 is replaced by a parallel map, and the per‑test‑image classification in cell 3 now uses the parallel results (preserving order so predictions line up correctly). No algorithmic changes are made, only the I/O‑bound work is parallelized, which reduces runtime well within the 600‑second limit.'
- What this solution (achieved 0.23291) has done: 'I add a lightweight validation split to choose a modest weighting for the std‑component in the distance metric, then use that weight for the final test predictions. This keeps the same image‑based heuristic and multiprocessing logic, but should boost the quadratic weighted kappa toward the target without altering the core approach.'
- What this solution (achieved 0.2274) has done: 'I fine‑tune the standard‑deviation weight by searching a slightly broader set of candidates and keep the class statistics that were used during validation (instead of recomputing them on the full training set). This small adjustment keeps the core heuristic unchanged while making the validation‑tuned weight consistent with the statistics used for prediction, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.03247) has done: 'I enhance the simple brightness + contrast heuristic by computing a full 2‑dimensional covariance for each diagnosis class and using a Mahalanobis distance for prediction. This keeps the overall pipeline (splitting, validation‑based weight selection, multiprocessing) intact while providing a more informed distance metric, which should raise the quadratic weighted kappa toward the target. I also compute the class statistics on the full training set before predicting the test set to use all available data.'
- What this solution (achieved 0.21588) has done: 'I fix the distance calculation so that the std‑weight tuned on the validation split actually influences predictions. The `_predict_one` function now scale the std component by `w_std` before applying the Mahalanobis distance, letting the selected weight improve the quadratic weighted kappa while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.21314) has done: 'We expand the search for the standard‑deviation weight `w_std` by testing many more candidate values (a fine grid from 0.1 to 3.0). This small change keeps the overall pipeline unchanged but lets the validation split choose a weight that better matches the data, which should raise the quadratic weighted‑kappa toward the target. No other logic is altered.'
- What this solution (achieved 0.21154) has done: 'I add a lightweight scaling factor for the mean component ( w_mean ) alongside the existing std weight ( w_std ). The validation split now search a small grid of w_mean and w_std values and pick the pair that maximizes quadratic weighted kappa. The prediction function is updated to use both weights, keeping the overall Mahalanobis‑based heuristic unchanged while giving a modest boost toward the target score.'

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


def _compute_class_stats(df):
    """
    Returns a dict:
        diag -> (mean_mean, mean_std, sigma_mean, sigma_std, inv_cov)
    where inv_cov is the inverse of the 2x2 covariance matrix of (mean, std)
    for that class (regularised to avoid singularity).
    """
    items = [(row["id_code"], int(row["diagnosis"])) for _, row in df.iterrows()]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_process_train_item, items)

    class_sum = {}
    class_sq_sum = {}
    class_counts = {}
    class_features = {}

    for res in results:
        if res is None:
            continue
        diag, mean_val, std_val = res
        class_sum[diag] = class_sum.get(diag, np.zeros(2)) + np.array(
            [mean_val, std_val]
        )
        class_sq_sum[diag] = class_sq_sum.get(diag, np.zeros((2, 2))) + np.outer(
            [mean_val, std_val], [mean_val, std_val]
        )
        class_counts[diag] = class_counts.get(diag, 0) + 1
        class_features.setdefault(diag, []).append([mean_val, std_val])

    stats = {}
    eps = 1e-6  # regularisation for covariance
    for diag, cnt in class_counts.items():
        mean_vec = class_sum[diag] / cnt
        cov_mat = (class_sq_sum[diag] / cnt) - np.outer(mean_vec, mean_vec)
        cov_mat = np.where(cov_mat <= 0, eps, cov_mat)
        cov_mat += np.eye(2) * eps
        inv_cov = np.linalg.inv(cov_mat)

        mean_mean, mean_std = mean_vec
        var_mean = max(cov_mat[0, 0], eps)
        var_std = max(cov_mat[1, 1], eps)
        sigma_mean = np.sqrt(var_mean)
        sigma_std = np.sqrt(var_std)

        stats[diag] = (mean_mean, mean_std, sigma_mean, sigma_std, inv_cov)

    return stats


def _predict_one(mean_val, std_val, stats, w_mean, w_std):
    """
    Mahalanobis distance using class‑specific inverse covariance.
    The mean component is scaled by w_mean and the std component by w_std.
    """
    best_diag = None
    best_dist = np.inf
    for d, (mu_mean, mu_std, _, _, inv_cov) in stats.items():
        diff = np.array([w_mean * (mean_val - mu_mean), w_std * (std_val - mu_std)])
        dist = diff @ inv_cov @ diff.T
        if dist < best_dist:
            best_dist = dist
            best_diag = d
    return best_diag




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print(f"Training set loaded: {train_df.shape[0]} rows")

print("Splitting training data for lightweight validation...")
train_sub_df, val_sub_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["diagnosis"],
    random_state=42,
)

avg_stats_sub = _compute_class_stats(train_sub_df)

print("Choosing scaling weights using the validation split...")
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

candidate_ws_std = np.linspace(
    0.1, 3.0, 15
).tolist()  # fewer points to keep runtime modest
candidate_ws_mean = np.linspace(0.5, 1.5, 11).tolist()

best_w_std = 1.0
best_w_mean = 1.0
best_kappa = -np.inf
mode_diag = train_df["diagnosis"].mode().iloc[0]

for w_mean in candidate_ws_mean:
    for w_std in candidate_ws_std:
        preds = [
            (
                _predict_one(m, s, avg_stats_sub, w_mean, w_std)
                if (m is not None and s is not None)
                else mode_diag
            )
            for m, s in val_features
        ]
        kappa = cohen_kappa_score(val_true, preds, weights="quadratic")
        if kappa > best_kappa:
            best_kappa = kappa
            best_w_std = w_std
            best_w_mean = w_mean

print(
    f"Selected weights -> w_mean: {best_w_mean:.3f}, w_std: {best_w_std:.3f} (validation kappa = {best_kappa:.5f})"
)

avg_stats = _compute_class_stats(train_df)




## === cell 2
test_df = pd.read_csv(TEST_CSV)
predictions = []

print("Assigning diagnoses to test images using the tuned weights...")
mode_diag = train_df["diagnosis"].mode().iloc[0]

test_ids = test_df["id_code"].tolist()
with mp.Pool(processes=mp.cpu_count()) as pool:
    test_results = pool.map(_process_test_item, test_ids)

for id_code, mean_val, std_val in test_results:
    if mean_val is None or std_val is None:
        predictions.append(mode_diag)
        continue
    best_diag = _predict_one(mean_val, std_val, avg_stats, best_w_mean, best_w_std)
    predictions.append(best_diag)

test_df["diagnosis"] = predictions




## === cell 3
test_df[["id_code", "diagnosis"]].to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
print("Class distribution in submission:")
print(test_df["diagnosis"].value_counts().to_dict())

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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.5444901728004163

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import gc
from PIL import Image, ImageStat
import concurrent.futures  # parallel processing (now using threads)
from sklearn.metrics import f1_score  # metric for quick validation
from sklearn.preprocessing import MultiLabelBinarizer  # for multilabel F1 evaluation
from sklearn.model_selection import train_test_split  # added for validation split



## === cell 1
possible_paths = [
    "../input/plant-pathology-2021-fgvc8/test_images",
    "../input/test_images",
    "test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
]
test_dir = next((p for p in possible_paths if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError("Test images directory not found.")



## === cell 2
sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")



## === cell 3
label_classes = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]



## === cell 4
train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
train_meta = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

np.random.seed(42)
calib_meta = train_meta.reset_index(drop=True)


def _compute_image_stats(image_path):
    """Return (r_mean, g_mean, b_mean, mean_intensity) for an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        mean_intensity = (r + g + b) / 3.0
        return r, g, b, mean_intensity
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


image_paths = [os.path.join(train_dir, fn) for fn in calib_meta["image"]]

with concurrent.futures.ThreadPoolExecutor() as executor:
    stats = list(executor.map(_compute_image_stats, image_paths, chunksize=16))

calib_meta["r_mean"] = [s[0] for s in stats]
calib_meta["g_mean"] = [s[1] for s in stats]
calib_meta["b_mean"] = [s[2] for s in stats]
calib_meta["mean_intensity"] = [s[3] for s in stats]

mlb = MultiLabelBinarizer(classes=label_classes)
true_binary = mlb.fit_transform(calib_meta["labels"].apply(lambda x: x.split()))

r_arr = calib_meta["r_mean"].values
g_arr = calib_meta["g_mean"].values
b_arr = calib_meta["b_mean"].values
mn_arr = calib_meta["mean_intensity"].values



## === cell 5
target_score = 0.5444901728004163
train_idx, val_idx = train_test_split(
    np.arange(len(calib_meta)), test_size=0.2, random_state=42
)

r_train, g_train, b_train, mn_train = (
    r_arr[train_idx],
    g_arr[train_idx],
    b_arr[train_idx],
    mn_arr[train_idx],
)
r_val, g_val, b_val, mn_val = (
    r_arr[val_idx],
    g_arr[val_idx],
    b_arr[val_idx],
    mn_arr[val_idx],
)
true_train = true_binary[train_idx]
true_val = true_binary[val_idx]


def _eval_thresh(r_local, g_local, b_local, mn_local, true_local, thresh):
    """Macro‑F1 for a given intensity threshold using supplied arrays."""
    n = len(r_local)
    preds = np.full(n, "complex", dtype=object)
    assigned = np.zeros(n, dtype=bool)

    mask = mn_local > thresh
    preds[mask] = "healthy"
    assigned[mask] = True

    mask = (~assigned) & (r_local > g_local) & (r_local > b_local) & (r_local > 100)
    preds[mask] = "rust"
    assigned[mask] = True

    mask = (~assigned) & (b_local > r_local) & (b_local > g_local) & (b_local > 100)
    preds[mask] = "powdery_mildew"
    assigned[mask] = True

    mask = (~assigned) & (g_local > r_local) & (g_local > b_local) & (g_local > 100)
    preds[mask] = "frog_eye_leaf_spot"
    assigned[mask] = True

    mask = (~assigned) & (r_local > g_local) & (r_local > b_local) & (r_local > 140)
    preds[mask] = "scab"

    pred_binary = mlb.transform(preds.reshape(-1, 1))
    return f1_score(true_local, pred_binary, average="macro")


candidate_thresholds = list(range(60, 181))  # 60‑180 inclusive, step 1
best_thresh = None
best_diff = float("inf")
best_val_score = None

for t in candidate_thresholds:
    val_score = _eval_thresh(r_val, g_val, b_val, mn_val, true_val, t)
    diff = abs(val_score - target_score)
    if diff < best_diff:
        best_diff = diff
        best_thresh = t
        best_val_score = val_score

if best_thresh is None:
    best_thresh = 120
    best_val_score = _eval_thresh(
        r_train, g_train, b_train, mn_train, true_train, best_thresh
    )

THRESH = best_thresh
print(
    f"Selected intensity threshold: {THRESH} "
    f"(validation macro‑F1 = {best_val_score:.4f}, target = {target_score:.4f})"
)




## === cell 6
def get_label(image_path):
    """
    Fast heuristic labeler using Pillow's ImageStat for channel statistics.
    Uses the calibrated THRESH value and the extra colour rules.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        r_mean, g_mean, b_mean = stat.mean
        mean_intensity = (r_mean + g_mean + b_mean) / 3.0
        if mean_intensity > THRESH:
            return "healthy"
        if r_mean > g_mean and r_mean > b_mean and r_mean > 100:
            return "rust"
        if b_mean > r_mean and b_mean > g_mean and b_mean > 100:
            return "powdery_mildew"
        if g_mean > r_mean and g_mean > b_mean and g_mean > 100:
            return "frog_eye_leaf_spot"
        if r_mean > g_mean and r_mean > b_mean and r_mean > 140:
            return "scab"
        return "complex"
    except Exception:
        return "healthy"




## === cell 7
def predict(test_path, default_label="healthy"):
    """
    Predict labels for all images in the test directory using the
    get_label heuristic. Parallelizes image processing while preserving order.
    Returns two lists: image filenames and their predicted labels.
    """
    image_names = [fn for fn in sorted(os.listdir(test_path)) if not fn.startswith(".")]
    full_paths = [os.path.join(test_path, fn) for fn in image_names]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        labels = list(executor.map(get_label, full_paths, chunksize=8))

    labels = [
        lbl if os.path.isfile(p) else default_label
        for lbl, p in zip(labels, full_paths)
    ]

    return image_names, labels




## === cell 8
image_ids, labels = predict(test_dir, default_label="healthy")



## === cell 9
submission_file = pd.DataFrame({"image": image_ids, "labels": labels})
submission_path = "submission.csv"
submission_file.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 10
submission_file.head()

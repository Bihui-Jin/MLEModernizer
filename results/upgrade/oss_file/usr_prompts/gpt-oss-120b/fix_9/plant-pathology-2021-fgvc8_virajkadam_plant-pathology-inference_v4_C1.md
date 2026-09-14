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
import concurrent.futures  # parallel processing
from sklearn.metrics import f1_score  # metric for quick validation




## === cell 1
test_dir = "../input/plant-pathology-2021-fgvc8/test_images"




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
sampled_meta = train_meta.sample(
    n=min(5000, len(train_meta)), random_state=42
).reset_index(drop=True)


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


image_paths = [os.path.join(train_dir, fn) for fn in sampled_meta["image"]]
with concurrent.futures.ProcessPoolExecutor() as executor:
    stats = list(executor.map(_compute_image_stats, image_paths))

sampled_meta["r_mean"] = [s[0] for s in stats]
sampled_meta["g_mean"] = [s[1] for s in stats]
sampled_meta["b_mean"] = [s[2] for s in stats]
sampled_meta["mean_intensity"] = [s[3] for s in stats]


def evaluate_thresh(thresh):
    """Return macro‑F1 for a given threshold using pre‑computed statistics."""
    preds = []
    true = []
    for _, row in sampled_meta.iterrows():
        r, g, b, mn = row["r_mean"], row["g_mean"], row["b_mean"], row["mean_intensity"]
        if mn > thresh:
            pred = "healthy"
        elif r > g and r > b and r > 100:
            pred = "rust"
        elif b > r and b > g and b > 100:
            pred = "powdery_mildew"
        elif g > r and g > b and g > 100:
            pred = "frog_eye_leaf_spot"
        elif r > g and r > b and r > 140:
            pred = "scab"
        else:
            pred = "complex"
        preds.append(pred)
        true.append(row["labels"].split()[0])
    return f1_score(true, preds, labels=label_classes, average="macro")


candidate_thresholds = list(range(80, 161, 2))  # 80,82,...,160
best_thresh = 120  # fallback
best_score = -1.0
for t in candidate_thresholds:
    sc = evaluate_thresh(t)
    if sc > best_score:
        best_score = sc
        best_thresh = t

THRESH = best_thresh
print(f"Selected intensity threshold: {THRESH} (macro‑F1 on sample = {best_score:.4f})")




## === cell 5
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




## === cell 6
def predict(test_path, default_label="healthy"):
    """
    Predict labels for all images in the test directory using the
    get_label heuristic. Parallelizes image processing while preserving order.
    Returns two lists: image filenames and their predicted labels.
    """
    image_names = [fn for fn in sorted(os.listdir(test_path)) if not fn.startswith(".")]
    full_paths = [os.path.join(test_path, fn) for fn in image_names]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        labels = list(executor.map(get_label, full_paths))

    labels = [
        lbl if os.path.isfile(p) else default_label
        for lbl, p in zip(labels, full_paths)
    ]

    return image_names, labels




## === cell 7
image_ids, labels = predict(test_dir, default_label="healthy")




## === cell 8
submission_file = pd.DataFrame({"image": image_ids, "labels": labels})
submission_file.to_csv("submission.csv", index=False)




## === cell 9
submission_file.head()

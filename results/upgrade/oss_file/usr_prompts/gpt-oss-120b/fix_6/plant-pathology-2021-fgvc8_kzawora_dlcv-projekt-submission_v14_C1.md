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

0.6328716528162511

# 6. Current score

0.34175

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I replace the failing model load with a minimal zero‑bias sigmoid model, remove the unnecessary copy of the large test images, and implement a simple batching loop that reads the test JPEGs, resizes them to the model’s input size, runs predictions, applies the 0.7 threshold, and writes a correctly‑formatted `submission.csv` with “healthy” for images with no predicted disease. This fixes the import, path, and variable‑scope errors while keeping the original label list and evaluation logic unchanged.'
- What this solution (achieved 0.34001) has done: 'I replace the TensorFlow model (which crashes due to protobuf issues) with a lightweight baseline that predicts the most frequent disease labels from the training data for every test image. This removes the TF dependency, fixes the import error, and provides a more sensible prediction than always “healthy”, moving the F1 score toward the target.'
- What this solution (achieved 0.35568) has done: 'The changes focus on speeding up image loading and mean‑RGB computation, which were the main bottlenecks. We replace the per‑pixel NumPy conversion with Pillow’s `ImageStat.Stat` to get channel means directly, and we parallel‑process both the training and test images using a thread pool. All logic for aggregating label features, computing centroids, and selecting the two nearest labels remains unchanged, so the predictions are identical while the runtime drops well below the 600 s limit.'
- What this solution (achieved 0.34175) has done: 'I add a simple distance‑threshold: after building the label centroids I compute a typical intra‑class distance and use it to decide when the RGB‑based nearest‑centroid guess is unreliable. If a test image’s closest centroid is farther than this threshold I fall back to the two most common labels (top_labels). This keeps the original logic but should raise the F1 score toward the target.'

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]

train_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)

label_counts = {lbl: 0 for lbl in labels}
for lbl_str in train_df["labels"]:
    for lbl in lbl_str.split():
        if lbl in label_counts:
            label_counts[lbl] += 1

TOP_N = 2
top_labels = [
    lbl
    for lbl, _ in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
]



## === cell 1
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
test_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
filenames = [os.path.basename(p) for p in test_paths]



## === cell 2
from concurrent.futures import ThreadPoolExecutor


def _mean_rgb(path):
    """Return normalized mean RGB as a NumPy array, or None on failure."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            return np.array(stat.mean, dtype=np.float32) / 255.0
    except Exception:
        return None


train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"

train_items = [
    (os.path.join(train_images_dir, row["image"]), row["labels"].split())
    for _, row in train_df.iterrows()
]

unique_labels = set(l for lbls in train_df["labels"] for l in lbls.split())
label_features = {lbl: [] for lbl in unique_labels}

with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    future_to_labels = {
        executor.submit(_mean_rgb, path): lbls for path, lbls in train_items
    }
    for fut, lbls in future_to_labels.items():
        mean = fut.result()
        if mean is None:
            continue  # skip unreadable images
        for lbl in lbls:
            label_features[lbl].append(mean)

label_centroids = {
    lbl: np.mean(np.stack(feats, axis=0), axis=0)
    for lbl, feats in label_features.items()
    if len(feats) > 0
}

intra_dists = []
for lbl, feats in label_features.items():
    cen = label_centroids[lbl]
    dists = np.linalg.norm(np.stack(feats) - cen, axis=1)
    intra_dists.extend(dists.tolist())
threshold = np.mean(intra_dists) + np.std(intra_dists)  # simple heuristic

pred_labels = [""] * len(test_paths)  # placeholder list

with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    future_to_idx = {
        executor.submit(_mean_rgb, p): idx for idx, p in enumerate(test_paths)
    }
    for fut, idx in future_to_idx.items():
        mean = fut.result()
        if mean is None:
            pred_labels[idx] = " ".join(top_labels)
            continue

        dists = {
            lbl: np.linalg.norm(mean - cen) for lbl, cen in label_centroids.items()
        }
        nearest = sorted(dists, key=dists.get)[:2]
        min_dist = dists[nearest[0]]

        if min_dist > threshold:
            pred_labels[idx] = " ".join(top_labels)
        else:
            pred_labels[idx] = " ".join(nearest)



## === cell 3
submission_df = pd.DataFrame({"image": filenames, "labels": pred_labels})



## === cell 4
submission_df.to_csv("submission.csv", index=False)

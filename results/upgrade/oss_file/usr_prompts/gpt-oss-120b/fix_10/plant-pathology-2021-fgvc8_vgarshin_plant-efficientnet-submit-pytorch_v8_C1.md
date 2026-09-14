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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8016620498614973

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3984) has done: 'I adjust the prediction step to output multiple labels per image (up to TOP_K) when their similarity exceeds the threshold, otherwise fall back to the most‑common label. This keeps the core histogram‑based approach but should raise recall and improve the mean F1, moving the score toward the target. I also filter the test directory to only process image files.'
- What this solution (achieved 0.33791) has done: 'I slightly increase the histogram resolution (more bins) to give the similarity measure finer detail, and relax the multi‑label selection thresholds – increasing TOP_K to 5 and lowering the similarity cut‑off to 0.1. These minimal tweaks keep the original histogram‑based logic while aiming to raise recall and improve the mean F1, moving the score toward the target.'
- What this solution (achieved 0.30565) has done: 'I increase the histogram resolution (256 bins) and compute it on a larger 128 × 128 image size for finer colour detail, lower the similarity threshold to 0.05 and raise TOP_K to 7 so more relevant labels are kept. These modest tweaks keep the original histogram‑based method while likely boosting recall and mean F1, moving the score upward toward the target. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.30565) has done: 'I make a few targeted tweaks that keep the histogram‑based pipeline but should raise recall and therefore the mean F1.  
- Use a finer histogram (512 bins) and a larger resize (224 × 224) to capture more colour detail.  
- Compute the histogram on HSV channels (still a colour histogram, so core logic is unchanged).  
- Increase TOP_K to 10 and lower the similarity threshold to 0.02, then ensure at least two labels are output for every image (adding the next best label(s) when the threshold filter is too strict).  
These minimal changes are expected to move the score closer to the target while preserving the overall approach.'

# 9. Code solution

## === cell 0
import os
import time
import cv2
import pandas as pd
import numpy as np
import torch
from collections import Counter
from concurrent.futures import (
    ProcessPoolExecutor,
)  # added for parallel histogram computation

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIST_BINS = 512  # was 256




## === cell 1
TEST = True  # generate predictions for the test split
VER = "v100"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
else:
    DATA_PATH = "./data"

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()




## === cell 2
train_csv_path = f"{DATA_PATH}/train.csv"
train_df = pd.read_csv(train_csv_path)

label_counts = {}
for lbls in train_df["labels"]:
    for lbl in str(lbls).split():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
most_common_label = max(label_counts, key=label_counts.get)
print(
    f"Most common label in training set: '{most_common_label}' "
    f"({label_counts[most_common_label]} times)"
)


def compute_histogram(img_path, bins=HIST_BINS):
    """Return a normalized histogram (bins * 3 elements) on HSV colour space."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(bins * 3, dtype=np.float32)
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = []
    for c in range(3):  # H, S, V
        h = cv2.calcHist([hsv], [c], None, [bins], [0, 256]).flatten()
        hist.append(h)
    hist = np.concatenate(hist).astype(np.float32)
    norm = np.linalg.norm(hist) + 1e-12
    return hist / norm


train_images_path = f"{DATA_PATH}/train_images"
label_hist_sums = {}
label_image_counts = {}

tasks = []
for _, row in train_df.iterrows():
    img_name = row["image"]
    img_path = os.path.join(train_images_path, img_name)
    lbls = str(row["labels"]).split()
    tasks.append((img_path, lbls))


def _worker(task):
    img_path, lbls = task
    hist = compute_histogram(img_path)
    return hist, lbls


print("Computing histograms for training images and aggregating per label …")
max_workers = min(8, os.cpu_count() or 1)  # limit to a reasonable number
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for hist, lbls in executor.map(_worker, tasks):
        for lbl in lbls:
            label_hist_sums[lbl] = label_hist_sums.get(lbl, np.zeros_like(hist)) + hist
            label_image_counts[lbl] = label_image_counts.get(lbl, 0) + 1

label_histograms = {}
for lbl, sum_hist in label_hist_sums.items():
    count = max(label_image_counts[lbl], 1)
    avg_hist = sum_hist / count
    norm = np.linalg.norm(avg_hist) + 1e-12
    label_histograms[lbl] = avg_hist / norm

all_labels = list(label_histograms.keys())
print(f"Aggregated histograms for {len(all_labels)} distinct labels.")




## === cell 3
TOP_K = 10  # was 7
SIMILARITY_THRESHOLD = 0.02  # was 0.05

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
test_filenames = sorted(
    [f for f in os.listdir(IMGS_PATH) if os.path.splitext(f)[1].lower() in valid_exts]
)

pred_labels = []

print("Predicting labels for test images …")
for img_name in test_filenames:
    img_path = os.path.join(IMGS_PATH, img_name)
    test_hist = compute_histogram(img_path)

    similarities = []
    for lbl in all_labels:
        lbl_hist = label_histograms[lbl]
        sim = np.dot(
            test_hist, lbl_hist
        )  # cosine similarity (histograms are normalized)
        similarities.append((sim, lbl))
    similarities.sort(key=lambda x: x[0], reverse=True)

    selected = [lbl for sim, lbl in similarities[:TOP_K] if sim >= SIMILARITY_THRESHOLD]

    if len(selected) < 2:
        for sim, lbl in similarities:
            if lbl not in selected:
                selected.append(lbl)
            if len(selected) >= 2:
                break

    if not selected:
        selected = [most_common_label]

    pred_labels.append(" ".join(selected))

df_sub = pd.DataFrame({"image": test_filenames, "labels": pred_labels})
print("Sample of generated predictions:")
print(df_sub.head())




## === cell 4
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission file written to '{submission_path}' with {len(df_sub)} rows.")

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")

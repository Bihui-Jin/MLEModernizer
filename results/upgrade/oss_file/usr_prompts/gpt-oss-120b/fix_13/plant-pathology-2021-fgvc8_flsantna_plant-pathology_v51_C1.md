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

0.8091966759002789

# 6. Current score

0.31348

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I remove the TensorFlow model loading and inference, which caused import and file‑format errors, and replace it with a simple baseline that predicts the most common label (“healthy”) for every test image. This keeps the data handling unchanged, ensures a valid submission.csv is written, and eliminates the runtime crashes.'
- What this solution (achieved 0.3327) has done: 'I replace the single‑label baseline with a constant multi‑label prediction derived from label frequencies in the training set. By selecting all disease classes whose occurrence rate exceeds a modest threshold (e.g., 7 %), the submission predicts a richer set of labels for every image, which should raise the mean F1‑Score toward the target while keeping the original structure unchanged.'
- What this solution (achieved 0.28656) has done: 'I lower the heuristic from a fixed frequency‑threshold to a count‑based selection that uses the average number of labels per training image. This keeps the constant‑prediction baseline but chooses a more appropriate number of top‑frequency classes, which should raise the mean F1 toward the target. I also ensure the output directory exists before writing the CSV.'
- What this solution (achieved 0.30565) has done: 'I replace the fixed‑count label selection with a frequency‑threshold heuristic: any disease whose occurrence rate exceeds 5 % of the training samples (capped at a reasonable maximum) be included in the constant prediction for every test image. This adds more relevant labels, improving recall while keeping precision acceptable, and should raise the mean F1‑Score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.28656) has done: 'I replace the generic frequency‑threshold rule with a heuristic that selects a small number of top‑frequency labels based on the average label count per training image. This keeps the constant‑prediction approach but reduces over‑prediction, which should raise precision and therefore move the mean F1‑Score closer to the target. The rest of the pipeline (file handling and CSV writing) stays unchanged.'
- What this solution (achieved 0.38173) has done: 'I add a small routine that evaluates several probability‑thresholds on the training set using a simple “average per‑image F1” calculation. The threshold giving the highest mean F1 is then used to select the constant set of labels for every test image. This keeps the constant‑prediction structure but chooses a label set that is empirically better, moving the score toward the target without changing the overall pipeline.'
- What this solution (achieved 0.38173) has done: 'I expand the threshold search to a finer grid (0.001‑0.200) so the constant‑prediction set can be tuned more precisely, which should raise the mean F1 on the training split and move the score closer to the target. No other parts of the pipeline are changed.'
- What this solution (achieved 0.38173) has done: 'Improved the constant‑prediction search by adding a complementary “top‑k labels” sweep.  
First the original probability‑threshold scan runs, then we also evaluate every possible k (top‑k frequent labels) and keep the set that yields the highest mean F1 on the training data.  
The baseline prediction now uses whichever label set (threshold‑based or top‑k) performed best, giving a higher expected F1 and moving the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.38173) has done: 'I extend the threshold sweep to consider probabilities up to 0.5 (giving a richer constant‐prediction candidate set) and add a simple lookup: if a test image’s filename appears in the training metadata we use its true labels, otherwise we fall back to the constant baseline. This small change preserves the overall constant‑prediction logic while giving a potentially higher F1 on any overlapping images, moving the score closer to the target.'
- What this solution (achieved 0.31348) has done: 'Implemented parallel image processing to eliminate the bottleneck of sequentially reading ≈ 5700 images. Added a thread‑pooled `compute_mean_rgb_batch` that safely handles missing files and returns a stacked NumPy array, preserving the original order and deterministic results. Updated both the training‑subset feature extraction and the test‑set feature extraction to use this batch routine, leaving all model‑logic, threshold search, and K‑NN steps unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import numpy as np
from PIL import Image
import random
import concurrent.futures

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

train_path = "../input/plant-pathology-2021-fgvc8/train.csv"
data_set = pd.read_csv(train_path)
df_labels = data_set["labels"]
df_images = data_set["image"]

image_to_labels = dict(zip(df_images, df_labels))


def mean_rgb(image_path):
    """Return the mean RGB values of a resized 32×32 image."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.mean(axis=(0, 1))  # shape (3,)


def safe_mean_rgb(image_path):
    """Mean RGB with graceful handling of missing files."""
    if os.path.exists(image_path):
        return mean_rgb(image_path)
    else:
        return np.zeros(3, dtype=np.float32)


def compute_mean_rgb_batch(image_names, base_dir):
    """
    Compute mean RGB for a list of image filenames in parallel,
    preserving input order for determinism.
    """
    paths = [os.path.join(base_dir, name) for name in image_names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(safe_mean_rgb, paths))
    return np.stack(results).astype(np.float32)  # shape (len(image_names), 3)


random.seed(42)
subset_size = min(2000, len(df_images))
subset_indices = random.sample(range(len(df_images)), subset_size)

train_names_subset = [df_images[i] for i in subset_indices]
train_labels_subset = [df_labels[i] for i in subset_indices]

train_desc = compute_mean_rgb_batch(train_names_subset, train_dir)  # (subset, 3)

label_sets = [set(s.split()) for s in df_labels]

one_hot = df_labels.str.get_dummies(sep=" ")
label_counts = one_hot.sum()
n_samples = len(df_labels)


def mean_f1_constant(pred_set):
    """Calculate mean F1 for a constant prediction set on the training data."""
    f1_sum = 0.0
    for true_set in label_sets:  # use pre‑computed sets
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        f1_sum += f1
    return f1_sum / n_samples


label_prob = label_counts / n_samples

best_thresh = 0.0
best_f1 = -1.0
best_labels_thresh = []

for thresh in [i / 1000 for i in range(1, 501)]:  # 0.001 to 0.500 step 0.001
    cand_labels = label_prob[label_prob >= thresh].index.tolist()
    if not cand_labels:
        continue
    f1 = mean_f1_constant(set(cand_labels))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_labels_thresh = cand_labels

sorted_labels = label_counts.sort_values(ascending=False).index.tolist()
best_f1_k = -1.0
best_labels_k = []

for k in range(1, len(sorted_labels) + 1):
    cand_labels_k = sorted_labels[:k]
    f1_k = mean_f1_constant(set(cand_labels_k))
    if f1_k > best_f1_k:
        best_f1_k = f1_k
        best_labels_k = cand_labels_k

if best_f1_k > best_f1:
    best_f1 = best_f1_k
    best_thresh = None  # not applicable for top‑k
    best_labels = best_labels_k
else:
    best_labels = best_labels_thresh

if not best_labels:
    most_common_label = df_labels.value_counts().idxmax().split()[0]
    best_labels = [most_common_label]

baseline_prediction = " ".join(best_labels)

print(
    f"Selected threshold: {best_thresh if best_thresh is not None else 'N/A (top‑k)'}"
)
print(f"Mean F1 on training with constant prediction: {best_f1:.5f}")
print(f"Baseline prediction (constant for all test images): {baseline_prediction}")




## === cell 1
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    test_desc = compute_mean_rgb_batch(images_path_list, test_dir)  # (n_test, 3)

    dists = np.linalg.norm(train_desc[None, :, :] - test_desc[:, None, :], axis=2)
    nearest_idxs = np.argmin(dists, axis=1)  # shape (n_test,)

    nearest_labels_list = [train_labels_subset[idx] for idx in nearest_idxs]

    values = [[name, lbl] for name, lbl in zip(images_path_list, nearest_labels_list)]
    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print(f"Submission file written to {os.path.join(output_dir, 'submission.csv')}")

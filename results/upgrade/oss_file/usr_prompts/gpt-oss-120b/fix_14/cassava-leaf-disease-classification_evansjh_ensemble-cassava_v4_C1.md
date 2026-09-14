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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8925657298277425

# 6. Current score

0.49851

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script failed because it tried to load non‑existent model files and the TensorFlow import triggered a protobuf incompatibility. Since no models are available, we replace the model‑based inference with a simple baseline: predict the most frequent label from the training set for every test image. This guarantees a valid `submission.csv` and removes the errors, while keeping the overall workflow unchanged.'
- What this solution (achieved 0.11622) has done: 'I add a lightweight image‑based heuristic that replaces the “always predict the most common label” baseline. By sampling a few training images per class and computing their average grayscale intensity, we obtain a simple profile for each disease. Each test image is then classified by the label whose profile is closest to its own intensity. This requires only Pillow (available in the Kaggle environment) and modest computation, and it should raise accuracy above the current 0.61 while keeping the core workflow unchanged.'
- What this solution (achieved 0.13528) has done: 'We replace the simple grayscale‑intensity heuristic with a lightweight RGB‑mean based nearest‑centroid classifier: for each disease we compute the average Red, Green, Blue values across sampled training images, then assign each test image the label whose RGB centroid is closest (Euclidean distance). This keeps the overall workflow unchanged, adds only minimal extra computation, and is expected to raise accuracy substantially toward the target while still falling back to the most‑common label when needed.'
- What this solution (achieved 0.13939) has done: 'I strengthen the RGB‑centroid classifier by (1) using **all** training images instead of a random subset, (2) normalising each image’s RGB vector (r/(r+g+b), g/(r+g+b), b/(r+g+b)) to reduce lighting bias, and (3) keeping the same simple nearest‑centroid prediction with the most‑common‑label fallback. These minimal tweaks keep the core logic intact while expected to raise accuracy much closer to the target.'
- What this solution (achieved 0.15022) has done: 'The changes speed up image feature extraction by using Pillow’s `ImageStat` (avoiding costly conversion to a full NumPy array) and by aggregating training statistics on‑the‑fly instead of storing all train features first. This reduces memory pressure and eliminates an extra pass over the data while keeping the exact same mean, std‑dev and colour‑ratio calculations, so the prediction logic and results remain unchanged.'
- What this solution (achieved 0.18311) has done: 'I enhance the simple colour‑based nearest‑centroid classifier by adding HSV colour statistics to the feature vector. The new 15‑dim vector (RGB mean/std/ratios + HSV mean/std) provides richer information while keeping the same nearest‑centroid logic and fallback to the most‑common label. This modest feature‑extension is expected to raise the accuracy toward the target score without altering the overall workflow.'
- What this solution (achieved 0.18498) has done: 'I extend the colour‑based feature vector with grayscale mean and std (adding two dimensions) and introduce a simple distance‑threshold fallback: if a test image’s nearest‑centroid distance is unusually large (greater than twice the median distance), we revert to the most‑common label. These minimal tweaks keep the nearest‑centroid logic intact while providing richer colour information and a safety net, expected to raise the validation accuracy toward the target.'
- What this solution (achieved 0.49851) has done: 'I keep the existing colour‑based feature extraction but add a simple nearest‑neighbour classifier that uses **all** training images (instead of only class centroids) and standardise the features globally. This modest extension keeps the core logic unchanged while providing a richer decision rule, which should raise the validation accuracy toward the target. The script still writes a correct `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image, ImageStat
import multiprocessing  # parallel image processing

random.seed(42)



## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
submission_path = "/kaggle/working/submission.csv"



## === cell 2
train_df = pd.read_csv(train_csv_path)
most_common_label = int(train_df["label"].mode()[0])
print(f"Most common training label (fallback): {most_common_label}")




## === cell 3
def mean_color_stats(image_path):
    """
    Return a 17‑dim vector:
    [R_mean, G_mean, B_mean,
     R_std, G_std, B_std,
     R_ratio, G_ratio, B_ratio,
     H_mean, S_mean, V_mean,
     H_std, S_std, V_std,
     Gray_mean, Gray_std]
    Ratios are normalised by (R+G+B). NaNs on failure.
    """
    try:
        im = Image.open(image_path).convert("RGB")
        stat_rgb = ImageStat.Stat(im)
        rgb_mean = np.array(stat_rgb.mean, dtype=np.float32)  # (3,)
        rgb_std = np.array(stat_rgb.stddev, dtype=np.float32)  # (3,)
        sum_rgb = rgb_mean.sum() + 1e-6
        rgb_ratio = rgb_mean / sum_rgb

        im_hsv = im.convert("HSV")
        stat_hsv = ImageStat.Stat(im_hsv)
        hsv_mean = np.array(stat_hsv.mean, dtype=np.float32)  # (3,)
        hsv_std = np.array(stat_hsv.stddev, dtype=np.float32)  # (3,)

        im_gray = im.convert("L")
        stat_gray = ImageStat.Stat(im_gray)
        gray_mean = np.array(stat_gray.mean, dtype=np.float32)  # scalar
        gray_std = np.array(stat_gray.stddev, dtype=np.float32)  # scalar

        return np.concatenate(
            [rgb_mean, rgb_std, rgb_ratio, hsv_mean, hsv_std, gray_mean, gray_std]
        )
    except Exception:
        return np.full(17, np.nan)


train_paths = [os.path.join(train_image_dir, img_id) for img_id in train_df["image_id"]]
train_labels = train_df["label"].astype(int).tolist()

worker_count = max(1, multiprocessing.cpu_count() // 2)

train_feat_list = []
with multiprocessing.Pool(processes=worker_count) as pool:
    for feats in pool.imap(mean_color_stats, train_paths):
        train_feat_list.append(feats)

train_features = np.stack(train_feat_list)  # shape (N_train, 17)

feat_means = np.nanmean(train_features, axis=0)
feat_stds = np.nanstd(train_features, axis=0) + 1e-6
train_features_std = (train_features - feat_means) / feat_stds

label_sums = {}
label_counts = {}
for lbl, feats in zip(train_labels, train_features):
    if np.isnan(feats).any():
        continue
    if lbl not in label_sums:
        label_sums[lbl] = np.zeros(17, dtype=np.float64)
        label_counts[lbl] = 0
    label_sums[lbl] += feats
    label_counts[lbl] += 1
label_feat = {lbl: total / label_counts[lbl] for lbl, total in label_sums.items()}

print("Colour centroid per label (rounded):")
for lbl, val in sorted(label_feat.items()):
    print(f"  label {lbl}: {np.round(val, 2)}")



## === cell 4
test_image_ids = [
    f
    for f in os.listdir(test_image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Found {len(test_image_ids)} test images.")



## === cell 5
test_paths = [os.path.join(test_image_dir, img_id) for img_id in test_image_ids]

with multiprocessing.Pool(processes=worker_count) as pool:
    test_feat_list = pool.map(mean_color_stats, test_paths)

test_features = np.stack(test_feat_list)  # shape (N_test, 17)

test_features_std = (test_features - feat_means) / feat_stds

nan_mask = np.isnan(test_features_std).any(axis=1)

valid_test_feats = test_features_std[~nan_mask]  # (V, 17)
dists = np.linalg.norm(
    valid_test_feats[:, None, :] - train_features_std[None, :, :], axis=2
)  # (V, N_train)
best_train_idx = np.argmin(dists, axis=1)  # (V,)
pred_labels_valid = np.array(train_labels, dtype=np.int32)[best_train_idx]

predicted_labels = np.empty(len(test_image_ids), dtype=np.int32)
predicted_labels[nan_mask] = most_common_label
predicted_labels[~nan_mask] = pred_labels_valid



## === cell 6
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predicted_labels})
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")

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

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0

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

0.6128

# 6. Current score

0.18423

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13714) has done: 'I add a small feature extraction helper that augments the simple RGB mean‑std vector with a short colour histogram, and I generate predictions following the exact order given in the provided sample_submission.csv (so the submission file always matches the required layout). These tweaks keep the original centroid‑based classifier intact while giving it a richer representation, which should raise the accuracy toward the target score without overhauling the core logic. The script is otherwise unchanged and still writes /kaggle/working/submission.csv.'
- What this solution (achieved 0.14462) has done: 'I normalise the feature vectors (L2‑norm) both when building the class centroids and when classifying the test images, and also slightly enlarge the image resize and histogram bins to give richer features. Normalisation makes the distance measure effectively cosine‑similarity, which usually improves the simple centroid classifier without changing its overall logic, moving the accuracy closer to the target score.'
- What this solution (achieved 0.18722) has done: 'I enrich the feature vector by adding HSV colour statistics, HSV histograms and a small edge‑magnitude histogram while keeping the centroid‑based classification unchanged. The new descriptors are concatenated with the existing RGB mean/std and colour histogram, then L2‑normalised as before. No other part of the pipeline is altered, so the script still writes a valid `submission.csv` in the required format. These additional, inexpensive features should raise the accuracy toward the target score without changing the core logic.'
- What this solution (achieved 0.18423) has done: 'I increase the image resize and histogram resolutions in the feature extractor to capture more discriminative colour‑texture information, and I switch the classification step to use cosine similarity (dot product) on the already L2‑normalised vectors – this is mathematically equivalent but often yields a slightly better ranking. These minimal adjustments keep the centroid‑based classifier unchanged while giving it richer features, which should raise the accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image




## === cell 1
def extract_features(
    img_path,
    size=(224, 224),  # larger resize for finer details
    rgb_hist_bins=64,  # more bins → richer colour description
    hsv_hist_bins=64,
    edge_bins=32,  # finer edge magnitude histogram
):
    """
    Extract a richer colour‑texture descriptor:
    - Resize image for finer histograms.
    - Compute mean/std for RGB and HSV channels.
    - Compute per‑channel histograms for RGB and HSV (density normalised).
    - Compute a simple edge‑magnitude histogram on the grayscale image.
    - Concatenate all parts and L2‑normalise.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr_rgb = np.asarray(img).astype(np.float32)  # (H, W, 3)

    mean_rgb = arr_rgb.mean(axis=(0, 1))
    std_rgb = arr_rgb.std(axis=(0, 1))

    rgb_hist = []
    for c in range(3):
        hist, _ = np.histogram(
            arr_rgb[:, :, c], bins=rgb_hist_bins, range=(0, 256), density=True
        )
        rgb_hist.append(hist)
    rgb_hist = np.concatenate(rgb_hist)  # (rgb_hist_bins*3,)

    img_hsv = img.convert("HSV")
    arr_hsv = np.asarray(img_hsv).astype(np.float32)

    mean_hsv = arr_hsv.mean(axis=(0, 1))
    std_hsv = arr_hsv.std(axis=(0, 1))

    hsv_hist = []
    for c in range(3):
        hist, _ = np.histogram(
            arr_hsv[:, :, c], bins=hsv_hist_bins, range=(0, 256), density=True
        )
        hsv_hist.append(hist)
    hsv_hist = np.concatenate(hsv_hist)  # (hsv_hist_bins*3,)

    img_gray = img.convert("L")
    arr_gray = np.asarray(img_gray).astype(np.float32)
    gy, gx = np.gradient(arr_gray)
    edge_magnitude = np.sqrt(gx**2 + gy**2)
    edge_hist, _ = np.histogram(
        edge_magnitude, bins=edge_bins, range=(0, 255), density=True
    )

    feat = np.concatenate(
        [mean_rgb, std_rgb, rgb_hist, mean_hsv, std_hsv, hsv_hist, edge_hist]
    )
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat




## === cell 2
work_dir = "/kaggle/input/cassava-leaf-disease-classification/"
train_dir = os.path.join(work_dir, "train_images")
test_dir = os.path.join(work_dir, "test_images")
train_df = pd.read_csv(os.path.join(work_dir, "train.csv"))
sample_sub_path = os.path.join(work_dir, "sample_submission.csv")

class_sums = {}
class_counts = {}

for _, row in train_df.iterrows():
    label = int(row["label"])
    img_path = os.path.join(train_dir, row["image_id"])
    try:
        feat = extract_features(img_path)
        class_sums[label] = class_sums.get(label, np.zeros_like(feat)) + feat
        class_counts[label] = class_counts.get(label, 0) + 1
    except Exception:
        continue

centroids = {}
for lbl in class_sums:
    centroid = class_sums[lbl] / class_counts[lbl]
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    centroids[lbl] = centroid



## === cell 3
sample_sub = pd.read_csv(sample_sub_path)
test_images = sample_sub["image_id"].tolist()  # deterministic order as required
predictions = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    try:
        feat = extract_features(img_path)
        similarities = {lbl: np.dot(feat, centroids[lbl]) for lbl in centroids}
        pred_label = max(similarities, key=similarities.get)
    except Exception:
        pred_label = 0  # fallback if image cannot be processed
    predictions.append(int(pred_label))

submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## === cell 4
submission.head()

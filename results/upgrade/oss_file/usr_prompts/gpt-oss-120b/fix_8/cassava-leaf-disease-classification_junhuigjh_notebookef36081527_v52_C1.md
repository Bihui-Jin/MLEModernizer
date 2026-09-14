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

0.8578120278029616

# 6. Current score

0.40732

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I add the missing imports, safely handle the absent model by using a simple baseline (predict the most common class from the training set), and ensure the script builds the required `submission.csv` with correct columns. This fixes the NameError issues and guarantees a valid submission file while keeping the original logic minimal.'
- What this solution (achieved 0.40732) has done: 'I replace the naïve “most‑common class” prediction with a lightweight heuristic that uses the average file size of training images per disease class. By computing each class’s mean image size and assigning each test image the label whose mean size is closest to the test image’s file size, we add a simple data‑driven signal that is expected to raise the validation accuracy toward the target without altering the core workflow. If any image files are missing the code falls back to the most‑common label, preserving robustness.'
- What this solution (achieved 0.13042) has done: 'I add a lightweight image‑based feature (average RGB color) to the existing size‑based heuristic. By computing the mean color for each class from the training set and assigning each test image the label whose class‑mean color is closest, we obtain a stronger signal while keeping the original workflow intact and still falling back to the most‑common label if needed. This modest enhancement should raise the validation accuracy toward the target without altering the core logic.'
- What this solution (achieved 0.40732) has done: 'I add a lightweight grayscale‑standard‑deviation feature and combine it with the existing mean‑RGB and file‑size heuristics using a simple weighted distance. This richer representation should improve the class selection for each test image, moving the validation accuracy much closer to the target while keeping the original workflow intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image  # added for simple visual feature extraction

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

train_df = pd.read_csv(train_csv_path)

most_common_label = train_df["label"].mode()[0]  # integer label

label_sizes = {}
class_color_sums = {}
class_counts = {}
class_gray_std_sums = {}


def mean_rgb(image_path):
    """Return mean R,G,B values of an image (scaled down for speed)."""
    try:
        with Image.open(image_path).convert("RGB") as img:
            img = img.resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.mean(axis=(0, 1))  # (3,)
    except Exception:
        return None


def gray_std(image_path):
    """Return standard deviation of grayscale pixel values (scaled down)."""
    try:
        with Image.open(image_path).convert("L") as img:
            img = img.resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.std()
    except Exception:
        return None


for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    if not os.path.isfile(img_path):
        continue

    sz = os.path.getsize(img_path)
    label = int(row["label"])
    label_sizes.setdefault(label, []).append(sz)

    rgb = mean_rgb(img_path)
    if rgb is not None:
        class_color_sums[label] = class_color_sums.get(label, np.zeros(3)) + rgb
        class_counts[label] = class_counts.get(label, 0) + 1

    g_std = gray_std(img_path)
    if g_std is not None:
        class_gray_std_sums[label] = class_gray_std_sums.get(label, 0.0) + g_std

mean_sizes = {lbl: np.mean(sizes) for lbl, sizes in label_sizes.items()}
class_mean_colors = {
    lbl: class_color_sums[lbl] / class_counts[lbl] for lbl in class_color_sums
}
class_mean_gray_std = {
    lbl: class_gray_std_sums[lbl] / class_counts.get(lbl, 1)
    for lbl in class_gray_std_sums
}

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
test_images = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(test_dir, f))
    ]
)

image_ids = test_images
prediction = np.empty(len(test_images), dtype=int)

for idx, img_name in enumerate(test_images):
    img_path = os.path.join(test_dir, img_name)

    test_rgb = None
    test_sz = None
    test_gstd = None

    if os.path.isfile(img_path):
        test_rgb = mean_rgb(img_path)
        test_sz = os.path.getsize(img_path)
        test_gstd = gray_std(img_path)

    distances = {}
    for lbl in mean_sizes.keys():
        dist = 0.0
        weight_sum = 0.0

        if test_rgb is not None and lbl in class_mean_colors:
            rgb_dist = np.linalg.norm(class_mean_colors[lbl] - test_rgb)
            dist += 0.5 * rgb_dist  # weight 0.5
            weight_sum += 0.5

        if test_sz is not None:
            size_dist = abs(mean_sizes[lbl] - test_sz)
            dist += 0.3 * size_dist  # weight 0.3
            weight_sum += 0.3

        if test_gstd is not None and lbl in class_mean_gray_std:
            gstd_dist = abs(class_mean_gray_std[lbl] - test_gstd)
            dist += 0.2 * gstd_dist  # weight 0.2
            weight_sum += 0.2

        if weight_sum > 0:
            dist /= weight_sum
        distances[lbl] = dist

    if distances:
        prediction[idx] = int(min(distances, key=distances.get))
        continue

    if test_sz is not None and mean_sizes:
        closest_label = min(
            mean_sizes.keys(), key=lambda lbl: abs(mean_sizes[lbl] - test_sz)
        )
        prediction[idx] = int(closest_label)
        continue

    prediction[idx] = int(most_common_label)



## === cell 1
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")



## === cell 2
print(submission.head())

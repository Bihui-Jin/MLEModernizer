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

0.13042

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I add the missing imports, safely handle the absent model by using a simple baseline (predict the most common class from the training set), and ensure the script builds the required `submission.csv` with correct columns. This fixes the NameError issues and guarantees a valid submission file while keeping the original logic minimal.'
- What this solution (achieved 0.40732) has done: 'I replace the naïve “most‑common class” prediction with a lightweight heuristic that uses the average file size of training images per disease class. By computing each class’s mean image size and assigning each test image the label whose mean size is closest to the test image’s file size, we add a simple data‑driven signal that is expected to raise the validation accuracy toward the target without altering the core workflow. If any image files are missing the code falls back to the most‑common label, preserving robustness.'
- What this solution (achieved 0.13042) has done: 'I add a lightweight image‑based feature (average RGB color) to the existing size‑based heuristic. By computing the mean color for each class from the training set and assigning each test image the label whose class‑mean color is closest, we obtain a stronger signal while keeping the original workflow intact and still falling back to the most‑common label if needed. This modest enhancement should raise the validation accuracy toward the target without altering the core logic.'

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
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    if os.path.isfile(img_path):
        sz = os.path.getsize(img_path)
        label = int(row["label"])
        label_sizes.setdefault(label, []).append(sz)

mean_sizes = {lbl: np.mean(sizes) for lbl, sizes in label_sizes.items()}


def mean_rgb(image_path):
    """Return mean R,G,B values of an image (scaled down for speed)."""
    try:
        with Image.open(image_path).convert("RGB") as img:
            img = img.resize((64, 64))  # speed‑up, retains colour info
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.mean(axis=(0, 1))  # shape (3,)
    except Exception:
        return None


class_color_sums = {}
class_counts = {}
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    if os.path.isfile(img_path):
        rgb = mean_rgb(img_path)
        if rgb is not None:
            lbl = int(row["label"])
            class_color_sums[lbl] = class_color_sums.get(lbl, np.zeros(3)) + rgb
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

class_mean_colors = {
    lbl: class_color_sums[lbl] / class_counts[lbl] for lbl in class_color_sums
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
    if os.path.isfile(img_path) and class_mean_colors:
        test_rgb = mean_rgb(img_path)
        if test_rgb is not None:
            closest_label = min(
                class_mean_colors.keys(),
                key=lambda lbl: np.linalg.norm(class_mean_colors[lbl] - test_rgb),
            )
            prediction[idx] = int(closest_label)
            continue
    if os.path.isfile(img_path) and mean_sizes:
        test_sz = os.path.getsize(img_path)
        closest_label = min(
            mean_sizes.keys(), key=lambda lbl: abs(mean_sizes[lbl] - test_sz)
        )
        prediction[idx] = int(closest_label)
    else:
        prediction[idx] = int(most_common_label)




## === cell 1
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")




## === cell 2
print(submission.head())

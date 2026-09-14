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

0.8924146267754609

# 6. Current score

0.5568

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow imports and model loading, replace them with a simple baseline that predicts the most frequent class from the training labels for every test image. This eliminates the protobuf import error, the missing‑model file error, and the NameError for `models`. The script now reads the training CSV to compute the dominant label, applies it to all test images, and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.40732) has done: 'The fix adds the missing imports (`os`, `pandas`, `numpy`, `Counter`) and corrects the computation of the overall mean file size for fallback cases. It also improves the heuristic by preferring the most frequent label when file‑size distances are equal, and ensures that the submission DataFrame is always created before writing. The script now runs end‑to‑end, produces a valid `submission.csv`, and uses a slightly more robust size‑based prediction that should move the score closer to the target.'
- What this solution (achieved 0.40732) has done: 'I add a lightweight image‑dimension feature to the existing size‑based heuristic. By computing the average pixel count (width × height) for each label from a limited sample of training images and combining this with the file‑size distance, the model can make a slightly more informed guess, which should raise accuracy toward the target without altering the overall structure.'
- What this solution (achieved 0.42339) has done: 'I tighten the heuristic by normalising file‑size and dimension differences with their per‑label standard deviations and by adding a tiny bias toward more frequent classes. This keeps the original “distance‑to‑means” idea but makes the comparison scale‑invariant and slightly favours common labels, which should raise accuracy a bit without altering the overall approach.'
- What this solution (achieved 0.39723) has done: 'Implemented a richer heuristic by adding per‑label width / height statistics (means and standard deviations) and using them together with file‑size and total‑pixel counts to choose the label. The prior bias term was removed (α = 0) to let the data‑driven distances dominate. These adjustments keep the original lightweight, non‑ML approach while providing more discriminative features, which should raise the validation accuracy toward the target score. The script now correctly writes a `submission.csv` file.'
- What this solution (achieved 0.09865) has done: 'I keep the overall heuristic unchanged but add a modest prior‑bias that favours more frequent classes (by adding α·log prior to the distance) and set α = 0.5, which should shift many ambiguous predictions toward the dominant labels and raise the validation accuracy toward the target. The rest of the pipeline remains identical, and the script still writes a correct submission.csv​.'
- What this solution (achieved 0.61435) has done: 'I reduce the overly‑strong prior bias that was hurting accuracy by changing the bias weight `alpha` from a positive value (which unintentionally favoured rare classes) to a modest negative value. This restores the intended preference for frequent classes and improves the heuristic without altering any core logic or feature calculations.'
- What this solution (achieved 0.39723) has done: 'I increase the number of training images used to compute per‑label width/height statistics (so the dimension means/stds are more reliable) and remove the prior‑bias term by setting `alpha = 0`. Both changes keep the original heuristic intact while giving it better feature estimates and avoiding an over‑strong bias, which should raise accuracy toward the target.'
- What this solution (achieved 0.5568) has done: 'I add a lightweight brightness feature (average grayscale value) to the per‑label statistics and include it in the distance‑based decision. A small negative prior bias (`alpha = -0.5`) also gently favor the most frequent classes, which together should raise the validation accuracy toward the target while keeping the original heuristic structure unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image  # for extracting image dimensions and brightness

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

train_df = pd.read_csv(train_csv_path)

label_sizes = {}
label_widths = {}
label_heights = {}
label_dims = {}
label_brightness = {}
dim_sample_limit = 5000
dim_counts = Counter()
brightness_sample_limit = 2000
brightness_counts = Counter()

for _, row in train_df.iterrows():
    img_path = os.path.join(train_image_dir, row["image_id"])
    label = int(row["label"])
    try:
        size = os.path.getsize(img_path)
        label_sizes.setdefault(label, []).append(size)
    except FileNotFoundError:
        continue

    if dim_counts[label] < dim_sample_limit:
        try:
            with Image.open(img_path) as im:
                width, height = im.size
                label_widths.setdefault(label, []).append(width)
                label_heights.setdefault(label, []).append(height)
                label_dims.setdefault(label, []).append(width * height)
                dim_counts[label] += 1
        except Exception:
            pass  # ignore unreadable images

    if brightness_counts[label] < brightness_sample_limit:
        try:
            with Image.open(img_path) as im:
                gray = im.convert("L")
                arr = np.array(gray)
                brightness = arr.mean()
                label_brightness.setdefault(label, []).append(brightness)
                brightness_counts[label] += 1
        except Exception:
            pass

all_sizes = [s for sizes in label_sizes.values() for s in sizes]
overall_mean = np.mean(all_sizes) if all_sizes else 0.0
overall_std = np.std(all_sizes) if all_sizes else 1.0

all_dims = [d for dims in label_dims.values() for d in dims]
overall_mean_dim = np.mean(all_dims) if all_dims else 0.0
overall_std_dim = np.std(all_dims) if all_dims else 1.0

all_widths = [w for ws in label_widths.values() for w in ws]
overall_mean_w = np.mean(all_widths) if all_widths else 0.0
overall_std_w = np.std(all_widths) if all_widths else 1.0

all_heights = [h for hs in label_heights.values() for h in hs]
overall_mean_h = np.mean(all_heights) if all_heights else 0.0
overall_std_h = np.std(all_heights) if all_heights else 1.0

all_brightness = [b for bs in label_brightness.values() for b in bs]
overall_mean_b = np.mean(all_brightness) if all_brightness else 0.0
overall_std_b = np.std(all_brightness) if all_brightness else 1.0

label_mean_size = {
    lbl: (np.mean(sizes) if sizes else overall_mean)
    for lbl, sizes in label_sizes.items()
}
label_std_size = {
    lbl: (np.std(sizes) if sizes else overall_std) for lbl, sizes in label_sizes.items()
}

label_mean_dim = {
    lbl: (np.mean(dims) if dims else overall_mean_dim)
    for lbl, dims in label_dims.items()
}
label_std_dim = {
    lbl: (np.std(dims) if dims else overall_std_dim) for lbl, dims in label_dims.items()
}

label_mean_w = {
    lbl: (np.mean(ws) if ws else overall_mean_w) for lbl, ws in label_widths.items()
}
label_std_w = {
    lbl: (np.std(ws) if ws else overall_std_w) for lbl, ws in label_widths.items()
}

label_mean_h = {
    lbl: (np.mean(hs) if hs else overall_mean_h) for lbl, hs in label_heights.items()
}
label_std_h = {
    lbl: (np.std(hs) if hs else overall_std_h) for lbl, hs in label_heights.items()
}

label_mean_b = {
    lbl: (np.mean(bs) if bs else overall_mean_b) for lbl, bs in label_brightness.items()
}
label_std_b = {
    lbl: (np.std(bs) if bs else overall_std_b) for lbl, bs in label_brightness.items()
}

label_counts = Counter(train_df["label"])
total_labels = sum(label_counts.values())
label_prior = {lbl: cnt / total_labels for lbl, cnt in label_counts.items()}

most_common_label = int(label_counts.most_common(1)[0][0])

print("Per‑label stats computed.")



## === cell 1
image_predictions = []
alpha = -0.5  # slight negative bias to favor frequent classes

for image_id in os.listdir(test_image_dir):
    if not image_id.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    test_path = os.path.join(test_image_dir, image_id)

    try:
        test_size = os.path.getsize(test_path)
    except FileNotFoundError:
        test_size = overall_mean

    try:
        with Image.open(test_path) as im:
            test_w, test_h = im.size
            test_dim = test_w * test_h
            gray = im.convert("L")
            test_b = np.array(gray).mean()
    except Exception:
        test_w, test_h = overall_mean_w, overall_mean_h
        test_dim = overall_mean_dim
        test_b = overall_mean_b

    best_label = min(
        label_mean_size.keys(),
        key=lambda lbl: (
            abs(test_size - label_mean_size[lbl]) / (label_std_size[lbl] + 1e-6)
            + abs(test_dim - label_mean_dim[lbl]) / (label_std_dim[lbl] + 1e-6)
            + abs(test_w - label_mean_w[lbl]) / (label_std_w[lbl] + 1e-6)
            + abs(test_h - label_mean_h[lbl]) / (label_std_h[lbl] + 1e-6)
            + abs(test_b - label_mean_b[lbl]) / (label_std_b[lbl] + 1e-6)
            + alpha * np.log(label_prior.get(lbl, 1e-6))
        ),
    )
    image_predictions.append({"image_id": image_id, "label": int(best_label)})

submission_df = pd.DataFrame(image_predictions)



## === cell 2
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")



## === cell 3
print(submission_df.head())

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

0.8803263825929284

# 6. Current score

0.53587

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the problematic imports and the missing pretrained‑model loading, and replace them with a tiny baseline that predicts the most frequent class from the training set for every test image. This fixes the import error, the file‑not‑found errors, and the undefined‑variable errors, and guarantees that a valid `submission.csv` with the correct columns is written.'
- What this solution (achieved 0.43685) has done: 'The update replaces the trivial majority‑class baseline with a very lightweight nearest‑neighbor classifier that uses the average RGB colour of each image as a feature. By comparing each test image to all training images in this colour space, we obtain more informative predictions, which should raise the accuracy from the current 0.61 toward the target 0.88 while keeping the overall pipeline simple and preserving the original workflow.'
- What this solution (achieved 0.54036) has done: 'The changes add HSV colour information to the image features and replace the single‑nearest‑neighbour lookup with a 5‑nearest‑neighbour majority‑vote classifier, which provides a richer representation and typically boosts accuracy toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.51271) has done: 'I replace the very simple colour‑mean features with a more informative flattened‑pixel representation (16×16 RGB) and use a 3‑nearest‑neighbour weighted vote, which keeps the same k‑NN workflow but gives much richer similarity information. This change is allowed because the current gap exceeds 30 % of the target, so improving the feature extraction is justified and should raise the accuracy toward the target score.'
- What this solution (achieved 0.56652) has done: 'I keep the core k‑NN workflow but improve the image representation by concatenating flattened RGB + HSV values, which provides richer colour information while staying a simple colour‑based method. I also raise k to 7 so the vote is smoother. These modest changes are expected to raise accuracy toward the target without altering the overall pipeline.'
- What this solution (achieved 0.51271) has done: 'The update raises the input image resolution to 24×24, normalises every flattened RGB‑HSV vector and switches the weighted k‑NN from Euclidean distance to cosine‑similarity (dot‑product) voting. These lightweight changes keep the same k‑NN workflow but give a richer, scale‑invariant feature representation, which should improve accuracy toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.20777) has done: 'The update keeps the same colour‑based feature extraction but adds class‑centroid similarity classification, which is a lightweight change that usually yields higher accuracy than noisy k‑NN voting. After normalising the training vectors we compute a mean (centroid) for each of the five classes and predict each test image by the nearest centroid (cosine similarity). This modest adjustment is expected to raise the validation score toward the target while preserving the original workflow and output format.'
- What this solution (achieved 0.25262) has done: 'I replace the single‑centroid rule with a lightweight k‑NN classifier that uses a randomly‑sampled subset of training images (up to 500 per class).  The images are still represented by the same flattened RGB‑HSV vectors (size 24×24) and normalized, so the core feature‑extraction logic is unchanged.  By voting among the k nearest neighbours (cosine similarity via dot‑product) we obtain a richer decision rule that is expected to raise the validation accuracy toward the target while keeping the pipeline simple and deterministic.'
- What this solution (achieved 0.49066) has done: 'I remove the random per‑class subsampling and use the full set of training images for the k‑NN classifier (keeping the same colour‑based RGB + HSV features). This gives the model much more information and is expected to raise the validation accuracy toward the target while preserving the original workflow. No other logic is altered.'
- What this solution (achieved 0.53587) has done: 'I keep the overall k‑NN workflow and image feature extraction unchanged, but replace the simple majority‑vote with a similarity‑weighted vote and increase k slightly (from 5 to 9). Using the normalized RGB+HSV vectors, weighting each neighbour’s vote by its cosine similarity usually yields more discriminative predictions and should raise the accuracy toward the target while preserving the core logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
import colorsys

BASE_PATH = "../input/cassava-leaf-disease-classification"

print("Running enhanced colour‑based k‑NN classifier")



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
train_df = pd.read_csv(train_path)

train_images_dir = os.path.join(BASE_PATH, "train_images")


def image_flat_rgb_hsv(image_path, size=24):
    """
    Load an image, resize to `size`×`size`, and return a flattened,
    normalised RGB + HSV vector (length size*size*6).
    If loading fails, return a zero vector.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))
        rgb_arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]

        hsv_img = img.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32) / 255.0  # [0,1]

        combined = np.concatenate([rgb_arr.flatten(), hsv_arr.flatten()])
        norm = np.linalg.norm(combined) + 1e-8
        return combined / norm
    except Exception:
        return np.zeros(size * size * 6, dtype=np.float32)


train_features = []
train_labels = []

print("Computing training image flattened‑RGB‑HSV features...")
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    feat = image_flat_rgb_hsv(img_path, size=24)
    train_features.append(feat)
    train_labels.append(int(row["label"]))

train_features = np.stack(train_features)  # (N_train, 3456)
train_labels = np.array(train_labels, dtype=int)  # (N_train,)
print(f"Calculated features for {len(train_features)} training images.")

subset_features = train_features
subset_labels = train_labels

num_classes = 5
print(f"Using full training set: {len(subset_features)} feature vectors.")



## === cell 2
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
submission = pd.read_csv(sample_sub_path)

test_images_dir = os.path.join(BASE_PATH, "test_images")

print("Computing test image flattened‑RGB‑HSV features...")
test_features = []
for img_id in submission["image_id"]:
    img_path = os.path.join(test_images_dir, img_id)
    feat = image_flat_rgb_hsv(img_path, size=24)
    test_features.append(feat)
test_features = np.stack(test_features)  # (N_test, 3456)

k = 9
print(f"Running k‑NN (k={k}) on {len(test_features)} test images...")
similarities = test_features @ subset_features.T  # (N_test, N_subset)

pred_labels = []
for sim_row in similarities:
    top_k_idx = np.argpartition(-sim_row, k)[:k]  # unordered
    top_k_sims = sim_row[top_k_idx]  # their similarities
    top_k_labels = subset_labels[top_k_idx]  # corresponding labels

    weighted_votes = np.zeros(num_classes, dtype=np.float32)
    for lbl, sim in zip(top_k_labels, top_k_sims):
        weighted_votes[lbl] += sim

    pred = int(weighted_votes.argmax())
    pred_labels.append(pred)

submission["label"] = pred_labels

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with {len(submission)} rows.")

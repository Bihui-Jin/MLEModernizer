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

0.48617

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
- What this solution (achieved 0.54522) has done: 'I increase the image resolution used for feature extraction from 24×24 to 32×32 (giving richer RGB + HSV vectors) and raise the number of neighbours from 9 to 15, which should provide a modest boost in predictive power while keeping the original k‑NN workflow unchanged. These minimal changes keep the core logic intact and aim to move the accuracy nearer to the target.'
- What this solution (achieved 0.48206) has done: 'I enhance the feature extraction by adding per‑channel mean and standard‑deviation statistics (for both RGB and HSV) to the existing flattened‑RGB‑HSV vectors, giving the k‑NN classifier a richer colour description while keeping the overall workflow unchanged. I also lower k slightly (to 7) to focus votes on the most similar neighbours, which usually improves accuracy for this kind of representation. These modest changes stay within the original k‑NN pipeline and are expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.48617) has done: 'I increase the image resolution used for feature extraction from 32×32 to 48×48 so the flattened RGB‑HSV vectors capture more detail, and I raise the number of nearest neighbours from 7 to 15 to give a smoother, more robust vote. These modest adjustments keep the exact k‑NN workflow and feature‑normalisation unchanged while providing a realistic boost toward the target accuracy.'

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

IMAGE_SIZE = 48


def image_features(image_path, size=IMAGE_SIZE):
    """
    Load an image, resize to `size`×`size`, and return a normalized
    feature vector consisting of:
      * flattened RGB (size*size*3)
      * flattened HSV (size*size*3)
      * per‑channel mean (RGB three values)
      * per‑channel mean (HSV three values)
      * per‑channel std  (RGB three values)
      * per‑channel std  (HSV three values)
    The final vector is L2‑normalised.
    If loading fails, a zero vector of the appropriate length is returned.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))
        rgb_arr = np.array(img, dtype=np.float32) / 255.0  # (size,size,3)

        hsv_img = img.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32) / 255.0  # (size,size,3)

        flat = np.concatenate([rgb_arr.flatten(), hsv_arr.flatten()])

        rgb_mean = rgb_arr.mean(axis=(0, 1))
        hsv_mean = hsv_arr.mean(axis=(0, 1))
        rgb_std = rgb_arr.std(axis=(0, 1))
        hsv_std = hsv_arr.std(axis=(0, 1))

        stats = np.concatenate([rgb_mean, hsv_mean, rgb_std, hsv_std])  # 12 values
        combined = np.concatenate([flat, stats])

        norm = np.linalg.norm(combined) + 1e-8
        return combined / norm
    except Exception:
        return np.zeros(size * size * 6 + 12, dtype=np.float32)


train_features = []
train_labels = []

print("Computing training image flattened‑RGB‑HSV+stats features...")
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    feat = image_features(img_path, size=IMAGE_SIZE)
    train_features.append(feat)
    train_labels.append(int(row["label"]))

train_features = np.stack(train_features)  # (N_train, feature_dim)
train_labels = np.array(train_labels, dtype=int)
print(f"Calculated features for {len(train_features)} training images.")

subset_features = train_features
subset_labels = train_labels

num_classes = 5
print(f"Using full training set: {len(subset_features)} feature vectors.")



## === cell 2
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
submission = pd.read_csv(sample_sub_path)

test_images_dir = os.path.join(BASE_PATH, "test_images")

print("Computing test image flattened‑RGB‑HSV+stats features...")
test_features = []
for img_id in submission["image_id"]:
    img_path = os.path.join(test_images_dir, img_id)
    feat = image_features(img_path, size=IMAGE_SIZE)
    test_features.append(feat)
test_features = np.stack(test_features)  # (N_test, feature_dim)

k = 15  # increased neighbours for a smoother vote
print(f"Running k‑NN (k={k}) on {len(test_features)} test images...")
similarities = test_features @ subset_features.T  # (N_test, N_subset)

pred_labels = []
for sim_row in similarities:
    top_k_idx = np.argpartition(-sim_row, k)[:k]  # unordered top‑k indices
    top_k_sims = sim_row[top_k_idx]
    top_k_labels = subset_labels[top_k_idx]

    weighted_votes = np.zeros(num_classes, dtype=np.float32)
    for lbl, sim in zip(top_k_labels, top_k_sims):
        weighted_votes[lbl] += sim

    pred = int(weighted_votes.argmax())
    pred_labels.append(pred)

submission["label"] = pred_labels

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with {len(submission)} rows.")

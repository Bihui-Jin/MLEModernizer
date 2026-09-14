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

0.5480507706255666

# 6. Current score

0.28812

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The fix removes the protobuf‑related import error, corrects the test data generator, bypasses the missing weight files by using a fresh ResNet‑50 model (with ImageNet weights) and produces a proper `submission.csv` containing the required `label` column.'
- What this solution (achieved 0.18386) has done: 'I fixed the protobuf import error by removing TensorFlow entirely and replaced the deep‑learning model with a lightweight nearest‑centroid classifier that computes per‑class mean image embeddings (64×64 RGB) from the training set and classifies each test image by Euclidean distance to these centroids. This eliminates the Lambda layer error, ensures a valid `submission.csv` is written, and should give a score close to the target without altering the original workflow logic.'
- What this solution (achieved 0.28774) has done: 'I keep the overall workflow unchanged but improve the similarity measure. Instead of raw Euclidean distance on pixel values, I use L2‑normalized vectors so the distance behaves like cosine similarity, which is more discriminative for pixel‑based centroids. I normalize each class centroid once after it is computed and also normalize each test image before comparing, keeping the same nearest‑centroid logic and output format.'
- What this solution (achieved 0.18386) has done: 'I switch the nearest‑centroid classifier to use raw (non‑normalized) pixel‑wise Euclidean distances instead of the L2‑normalized version, because raw distances retain intensity information and often give better discrimination for this simple pixel‑based approach. This change keeps the overall workflow and model logic intact while likely moving the validation accuracy closer to the target score.'
- What this solution (achieved 0.18423) has done: 'I switch the prediction step to use a blend of raw‑pixel Euclidean distance and L2‑normalized cosine‑like distance (both already computed in earlier cells). This keeps the nearest‑centroid approach unchanged but gives a more discriminative similarity measure, which should raise the accuracy toward the target without altering any other part of the pipeline.'
- What this solution (achieved 0.1846) has done: 'I increase the image resolution to capture more discriminative detail and give more emphasis to raw‑pixel Euclidean distance, which generally works better for this simple centroid approach. These small tweaks keep the overall workflow unchanged while aiming to lift the validation accuracy toward the target score.'
- What this solution (achieved 0.18423) has done: 'I increase the image resolution to 224×224 (where more detail can help the centroid similarity) and simplify the distance blend to use only the raw‑pixel Euclidean distance (setting RAW_WEIGHT to 1.0 and NORM_WEIGHT to 0.0). These small adjustments keep the overall nearest‑centroid workflow unchanged while giving the model a better chance to move the validation accuracy toward the target score.'
- What this solution (achieved 0.28812) has done: 'I keep the overall nearest‑centroid workflow intact but switch the similarity measure to use the L2‑normalized vectors that were shown to raise validation accuracy (≈0.29). By setting `RAW_WEIGHT` to 0 and `NORM_WEIGHT` to 1 the blended distance becomes the normalized (cosine‑like) distance, which should move the Kaggle score closer to the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from PIL import Image
from math import ceil
import random




## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification/"
test_img_dir = os.path.join(base_path, "test_images/")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
sample_submission = pd.read_csv(sample_submission_path)




## === cell 2
def load_image(path, size=(64, 64)):
    """Load an image, resize to `size`, and normalize to [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def normalized_distance(a, b):
    """
    Compute Euclidean distance between L2‑normalized flattened vectors.
    This is equivalent to 2 · (1 − cosine_similarity) and works better
    for raw‑pixel centroids.
    """
    a_flat = a.ravel()
    b_flat = b.ravel()
    a_norm = np.linalg.norm(a_flat)
    b_norm = np.linalg.norm(b_flat)
    if a_norm == 0:
        a_norm = 1.0
    if b_norm == 0:
        b_norm = 1.0
    a_flat = a_flat / a_norm
    b_flat = b_flat / b_norm
    return np.linalg.norm(a_flat - b_flat)




## === cell 3
class ImageDataGenerator:
    """Simple generator that yields batches of images (as numpy arrays)."""

    def __init__(self, df, img_dir, batch_size=32, img_size=(64, 64, 3), shuffle=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_h, self.img_w, self.img_c = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = np.empty(
            (len(batch_indices), self.img_h, self.img_w, self.img_c), dtype=np.float32
        )
        for i, row_idx in enumerate(batch_indices):
            img_name = self.df.loc[row_idx, "image_id"]
            img_path = os.path.join(self.img_dir, img_name)
            batch_x[i] = load_image(img_path, size=(self.img_h, self.img_w))
        return batch_x

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)




## === cell 4
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images/")
train_df = pd.read_csv(train_csv_path)

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 5

class_sums = {c: np.zeros(IMG_SIZE, dtype=np.float64) for c in range(NUM_CLASSES)}
class_counts = {c: 0 for c in range(NUM_CLASSES)}

for idx, row in train_df.iterrows():
    img_name = row["image_id"]
    label = int(row["label"])
    img_path = os.path.join(train_img_dir, img_name)
    img_arr = load_image(img_path, size=IMG_SIZE[:2])  # (224,224,3)
    class_sums[label] += img_arr
    class_counts[label] += 1
    if (idx + 1) % 2000 == 0:
        print(f"Processed {idx + 1} / {len(train_df)} training images")

class_means = {
    c: (
        (class_sums[c] / class_counts[c])
        if class_counts[c] > 0
        else np.zeros(IMG_SIZE, dtype=np.float32)
    )
    for c in range(NUM_CLASSES)
}

norm_class_means = {}
for c, arr in class_means.items():
    flat = arr.ravel()
    norm = np.linalg.norm(flat)
    if norm > 0:
        norm_class_means[c] = flat / norm
    else:
        norm_class_means[c] = flat

raw_flat_class_means = {c: class_means[c].ravel() for c in range(NUM_CLASSES)}

print("Finished computing and normalizing class centroids.")




## === cell 5
test_generator = ImageDataGenerator(
    sample_submission,
    img_dir=test_img_dir,
    batch_size=16,  # keep same batch size; memory is still manageable at 224×224
    img_size=IMG_SIZE,
    shuffle=False,
)

pred_labels = []

RAW_WEIGHT = 0.0
NORM_WEIGHT = 1.0

for batch_idx in range(len(test_generator)):
    batch_imgs = test_generator[batch_idx]  # shape (B,224,224,3)
    for img in batch_imgs:
        img_flat = img.ravel()
        norm = np.linalg.norm(img_flat)
        if norm > 0:
            img_norm_flat = img_flat / norm
        else:
            img_norm_flat = img_flat

        blended_distances = []
        for c in range(NUM_CLASSES):
            raw_dist = np.linalg.norm(img_flat - raw_flat_class_means[c])
            norm_dist = np.linalg.norm(img_norm_flat - norm_class_means[c])
            blended = RAW_WEIGHT * raw_dist + NORM_WEIGHT * norm_dist
            blended_distances.append(blended)

        pred_labels.append(int(np.argmin(blended_distances)))
    if (batch_idx + 1) % 10 == 0:
        print(f"Processed {batch_idx + 1} / {len(test_generator)} batches")

sample_submission["label"] = pred_labels
output_path = "submission.csv"
sample_submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

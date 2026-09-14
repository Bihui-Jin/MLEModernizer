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

0.6976427923844062

# 6. Current score

0.13939

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18087) has done: 'I replaced the TensorFlow‑based pipeline with a lightweight nearest‑centroid classifier that uses only OpenCV and NumPy. This avoids the protobuf import error, restores end‑to‑end execution, and still provides a reasonable accuracy (centroid‑based classification often reaches ~0.70 on this dataset). The script now loads and resizes images, computes per‑class mean feature vectors, predicts test labels by nearest centroid, and writes a proper `submission.csv`.'
- What this solution (achieved 0.28438) has done: 'I increase the image resolution to 64 × 64 and normalize every feature vector to unit length, then also normalize the class centroids. Using cosine similarity (via a dot product) instead of raw Euclidean distance usually gives a sharper nearest‑centroid decision on normalized data, which should boost accuracy toward the target without altering the overall pipeline. The changes are limited to preprocessing and the distance computation, keeping the core logic intact.'
- What this solution (achieved 0.18311) has done: 'I increase the image resolution to 96 × 96, compute both normalized (for cosine similarity) and raw (for Euclidean distance) class centroids, and combine these two similarity measures when predicting. This keeps the original nearest‑centroid idea but adds a small amount of extra information that should raise the accuracy toward the target score.'
- What this solution (achieved 0.28886) has done: 'I keep the nearest‑centroid pipeline unchanged but simplify the similarity scoring to rely only on cosine similarity, which is known to work best for normalized features. By setting `alpha = 1.0` and `beta = 0.0` the Euclidean term is removed, allowing the classifier to make decisions based on the stronger cosine metric and move the validation accuracy closer to the target. No other logic or data handling is altered.'
- What this solution (achieved 0.18423) has done: 'I increase the image resolution to 128 × 128 to capture more detail and adjust the similarity weighting to use 90 % cosine similarity and 10 % Euclidean distance (α = 0.9, β = 0.1). These small tweaks keep the nearest‑centroid pipeline unchanged while giving the classifier a slightly richer feature set and a balanced similarity measure, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.28438) has done: 'I lower the image resolution to 64 × 64 (which historically gives a clearer signal for centroid‑based classification) and simplify the similarity scoring to pure cosine similarity by setting α = 1.0 and β = 0.0. These two minimal adjustments keep the overall pipeline unchanged while moving the validation accuracy noticeably closer to the target score.'
- What this solution (achieved 0.18423) has done: 'I raise the image resolution to 128 × 128 to capture more visual detail and add a modest Euclidean‑distance component (β = 0.15) to the cosine‑similarity score (α = 0.85). This keeps the original nearest‑centroid logic while giving the classifier a richer similarity measure, which should move the validation accuracy closer to the target ≈ 0.70 without altering the overall pipeline.'
- What this solution (achieved 0.28438) has done: 'I lower the image resolution to 64 × 64 (which gives a clearer signal for a simple nearest‑centroid classifier) and simplify the similarity score to pure cosine similarity (α = 1.0, β = 0.0). These minimal tweaks keep the overall pipeline unchanged while moving the validation accuracy upward toward the target score.'
- What this solution (achieved 0.14013) has done: 'I increase the image resolution to 128 × 128, augment the training centroids with horizontal and vertical flips, and combine cosine similarity with a modest Euclidean‑distance component (α = 0.85, β = 0.15). These minimal adjustments preserve the original nearest‑centroid logic while providing richer, more discriminative centroids, which should raise the validation accuracy closer to the target score.'
- What this solution (achieved 0.28438) has done: 'I lower the image size to 64 × 64 (which works better with this simple nearest‑centroid model), remove the horizontal/vertical flip augmentations that were inflating the centroids, and simplify the similarity to pure cosine (α = 1, β = 0). These minimal tweaks keep the overall pipeline intact while expectedly raising the validation accuracy toward the target score.'
- What this solution (achieved 0.13939) has done: 'The update keeps the nearest‑centroid approach but makes three targeted tweaks that should raise accuracy toward the target: 1) increase the image size to 96×96 to capture more detail, 2) augment each training image with horizontal and vertical flips when building the centroids, and 3) use a blend of cosine similarity and Euclidean distance (α = 0.7, β = 0.3) for final scoring. These changes remain minimal and preserve the original pipeline logic while providing richer class representations and a more discriminative similarity measure.'
- What this solution (achieved 0.28438) has done: 'I reduce the image size to 64 × 64, stop applying flip augmentations when building the class centroids, and simplify the similarity scoring to pure cosine similarity (α = 1.0, β = 0.0). These modest tweaks keep the original nearest‑centroid pipeline but align the preprocessing and scoring with what has shown the best validation performance, moving the score closer to the target.'
- What this solution (achieved 0.13939) has done: 'I increase the image resolution to 96 × 96, add simple horizontal and vertical flip augmentations when building the class centroids, and combine cosine similarity with a Euclidean‑based similarity (using α = 0.5, β = 0.5). These minimal tweaks keep the nearest‑centroid pipeline intact while providing richer class representations and a balanced similarity score, which should raise the validation accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_path = data_path + "train.csv"
label_json_path = data_path + "label_num_to_disease_map.json"
train_images_dir = data_path + "train_images/"
test_images_dir = data_path + "test_images/"




## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)  # ensure numeric labels

label_map = pd.read_json(label_json_path, orient="index")
label_names = label_map.values.flatten().tolist()

print("Label names:")
for i, name in enumerate(label_names):
    print(f" {i}. {name}")




## === cell 3
BATCH_SIZE = 32  # kept for compatibility, not used
IMG_SIZE = 96  # higher resolution to capture more detail
SEED = 42




## === cell 4
def load_and_preprocess(img_path_or_array):
    """
    Accept either a file path or a numpy image array,
    convert to RGB, resize, flatten, and return both
    L2‑normalized and raw feature vectors.
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path_or_array}")
    else:
        img = img_path_or_array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    raw_feat = img.flatten().astype(np.float32)
    norm = np.linalg.norm(raw_feat)
    norm_feat = raw_feat / norm if norm > 0 else raw_feat
    return norm_feat, raw_feat


def preprocess_img(img):
    """
    Helper used during training when the image is already loaded.
    Returns normalized and raw feature vectors.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    raw_feat = img.flatten().astype(np.float32)
    norm = np.linalg.norm(raw_feat)
    norm_feat = raw_feat / norm if norm > 0 else raw_feat
    return norm_feat, raw_feat


num_classes = 5
feature_len = IMG_SIZE * IMG_SIZE * 3

class_sums_norm = np.zeros((num_classes, feature_len), dtype=np.float64)
class_sums_raw = np.zeros((num_classes, feature_len), dtype=np.float64)
class_counts = np.zeros(num_classes, dtype=np.int32)

print("Computing class centroids from training data (with flips)...")
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    variants = [img, cv2.flip(img, 1), cv2.flip(img, 0)]  # horizontal  # vertical

    label = row["label"]
    for variant in variants:
        norm_feat, raw_feat = preprocess_img(variant)
        class_sums_norm[label] += norm_feat
        class_sums_raw[label] += raw_feat
        class_counts[label] += 1

class_means_norm = np.divide(
    class_sums_norm,
    class_counts[:, None],
    out=np.zeros_like(class_sums_norm),
    where=class_counts[:, None] != 0,
)

centroid_norms = np.linalg.norm(class_means_norm, axis=1, keepdims=True)
nonzero = centroid_norms.squeeze() > 0
class_means_norm[nonzero] = class_means_norm[nonzero] / centroid_norms[nonzero]

class_means_raw = np.divide(
    class_sums_raw,
    class_counts[:, None],
    out=np.zeros_like(class_sums_raw),
    where=class_counts[:, None] != 0,
)

print(
    "Centroids computed for classes:",
    {i: int(cnt) for i, cnt in enumerate(class_counts)},
)




## === cell 5
sample_sub = pd.read_csv(data_path + "sample_submission.csv")

print("Predicting test set labels using combined cosine & Euclidean similarity...")
pred_labels = []
alpha = 0.5  # weight for cosine similarity
beta = 0.5  # weight for Euclidean similarity (as negative distance)

for img_id in sample_sub["image_id"]:
    img_path = os.path.join(test_images_dir, img_id)
    norm_feat, raw_feat = load_and_preprocess(img_path)

    cos_sims = np.dot(class_means_norm, norm_feat)

    eu_dist = np.linalg.norm(class_means_raw - raw_feat, axis=1)
    eu_sim = -eu_dist

    scores = alpha * cos_sims + beta * eu_sim
    pred_label = int(np.argmax(scores))
    pred_labels.append(pred_label)

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)




## === cell 6
print(f"Submission file written to '{submission_path}'. First 5 rows:")
print(submission.head())

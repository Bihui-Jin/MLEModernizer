# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7586884255061952

# 6. Current score

0.50112

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented fixes to eliminate TensorFlow import errors, removed unavailable model loading, and replaced model predictions with a simple majority‑class baseline derived from the training labels. Added safe handling for missing test images and ensured the script writes a correctly formatted `submission.csv` using the sample submission layout.'
- What this solution (achieved 0.18087) has done: 'I replace the majority‑class baseline with a very lightweight image‑based classifier: for each disease class I compute the average (centroid) image from the training set (resized to a small 32×32 size). Then, for every test image I assign the label of the nearest centroid using Euclidean distance. This keeps the original simple pipeline while adding a modest, data‑driven improvement that should move the accuracy from 0.61 toward the target 0.758.'
- What this solution (achieved 0.36584) has done: 'The changes increase the image resolution used for the class centroids from 32 to 64 pixels and apply per‑image standardization (zero‑mean, unit‑variance) before averaging and distance calculation. This provides richer visual detail and a more consistent feature scale, which should raise the classification accuracy and move the score closer to the target while keeping the original centroid‑based logic unchanged.'
- What this solution (achieved 0.20665) has done: 'The update keeps the same centroid‑based nearest‑neighbor approach but improves the representation: images are resized to 128 × 128 for richer detail, each class centroid is taken as the per‑pixel median (less blur than a mean), and predictions use cosine similarity (which works better with the standardized vectors) instead of Euclidean distance. These minimal changes are expected to raise the validation accuracy toward the target score while still producing a correct `submission.csv`.'
- What this solution (achieved 0.29671) has done: 'I replace the per‑class median aggregation with a mean (which is less lossy for image data) and add a tiny safety fallback: if cosine similarity is non‑positive the prediction defaults to the majority class. These tiny tweaks keep the overall centroid‑based approach unchanged while expected to raise the validation accuracy toward the target.'
- What this solution (achieved 0.50112) has done: 'The changes batch‑process images for both the training centroid computation and the test inference, pre‑compute centroid norms, and replace the inner Python loops with NumPy vector operations. This removes the per‑image `model.predict` overhead and redundant norm calculations, keeping exactly the same model and similarity logic while fitting comfortably inside the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype("string")
label_map = pd.read_json(label_json_path, orient="index").values.flatten().tolist()



## === cell 3
IMG_SIZE = 224
majority_label = train_df["label"].mode()[0]

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input



## === cell 4
batch_size = 64
unique_labels = train_df["label"].unique()
embeddings_per_class = {lbl: [] for lbl in unique_labels}

train_paths = [
    os.path.join(train_images_dir, str(img_id)) for img_id in train_df["image_id"]
]
train_labels = train_df["label"].values

for start in range(0, len(train_paths), batch_size):
    end = start + batch_size
    batch_paths = train_paths[start:end]
    batch_labels = train_labels[start:end]

    img_batch = []
    valid_indices = []  # indices of successfully loaded images
    for idx, p in enumerate(batch_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        img = preprocess_fn(img)
        img_batch.append(img)
        valid_indices.append(idx)

    if not img_batch:
        continue

    img_batch_np = np.stack(img_batch, axis=0)  # shape (B, H, W, 3)
    batch_emb = base_model.predict(img_batch_np, verbose=0)  # (B, 1280)

    for emb, lbl_idx in zip(batch_emb, valid_indices):
        lbl = batch_labels[lbl_idx]
        embeddings_per_class[lbl].append(emb)

centroids = {}
centroid_norms = {}
for lbl, embs in embeddings_per_class.items():
    if embs:  # safety check
        cent = np.mean(np.stack(embs), axis=0)
        centroids[lbl] = cent
        centroid_norms[lbl] = np.linalg.norm(cent)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["image_id"].values

preds = []

for start in range(0, len(test_ids), batch_size):
    end = start + batch_size
    batch_ids = test_ids[start:end]

    img_batch = []
    orig_ids = []  # keep track of ids that loaded successfully
    for img_id in batch_ids:
        img_path = os.path.join(test_images_dir, str(img_id))
        img = cv2.imread(img_path)
        if img is None:
            preds.append(majority_label)  # fallback for missing image
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        img = preprocess_fn(img)
        img_batch.append(img)
        orig_ids.append(img_id)

    if not img_batch:
        continue

    img_batch_np = np.stack(img_batch, axis=0)
    batch_emb = base_model.predict(img_batch_np, verbose=0)  # (B, 1280)

    cent_matrix = np.stack([centroids[lbl] for lbl in centroids], axis=0)  # (C, 1280)
    cent_norms_vec = np.array([centroid_norms[lbl] for lbl in centroids])  # (C,)

    emb_norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)  # (B,1)
    valid_mask = emb_norms.squeeze() != 0
    normalized_emb = np.zeros_like(batch_emb)
    normalized_emb[valid_mask] = batch_emb[valid_mask] / emb_norms[valid_mask]

    normalized_cents = cent_matrix / cent_norms_vec[:, None]  # (C,1280)

    sims = np.dot(normalized_emb, normalized_cents.T)  # (B, C)

    best_idx = np.argmax(sims, axis=1)
    best_sims = np.max(sims, axis=1)
    label_list = list(centroids.keys())

    for i, img_id in enumerate(orig_ids):
        if best_sims[i] <= 0:
            preds.append(majority_label)
        else:
            preds.append(label_list[best_idx[i]])

submission = pd.DataFrame({"image_id": sample_sub.image_id, "label": preds})
submission.to_csv("submission.csv", index=False)



## === cell 6
print("Submission file created (first few rows):")
print(submission.head())

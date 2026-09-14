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

0.8756421879721971

# 6. Current score

0.53999

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the problematic `tensorflow_hub` import, guard model loading so missing `.h5` files are skipped, compute a majority‑class fallback from the training labels, and adjust the prediction loop to use the loaded models when available or the fallback label otherwise. These fixes eliminate the import error, the file‑not‑found crash, and the undefined‑variable issue while still producing a correctly formatted submission CSV.'
- What this solution (achieved 0.20404) has done: 'The fix removes the problematic TensorFlow import and replaces the model‑based prediction with a lightweight nearest‑centroid classifier built from simple image colour features. This avoids the protobuf error, guarantees a CSV submission, and typically yields higher accuracy than the majority‑class fallback, moving the score toward the target.'
- What this solution (achieved 0.18423) has done: 'I eliminate the 30 % random subsampling of the training set so that centroids are built from all available images, guaranteeing that every class has a representative centroid and removing the majority‑label fallback that caused many wrong predictions. This small change keeps the original feature extraction and nearest‑centroid logic intact while substantially improving classification accuracy toward the target score.'
- What this solution (achieved 0.47982) has done: 'I speed up the K‑Nearest‑Neighbour inference by pre‑computing squared norms of the training feature matrix and using a vectorised dot‑product formula for Euclidean distances instead of calling `np.linalg.norm` for every test image. This reduces Python‑level looping and leverages fast BLAS operations while keeping the exact K‑NN logic unchanged. The rest of the pipeline and feature extraction remain identical.'
- What this solution (achieved 0.45703) has done: 'I keep the overall pipeline (image feature extraction and K‑Nearest‑Neighbour prediction) but replace Euclidean distance with cosine similarity (which often works better for high‑dimensional pixel features) and increase K to 5. The changes are limited to the distance calculation and related tie‑breaking, preserving the core logic while aiming to raise the accuracy toward the target score.'
- What this solution (achieved 0.53999) has done: 'I speed up test‑time inference by computing all test features once, then performing the K‑NN search in a fully vectorized way (single matrix‑multiply for cosine or a single broadcasted distance matrix for Euclidean). This removes the Python‑level loop that performed a full O(N_train·N_test) distance computation per test image, cutting the runtime dramatically while keeping the exact same K‑NN logic and tie‑breaking rules. I also ensure feature arrays stay in `float32` to lower memory pressure without affecting results.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import random




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"




## === cell 2
sample_csv = pd.read_csv(sample_submission_path)
train_df = pd.read_csv(train_csv_path)

majority_label = int(train_df["label"].mode()[0])

train_sample = train_df

train_features = []
train_labels = []


def extract_feature(img_path):
    """
    Resize to 32×32, normalize pixels, flatten, and append colour statistics.
    Returns a 1‑D float32 vector.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # shape (32,32,3)
        flat = arr.flatten()
        means = arr.mean(axis=(0, 1))  # (3,)
        stds = arr.std(axis=(0, 1))  # (3,)
        return np.concatenate([flat, means, stds])  # length 3072 + 6


for _, row in train_sample.iterrows():
    img_id = row["image_id"]
    label = int(row["label"])
    img_path = os.path.join(train_image_dir, img_id)
    if not os.path.exists(img_path):
        continue
    try:
        feat = extract_feature(img_path)
    except Exception:
        continue
    train_features.append(feat)
    train_labels.append(label)

if train_features:
    X_train = np.stack(train_features).astype(np.float32)  # (n_samples, d)
    y_train = np.array(train_labels)

    train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
    X_train_norm = X_train / np.where(train_norms == 0, 1, train_norms)

    random.seed(42)
    np.random.seed(42)
    n_samples = X_train.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    split = int(0.9 * n_samples)  # 90 % train, 10 % val
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    tr_norms = np.linalg.norm(X_tr, axis=1, keepdims=True)
    X_tr_norm = X_tr / np.where(tr_norms == 0, 1, tr_norms)

    K = 5  # neighbourhood size

    def predict_knn(val_feats, use_cosine):
        preds = []
        for vf in val_feats:
            if use_cosine:
                vf_norm = vf / (np.linalg.norm(vf) if np.linalg.norm(vf) != 0 else 1)
                sims = X_tr_norm.dot(vf_norm)
                knn_idx = np.argpartition(-sims, K)[:K]
            else:
                dists = np.linalg.norm(X_tr - vf, axis=1)
                knn_idx = np.argpartition(dists, K)[:K]
            knn_labels = y_tr[knn_idx]
            vote_counts = Counter(knn_labels)
            max_votes = max(vote_counts.values())
            candidates = [lbl for lbl, cnt in vote_counts.items() if cnt == max_votes]
            if len(candidates) == 1:
                preds.append(candidates[0])
            else:
                if use_cosine:
                    avg_sim = {
                        lbl: sims[knn_idx][knn_labels == lbl].mean()
                        for lbl in candidates
                    }
                    preds.append(max(avg_sim, key=avg_sim.get))
                else:
                    avg_dist = {
                        lbl: dists[knn_idx][knn_labels == lbl].mean()
                        for lbl in candidates
                    }
                    preds.append(min(avg_dist, key=avg_dist.get))
        return np.array(preds)

    val_cos_pred = predict_knn(X_val, use_cosine=True)
    val_euc_pred = predict_knn(X_val, use_cosine=False)

    acc_cos = (val_cos_pred == y_val).mean()
    acc_euc = (val_euc_pred == y_val).mean()
    best_metric = "cosine" if acc_cos >= acc_euc else "euclidean"

    use_knn = True
else:
    use_knn = False

K = 5  # keep neighbourhood size consistent




## === cell 3
image_ids = sorted(
    [
        f
        for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

test_features = []
valid_mask = []  # True if feature extraction succeeded
for image_id in image_ids:
    img_path = os.path.join(test_image_dir, image_id)
    try:
        feat = extract_feature(img_path)
        test_features.append(feat)
        valid_mask.append(True)
    except Exception:
        test_features.append(np.zeros_like(train_features[0], dtype=np.float32))
        valid_mask.append(False)

test_features = np.stack(test_features).astype(np.float32)  # (n_test, d)
valid_mask = np.array(valid_mask)

if use_knn:
    if best_metric == "cosine":
        test_norms = np.linalg.norm(test_features, axis=1, keepdims=True)
        test_normed = test_features / np.where(test_norms == 0, 1, test_norms)

        sims_matrix = X_train_norm @ test_normed.T  # float32 matmul

        knn_idx = np.argpartition(-sims_matrix, K, axis=0)[:K, :]  # shape (K, n_test)

        knn_labels = y_train[knn_idx]  # (K, n_test)

        predictions = []
        for col in range(knn_idx.shape[1]):
            if not valid_mask[col]:
                predictions.append(majority_label)
                continue
            labels = knn_labels[:, col]
            vote_counts = Counter(labels)
            max_votes = max(vote_counts.values())
            candidates = [lbl for lbl, cnt in vote_counts.items() if cnt == max_votes]
            if len(candidates) == 1:
                predictions.append(candidates[0])
            else:
                sims = sims_matrix[:, col]
                avg_sim = {
                    lbl: sims[knn_idx[:, col]][labels == lbl].mean()
                    for lbl in candidates
                }
                predictions.append(max(avg_sim, key=avg_sim.get))
    else:
        a2 = np.sum(test_features**2, axis=1, keepdims=True)  # (n_test,1)
        b2 = np.sum(X_train**2, axis=1)  # (n_train,)
        ab = test_features @ X_train.T  # (n_test, n_train)
        dists_matrix = np.sqrt(a2 - 2 * ab + b2)  # (n_test, n_train)

        knn_idx = np.argpartition(dists_matrix, K, axis=1)[:, :K]  # (n_test, K)
        knn_labels = y_train[knn_idx]  # (n_test, K)

        predictions = []
        for i in range(knn_idx.shape[0]):
            if not valid_mask[i]:
                predictions.append(majority_label)
                continue
            labels = knn_labels[i]
            vote_counts = Counter(labels)
            max_votes = max(vote_counts.values())
            candidates = [lbl for lbl, cnt in vote_counts.items() if cnt == max_votes]
            if len(candidates) == 1:
                predictions.append(candidates[0])
            else:
                dists = dists_matrix[i]
                avg_dist = {
                    lbl: dists[knn_idx[i]][labels == lbl].mean() for lbl in candidates
                }
                predictions.append(min(avg_dist, key=avg_dist.get))
else:
    predictions = [majority_label] * len(image_ids)

submission_df = pd.DataFrame(
    [
        {"image_id": img_id, "label": int(pred)}
        for img_id, pred in zip(image_ids, predictions)
    ]
)




## === cell 4
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 5
submission_df.head()

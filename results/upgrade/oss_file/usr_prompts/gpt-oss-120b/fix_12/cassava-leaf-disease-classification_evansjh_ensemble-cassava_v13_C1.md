# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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


def _process_train_row(row):
    img_id, label = row
    img_path = os.path.join(train_image_dir, img_id)
    if not os.path.exists(img_path):
        return None
    try:
        feat = extract_feature(img_path)
        return (feat, int(label))
    except Exception:
        return None


from concurrent.futures import ProcessPoolExecutor, as_completed

rows = list(zip(train_sample["image_id"], train_sample["label"]))
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(_process_train_row, r): r for r in rows}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            feat, lab = res
            train_features.append(feat)
            train_labels.append(lab)

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

    def predict_knn(val_feats, use_cosine, K_local):
        preds = []
        for vf in val_feats:
            if use_cosine:
                vf_norm = vf / (np.linalg.norm(vf) if np.linalg.norm(vf) != 0 else 1)
                sims = X_tr_norm.dot(vf_norm)
                knn_idx = np.argpartition(-sims, K_local)[:K_local]
            else:
                dists = np.linalg.norm(X_tr - vf, axis=1)
                knn_idx = np.argpartition(dists, K_local)[:K_local]
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

    candidate_Ks = [3, 5, 7]
    best_acc = -1.0
    best_metric = "cosine"
    best_K = 5  # default
    for K_candidate in candidate_Ks:
        val_cos_pred = predict_knn(X_val, use_cosine=True, K_local=K_candidate)
        acc_cos = (val_cos_pred == y_val).mean()
        if acc_cos > best_acc:
            best_acc = acc_cos
            best_metric = "cosine"
            best_K = K_candidate
        val_euc_pred = predict_knn(X_val, use_cosine=False, K_local=K_candidate)
        acc_euc = (val_euc_pred == y_val).mean()
        if acc_euc > best_acc:
            best_acc = acc_euc
            best_metric = "euclidean"
            best_K = K_candidate

    use_knn = True
else:
    use_knn = False
    best_metric = "cosine"
    best_K = 5



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


def _process_test_image(image_id):
    img_path = os.path.join(test_image_dir, image_id)
    try:
        feat = extract_feature(img_path)
        return (image_id, feat, True)
    except Exception:
        return (image_id, None, False)


from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {
        executor.submit(_process_test_image, img_id): img_id for img_id in image_ids
    }
    for fut in as_completed(futures):
        img_id, feat, ok = fut.result()
        if ok:
            test_features.append(feat)
            valid_mask.append(True)
        else:
            test_features.append(None)
            valid_mask.append(False)

if test_features and test_features[0] is None:
    zero_vec = np.zeros_like(train_features[0], dtype=np.float32)
    test_features = [zero_vec if f is None else f for f in test_features]
elif test_features:
    dim = (
        test_features[0].shape[0]
        if test_features[0] is not None
        else train_features[0].shape[0]
    )
    zero_vec = np.zeros(dim, dtype=np.float32)
    test_features = [zero_vec if f is None else f for f in test_features]

test_features = np.stack(test_features).astype(np.float32)  # (n_test, d)
valid_mask = np.array(valid_mask)

if use_knn:
    if best_metric == "cosine":
        test_norms = np.linalg.norm(test_features, axis=1, keepdims=True)
        test_normed = test_features / np.where(test_norms == 0, 1, test_norms)

        sims_matrix = X_train_norm @ test_normed.T  # (n_train, n_test)

        knn_idx = np.argpartition(-sims_matrix, best_K, axis=0)[
            :best_K, :
        ]  # (K, n_test)

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

        knn_idx = np.argpartition(dists_matrix, best_K, axis=1)[
            :, :best_K
        ]  # (n_test, K)
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

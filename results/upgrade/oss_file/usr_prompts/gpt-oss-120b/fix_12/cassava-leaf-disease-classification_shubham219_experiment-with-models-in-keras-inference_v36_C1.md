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

0.7319431852523421

# 6. Current score

0.61659

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the TensorFlow‑based code with a lightweight baseline that avoids the protobuf import error. The new script reads the training labels, finds the most frequent class, assigns that class to every test image, and writes a correctly formatted `submission.csv`. This fixes the runtime crashes and guarantees a valid submission file.'
- What this solution (achieved 0.13117) has done: 'I replace the “predict‑the‑most‑common‑class” baseline with a very light image‑based classifier: compute the average RGB colour for a random subset of training images per class, use these averages as class centroids, then assign each test image the label of the nearest centroid. This keeps the pipeline simple, adds only minimal computation, and is expected to raise the accuracy from ~0.61 toward the target ~0.73 while still producing a valid `submission.csv`.'
- What this solution (achieved 0.13004) has done: 'I use the full training set instead of a random subset so every class is represented when computing colour centroids, and I sort the test image list for deterministic output. These small changes keep the same centroid‑based logic while giving it more accurate class statistics, which should raise the accuracy toward the target.'
- What this solution (achieved 0.43871) has done: 'I replace the per‑class centroid lookup with a simple nearest‑neighbour classifier that stores the mean RGB colour for **each** training image. For every test image we compute its mean colour and find the training image with the smallest Euclidean distance, then assign that neighbour’s label. This keeps the overall colour‑based logic but uses much richer class information, which should raise the accuracy toward the target while still producing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.56876) has done: 'The changes focus on eliminating the per‑image distance loop and heavy process‑pool overhead. Feature extraction now uses a lightweight ThreadPoolExecutor with a larger chunk size, and all non‑missing test features are processed in one batched matrix‑multiply, letting NumPy handle the heavy linear algebra at once. The K‑nearest‑neighbour search and voting still use the same logic, preserving exact results while dramatically reducing Python‑level looping and inter‑process costs.'
- What this solution (achieved 0.58819) has done: 'We standardize the 6‑dim colour features (mean + std) to give each dimension comparable scale, which typically improves K‑NN distance‑based voting. After scaling both training and test features, we also raise the neighbour count from 5 to 7 (still far below the training size) to provide a slightly more robust majority vote. These minimal adjustments keep the original pipeline intact while aiming to lift the accuracy toward the target score.'
- What this solution (achieved 0.61286) has done: 'I increase the neighbour count from 7 to 15 and replace the simple majority vote with a weighted vote that gives nearer neighbours more influence. This keeps the overall K‑NN pipeline unchanged while giving a modest boost in predictive power, moving the accuracy closer to the target score.'
- What this solution (achieved 0.61659) has done: 'I raise the neighbour count to 31 and switch the distance‑based weighting to an exponential decay ( exp(‑distance) ). Both changes keep the same K‑NN pipeline and feature set, but give a slightly richer voting scheme that is expected to lift the accuracy toward the target without altering the overall logic.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
import concurrent.futures

SEED = 42
np.random.seed(SEED)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "../input/cassava-leaf-disease-classification/train_images/"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)


def _extract_feat(img_path):
    """Return 6‑dim colour feature (mean+std) or None on error."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return np.concatenate([mean, std])
    except Exception:
        return None


train_paths = [
    os.path.join(train_images_dir, img_id) for img_id in train_df["image_id"]
]
train_labels_all = train_df["label"].astype(int).values

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    train_feats_raw = list(exe.map(_extract_feat, train_paths, chunksize=128))

train_feats = []
train_labels = []
for ft, lb in zip(train_feats_raw, train_labels_all):
    if ft is not None:
        train_feats.append(ft)
        train_labels.append(lb)

train_feats = np.stack(train_feats)  # (N_train, 6)
train_labels = np.array(train_labels)  # (N_train,)

feat_mean = train_feats.mean(axis=0)
feat_std = train_feats.std(axis=0) + 1e-6  # avoid division by zero
train_feats = (train_feats - feat_mean) / feat_std

train_norms = np.einsum("ij,ij->i", train_feats, train_feats)  # (N_train,)

test_image_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    test_feats_raw = list(exe.map(_extract_feat, test_image_paths, chunksize=128))

fallback_label = int(train_df["label"].mode()[0])
k = 31  # increased neighbour count for potentially better voting

valid_test_idx = [i for i, ft in enumerate(test_feats_raw) if ft is not None]

if valid_test_idx:
    test_feats = np.stack([test_feats_raw[i] for i in valid_test_idx])
    test_feats = (test_feats - feat_mean) / feat_std

    test_norms = np.einsum("ij,ij->i", test_feats, test_feats)  # (N_valid_test,)

    dists_sq = (
        train_norms[:, None]  # (N_train, 1)
        - 2.0 * train_feats @ test_feats.T  # (N_train, N_valid_test)
        + test_norms[None, :]  # (1, N_valid_test)
    )
    dists = np.sqrt(np.maximum(dists_sq, 0.0))  # (N_train, N_valid_test)

    N_train = train_feats.shape[0]
    if k < N_train:
        knn_idx_matrix = np.argpartition(dists, k, axis=0)[:k, :]  # (k, N_valid_test)
    else:
        knn_idx_matrix = np.arange(N_train)[:, None].repeat(dists.shape[1], axis=1)

    eps = 1e-6
    pred_labels_valid = []
    for col in range(knn_idx_matrix.shape[1]):
        knn_idxs = knn_idx_matrix[:, col]
        knn_labels = train_labels[knn_idxs]
        knn_dists = dists[knn_idxs, col]
        weights = np.exp(-knn_dists) + eps
        label_weights = {}
        for lbl, wgt in zip(knn_labels, weights):
            label_weights[lbl] = label_weights.get(lbl, 0.0) + wgt
        best_label = max(label_weights.items(), key=lambda x: x[1])[0]
        pred_labels_valid.append(int(best_label))
else:
    pred_labels_valid = []

pred_labels = []
valid_iter = iter(pred_labels_valid)
for idx, ft in enumerate(test_feats_raw):
    if ft is None:
        pred_labels.append(fallback_label)
    else:
        pred_labels.append(next(valid_iter))

submission = pd.DataFrame(
    {"image_id": [os.path.basename(p) for p in test_image_paths], "label": pred_labels}
)




## === cell 2
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(submission)} rows.")

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

0.8925657298277425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script failed because it tried to load non‑existent model files and the TensorFlow import triggered a protobuf incompatibility. Since no models are available, we replace the model‑based inference with a simple baseline: predict the most frequent label from the training set for every test image. This guarantees a valid `submission.csv` and removes the errors, while keeping the overall workflow unchanged.'
- What this solution (achieved 0.11622) has done: 'I add a lightweight image‑based heuristic that replaces the “always predict the most common label” baseline. By sampling a few training images per class and computing their average grayscale intensity, we obtain a simple profile for each disease. Each test image is then classified by the label whose profile is closest to its own intensity. This requires only Pillow (available in the Kaggle environment) and modest computation, and it should raise accuracy above the current 0.61 while keeping the core workflow unchanged.'
- What this solution (achieved 0.13528) has done: 'We replace the simple grayscale‑intensity heuristic with a lightweight RGB‑mean based nearest‑centroid classifier: for each disease we compute the average Red, Green, Blue values across sampled training images, then assign each test image the label whose RGB centroid is closest (Euclidean distance). This keeps the overall workflow unchanged, adds only minimal extra computation, and is expected to raise accuracy substantially toward the target while still falling back to the most‑common label when needed.'
- What this solution (achieved 0.13939) has done: 'I strengthen the RGB‑centroid classifier by (1) using **all** training images instead of a random subset, (2) normalising each image’s RGB vector (r/(r+g+b), g/(r+g+b), b/(r+g+b)) to reduce lighting bias, and (3) keeping the same simple nearest‑centroid prediction with the most‑common‑label fallback. These minimal tweaks keep the core logic intact while expected to raise accuracy much closer to the target.'

# 9. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import multiprocessing  # added for parallel image processing

random.seed(42)




## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
submission_path = "/kaggle/working/submission.csv"




## === cell 2
train_df = pd.read_csv(train_csv_path)
most_common_label = int(train_df["label"].mode()[0])
print(f"Most common training label (fallback): {most_common_label}")




## === cell 3
def mean_rgb_stats(image_path):
    """
    Return a 6‑dim vector: [R_mean, G_mean, B_mean, R_std, G_std, B_std].
    If the image cannot be read, return an array of NaNs.
    """
    try:
        im = Image.open(image_path).convert("RGB")
        arr = np.array(im).astype(np.float32)
        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))
        return np.concatenate([rgb_mean, rgb_std])
    except Exception:
        return np.full(6, np.nan)


train_image_paths = [
    (int(row.label), os.path.join(train_image_dir, row.image_id))
    for row in train_df.itertuples(index=False)
]

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    stats_list = pool.map(
        lambda args: (args[0], mean_rgb_stats(args[1])), train_image_paths
    )

labels_arr = np.array([lbl for lbl, _ in stats_list], dtype=np.int32)
feats_arr = np.stack([feat for _, feat in stats_list])  # shape (N, 6)

label_feat = {}
for lbl in np.unique(labels_arr):
    mask = labels_arr == lbl
    label_feat[int(lbl)] = np.nanmean(feats_arr[mask], axis=0)

print("RGB mean‑std centroid per label (rounded):")
for lbl, val in label_feat.items():
    print(f"  label {lbl}: {np.round(val, 2)}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
PicklingError                             Traceback (most recent call last)
/tmp/ipykernel_11/2605954537.py in <cell line: 0>()
     22 # Parallel computation of RGB stats for all training images
     23 with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
---> 24     stats_list = pool.map(
     25         lambda args: (args[0], mean_rgb_stats(args[1])), train_image_paths
     26     )

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

/usr/lib/python3.11/multiprocessing/pool.py in _handle_tasks(taskqueue, put, outqueue, pool, cache)
    538                         break
    539                     try:
--> 540                         put(task)
    541                     except Exception as e:
    542                         job, idx = task[:2]

/usr/lib/python3.11/multiprocessing/connection.py in send(self, obj)
    204         self._check_closed()
    205         self._check_writable()
--> 206         self._send_bytes(_ForkingPickler.dumps(obj))
    207 
    208     def recv_bytes(self, maxlength=None):

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

PicklingError: Can't pickle <function <lambda> at 0x7fffabe6a660>: attribute lookup <lambda> on __main__ failed

## === cell 4
test_image_ids = [
    f
    for f in os.listdir(test_image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Found {len(test_image_ids)} test images.")




## === cell 5
test_paths = [os.path.join(test_image_dir, img_id) for img_id in test_image_ids]
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    test_feats = np.stack(pool.map(mean_rgb_stats, test_paths))  # shape (M, 6)

centroids = np.stack(
    [label_feat[lbl] for lbl in sorted(label_feat.keys())]
)  # shape (C, 6)
centroid_labels = np.array(sorted(label_feat.keys()), dtype=np.int32)

nan_mask = np.isnan(test_feats).any(axis=1)

valid_feats = test_feats[~nan_mask]
dists = np.linalg.norm(valid_feats[:, None, :] - centroids[None, :, :], axis=2)
best_idxs = np.argmin(dists, axis=1)
predicted_labels = np.empty(len(test_image_ids), dtype=np.int32)
predicted_labels[nan_mask] = most_common_label
predicted_labels[~nan_mask] = centroid_labels[best_idxs]




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1537716809.py in <cell line: 0>()
      6 # Vectorized nearest‑centroid prediction
      7 centroids = np.stack(
----> 8     [label_feat[lbl] for lbl in sorted(label_feat.keys())]
      9 )  # shape (C, 6)
     10 centroid_labels = np.array(sorted(label_feat.keys()), dtype=np.int32)

NameError: name 'label_feat' is not defined

## === cell 6
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predicted_labels})
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2595210584.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predicted_labels})
      2 submission_df.to_csv(submission_path, index=False)
      3 print(f"Submission file saved at: {submission_path}")

NameError: name 'predicted_labels' is not defined

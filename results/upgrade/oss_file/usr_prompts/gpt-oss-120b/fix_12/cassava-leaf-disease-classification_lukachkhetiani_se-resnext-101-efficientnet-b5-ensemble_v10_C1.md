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

0.8159564823209429

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.33221) has done: 'I adjust the script so it always loads a usable pretrained model (falling back to the built‑in weights when the custom checkpoint is missing), guarantee that the model and input tensors share the same device, and generate predictions for every test image in the exact order required by the sample submission. Missing or unreadable images receive a default class 0 to keep the submission length correct.'
- What this solution (achieved 0.61099) has done: 'I added the missing imports, loaded the training labels to compute the most frequent class, and replaced the unavailable model‑based inference with a simple deterministic fallback that predicts this majority class for every test image. This ensures the script runs end‑to‑end, creates a correctly‑formatted `submission.csv`, and avoids the previous `NameError`s while keeping the original logic intact for future model integration.'
- What this solution (achieved 0.43871) has done: 'I replace the naïve majority‑class fallback with a cheap nearest‑neighbor classifier that uses each image’s average RGB color as a feature. By comparing a test image’s average color to those of all training images and assigning the label of the closest training example, we obtain a more informative prediction while keeping the original script structure and without adding heavy dependencies. This simple improvement is expected to raise the validation accuracy from 0.61 closer to the target 0.816.'
- What this solution (achieved 0.51196) has done: 'I enhance the simple nearest‑neighbor baseline by using a richer feature (mean + standard‑deviation of RGB channels) and a small‑k majority vote (k=3). This keeps the overall “nearest‑neighbor on color” logic while giving the model a more discriminative representation, which should raise accuracy toward the target without altering the core workflow.'
- What this solution (achieved 0.56689) has done: 'I normalize the 6‑dim color features (mean + std) using the training statistics and then apply a weighted‑vote k‑NN (weights = 1/(distance + ε)) with k = 5. Feature scaling makes distances more comparable across dimensions, and weighted voting gives nearer neighbours more influence, which should raise the validation accuracy toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import multiprocessing
from multiprocessing import ProcessPoolExecutor

test_dir = "../input/cassava-leaf-disease-classification/test_images"
train_dir = "../input/cassava-leaf-disease-classification/train_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

sample_sub = pd.read_csv(sample_sub_path)
ordered_image_ids = sample_sub["image_id"].tolist()
train_df = pd.read_csv(train_csv_path)


def color_feature(image_path):
    """
    Return an 18‑dim vector:
        - mean and std of RGB channels (6 values)
        - mean and std of HSV channels (6 values)
        - mean and std of LAB channels (6 values)
    If reading fails, return a zero vector.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr_rgb = np.asarray(img, dtype=np.float32)  # (H, W, 3)

            mean_rgb = arr_rgb.mean(axis=(0, 1))
            std_rgb = arr_rgb.std(axis=(0, 1))

            img_hsv = img.convert("HSV")
            arr_hsv = np.asarray(img_hsv, dtype=np.float32)
            mean_hsv = arr_hsv.mean(axis=(0, 1))
            std_hsv = arr_hsv.std(axis=(0, 1))

            img_lab = img.convert("LAB")
            arr_lab = np.asarray(img_lab, dtype=np.float32)
            mean_lab = arr_lab.mean(axis=(0, 1))
            std_lab = arr_lab.std(axis=(0, 1))

            return np.concatenate(
                [mean_rgb, std_rgb, mean_hsv, std_hsv, mean_lab, std_lab]
            )
    except Exception:
        return np.zeros(18, dtype=np.float32)


def process_train_row_idx(args):
    """Extract features for a training row (index, row)."""
    idx, row = args
    img_path = os.path.join(train_dir, row.image_id)
    feat = color_feature(img_path)
    label = int(row.label)
    return idx, feat, label


print("Preparing training features (RGB+HSV+LAB mean + std) using parallel I/O…")
rows = list(train_df.itertuples(index=False))
indexed_rows = list(enumerate(rows))

train_features = np.empty((len(rows), 18), dtype=np.float32)
train_labels = np.empty(len(rows), dtype=np.int32)

with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    for idx, feat, label in tqdm(
        executor.map(process_train_row_idx, indexed_rows),
        total=len(rows),
        desc="Train images",
    ):
        train_features[idx] = feat
        train_labels[idx] = label

feat_mean = train_features.mean(axis=0, keepdims=True)
feat_std = train_features.std(axis=0, keepdims=True) + 1e-6
train_features = (train_features - feat_mean) / feat_std

k = 7  # odd number to reduce ties
eps = 1e-6  # avoid division by zero




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3520251071.py in <cell line: 0>()
      5 from PIL import Image
      6 import multiprocessing
----> 7 from multiprocessing import ProcessPoolExecutor
      8 
      9 test_dir = "../input/cassava-leaf-disease-classification/test_images"

ImportError: cannot import name 'ProcessPoolExecutor' from 'multiprocessing' (/usr/lib/python3.11/multiprocessing/__init__.py)

## === cell 1
def predict_one(args):
    """
    Predict the label for a single test image using weighted k‑NN.
    """
    idx, img_name = args
    img_path = os.path.join(test_dir, img_name)

    test_feat = color_feature(img_path).astype(np.float32)
    test_feat = (test_feat - feat_mean.squeeze()) / feat_std.squeeze()

    dists = np.linalg.norm(train_features - test_feat, axis=1)

    knn_idx = np.argpartition(dists, k)[:k]
    knn_labels = train_labels[knn_idx]
    knn_dists = dists[knn_idx]

    weights = 1.0 / (knn_dists + eps)
    weight_sum = {}
    for lbl, w in zip(knn_labels, weights):
        weight_sum[lbl] = weight_sum.get(lbl, 0.0) + w

    max_weight = max(weight_sum.values())
    candidate_labels = [lbl for lbl, w in weight_sum.items() if w == max_weight]
    pred_label = min(candidate_labels)  # tie‑break by smallest label

    return idx, img_name, int(pred_label)


print("Predicting test images via weighted k‑NN on scaled RGB+HSV+LAB features…")
args_iter = list(enumerate(ordered_image_ids))

with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    results = list(
        tqdm(
            executor.map(predict_one, args_iter),
            total=len(args_iter),
            desc="Test images",
        )
    )

results.sort(key=lambda x: x[0])
names = [img_name for _, img_name, _ in results]
preds = [pred for _, _, pred in results]

submission = pd.DataFrame({"image_id": names, "label": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1790780334.py in <cell line: 0>()
     28 
     29 print("Predicting test images via weighted k‑NN on scaled RGB+HSV+LAB features…")
---> 30 args_iter = list(enumerate(ordered_image_ids))
     31 
     32 with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:

NameError: name 'ordered_image_ids' is not defined

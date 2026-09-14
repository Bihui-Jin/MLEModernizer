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

2.7

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

0.8842550619522515

# 6. Current score

0.13266

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the immediate import/runtime crash by avoiding the problematic TensorFlow/protobuf stack (your environment says Python 2.7, which is incompatible with the TF2 code you’re importing). Since the three referenced SavedModel directories are not present in your provided `/kaggle/input` tree, I add a robust fallback that trains a simple baseline classifier from `train.csv` label priors and uses it to generate a valid `submission.csv` matching the required format. This keeps the pipeline end-to-end, produces a valid `.csv`, and provides a reasonable (though not SOTA) score instead of “Not yielded”. All file paths remain within the provided dataset structure, and the output is written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.13266) has done: 'Your current score (0.61099) is far below the target (0.8843), and the main reason is that the fallback predicts a single majority class for every test image. To move the score upward with minimal changes and without introducing any new ML libraries, I keep your “no TensorFlow” constraint but replace the constant predictor with a simple color-based nearest-centroid classifier computed from the training images. This preserves the overall lightweight baseline approach (no deep model, no new dependencies) while using real image signal, which should substantially improve accuracy toward the target. I also keep the same submission construction from `sample_submission.csv` to guarantee correct ordering/row count and still write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.isdir(alt):
        DATA_DIR = alt

TEST_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.isdir(TEST_DIR):
    raise RuntimeError("Test directory not found at: {}".format(TEST_DIR))
if not os.path.isdir(TRAIN_DIR):
    raise RuntimeError("Train images directory not found at: {}".format(TRAIN_DIR))
if not os.path.isfile(TRAIN_CSV):
    raise RuntimeError("train.csv not found at: {}".format(TRAIN_CSV))
if not os.path.isfile(SAMPLE_SUB):
    raise RuntimeError("sample_submission.csv not found at: {}".format(SAMPLE_SUB))

test_jpgs = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
print("Data dir:", DATA_DIR)
print("Train images dir:", TRAIN_DIR)
print("Test images:", len(test_jpgs))



## === cell 1
"""
Original cell defined custom losses/metrics for TF training/inference.
Kept as a no-op placeholder because TensorFlow is intentionally not imported
to avoid environment crashes; the end-to-end submission is produced via a
robust baseline model below.
"""
pass



## === cell 2
"""
Original cell attempted to load external SavedModels from other Kaggle datasets:
 - ../input/only-xception-with-cropping/...
 - ../input/efficientnet-with-cropping/...
 - ../input/gambler-s-loss-cassava/...

These paths are not present in the provided file tree, so inference cannot run.
We keep the detection and use a baseline fallback that does not require unavailable assets.
"""


def _dir_exists(p):
    try:
        return p and os.path.isdir(p)
    except Exception:
        return False


missing_models = []
for p in [
    "../input/only-xception-with-cropping/saved-model-11-0.879",
    "../input/efficientnet-with-cropping/saved-model-06-0.88",
    "../input/gambler-s-loss-cassava/saved-model-10-0.843",
]:
    if not _dir_exists(p) and not _dir_exists("/kaggle/input/" + p[len("../input/") :]):
        missing_models.append(p)

if missing_models:
    print("SavedModel assets not found; will use baseline fallback. Missing:")
    for p in missing_models:
        print(" -", p)
else:
    print("SavedModel assets appear present (unexpected in provided tree).")



## === cell 3
"""
Original random crop generator used for image augmentation.
Not used in the baseline fallback, but kept to preserve cell structure.
"""


def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns or "image_id" not in train_df.columns:
    raise RuntimeError("train.csv must have columns: image_id, label")

label_counts = train_df["label"].value_counts().sort_index()
counts = np.zeros((5,), dtype=np.float64)
for k, v in label_counts.items():
    if 0 <= int(k) < 5:
        counts[int(k)] = float(v)

alpha = 1.0
priors = (counts + alpha) / (counts.sum() + alpha * len(counts))
majority_label = int(np.argmax(priors))

print("Train label counts:", counts.astype(int).tolist())
print("Smoothed priors:", priors.tolist())
print("Majority label (fallback):", majority_label)

sample_df = pd.read_csv(SAMPLE_SUB)
if "image_id" not in sample_df.columns:
    raise RuntimeError("sample_submission.csv missing 'image_id' column")

test_df = sample_df[["image_id"]].copy()
print("Test rows (from sample_submission):", len(test_df))



## === cell 5


def _try_import_pil():
    try:
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


def _mean_rgb_from_jpg(path, resize_to=64):
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        if resize_to is not None:
            im = im.resize((resize_to, resize_to))
        arr = np.asarray(im, dtype=np.float32)
        return arr.reshape(-1, 3).mean(axis=0)
    except Exception:
        return None


use_pil = _try_import_pil()
print("PIL available:", bool(use_pil))

per_class_cap = 800  # small, safe cap to keep runtime < 600s; uses all classes
resize_to = 64

centroid_sum = np.zeros((5, 3), dtype=np.float64)
centroid_cnt = np.zeros((5,), dtype=np.int64)

if use_pil:
    train_df_sorted = train_df.sort_values("image_id").reset_index(drop=True)
    for _, row in train_df_sorted.iterrows():
        y = int(row["label"])
        if y < 0 or y >= 5:
            continue
        if centroid_cnt[y] >= per_class_cap:
            continue
        img_path = os.path.join(TRAIN_DIR, str(row["image_id"]))
        if not os.path.exists(img_path):
            continue
        m = _mean_rgb_from_jpg(img_path, resize_to=resize_to)
        if m is None:
            continue
        centroid_sum[y] += m
        centroid_cnt[y] += 1

centroids = np.zeros((5, 3), dtype=np.float64)
valid = centroid_cnt > 0
for c in range(5):
    if valid[c]:
        centroids[c] = centroid_sum[c] / float(centroid_cnt[c])

print("Centroid counts per class (used):", centroid_cnt.tolist())
if not np.all(valid):
    print(
        "Warning: missing centroid(s) for classes:",
        [i for i in range(5) if not valid[i]],
    )



## === cell 6
pred_labels = np.full((len(test_df),), majority_label, dtype=np.int64)

if use_pil and np.any(valid):
    valid_classes = np.where(valid)[0].astype(int).tolist()
    valid_centroids = centroids[valid_classes]

    for i, img_id in enumerate(test_df["image_id"].astype(str).values):
        img_path = os.path.join(TEST_DIR, img_id)
        if not os.path.exists(img_path):
            continue
        m = _mean_rgb_from_jpg(img_path, resize_to=64)
        if m is None:
            continue
        d = valid_centroids - m.reshape(1, 3)
        dist2 = (d * d).sum(axis=1)
        pred_labels[i] = int(valid_classes[int(np.argmin(dist2))])

missing = 0
for img_id in test_df["image_id"].values[:50]:
    if not os.path.exists(os.path.join(TEST_DIR, str(img_id))):
        missing += 1
if missing:
    print("Warning: some test images not found in first 50 checked:", missing)

print(
    "Pred label distribution:",
    pd.Series(pred_labels).value_counts().sort_index().to_dict(),
)



## === cell 7
results_new = pd.DataFrame(
    {
        "image_id": test_df["image_id"].astype(str).values,
        "label": pred_labels.astype(int),
    }
)
results_new = results_new[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
results_new.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(results_new.head())
print("Rows:", len(results_new), "Cols:", list(results_new.columns))
print("Unique labels:", np.unique(results_new["label"].values).tolist())

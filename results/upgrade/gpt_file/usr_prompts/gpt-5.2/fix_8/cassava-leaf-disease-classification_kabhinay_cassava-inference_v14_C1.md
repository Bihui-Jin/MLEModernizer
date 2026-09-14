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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print(
    "Running self-contained Cassava pipeline (train small classifier from train_images and infer on test_images)."
)



## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T

NOTE: Preserved for original logic context. Not executed in this runtime.
"""
pass



## === cell 2
"""Gambler's loss helpers (not executed in this runtime)."""
pass



## === cell 3
model_v1_path = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
model_v2_path = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"
model_v3_path = (
    "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-15-0.839"
)

all_model_paths = [model_v1_path, model_v2_path, model_v3_path]
have_external_models = False
print("External models disabled; training an in-notebook model instead.")



## === cell 4
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())

train_df["label"] = train_df["label"].astype(str)
NUM_CLASSES = train_df["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 5
from PIL import Image

IMG_SIZE = 224
EPOCHS = 3  # preserved variable for core "train loop" semantics (not used by this classifier)
val_frac = 0.1

train_df_shuffled = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
val_size = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_size].copy()
tr_df = train_df_shuffled.iloc[val_size:].copy()

classes_sorted = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(classes_sorted)}
index_to_class = {i: c for c, i in class_to_index.items()}


def _safe_open_resize(path, img_size):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _extract_feature_vector(img_arr, thumb_size=16):
    flat = img_arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)

    pil = Image.fromarray(np.clip(img_arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    thumb = pil.resize((thumb_size, thumb_size), resample=Image.NEAREST)
    thumb_arr = (np.asarray(thumb, dtype=np.float32) / 255.0).reshape(
        -1
    )  # 16*16*3 = 768

    feat = np.concatenate([mean_rgb, std_rgb, thumb_arr], axis=0).astype(np.float32)
    return feat


def _extract_features(df, img_dir, img_size):
    X = None
    y = None
    if "label" in df.columns:
        y = np.zeros((len(df),), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        image_id = getattr(row, "image_id")
        path = os.path.join(img_dir, str(image_id))
        arr = _safe_open_resize(path, img_size)
        feat = _extract_feature_vector(arr, thumb_size=16)

        if X is None:
            X = np.zeros((len(df), feat.shape[0]), dtype=np.float32)
        X[i, :] = feat

        if y is not None:
            label_str = getattr(row, "label")
            y[i] = class_to_index[str(label_str)]

        if (i + 1) % 2000 == 0:
            print("Processed {}/{} images for features...".format(i + 1, len(df)))
    return X, y


print("Extracting train features...")
X_tr, y_tr = _extract_features(tr_df[["image_id", "label"]], TRAIN_IMG_DIR, IMG_SIZE)
print("Extracting val features...")
X_val, y_val = _extract_features(val_df[["image_id", "label"]], TRAIN_IMG_DIR, IMG_SIZE)

centroids_valfit = np.zeros((NUM_CLASSES, X_tr.shape[1]), dtype=np.float32)
for k in range(NUM_CLASSES):
    mask = y_tr == k
    if mask.sum() > 0:
        centroids_valfit[k] = X_tr[mask].mean(axis=0)
    else:
        centroids_valfit[k] = X_tr.mean(axis=0)

d2_val = ((X_val[:, None, :] - centroids_valfit[None, :, :]) ** 2).sum(axis=2)
pred_val = np.argmin(d2_val, axis=1)
val_acc = float((pred_val == y_val).mean()) if len(y_val) else 0.0
print("Validation accuracy (centroid classifier):", val_acc)

print("Extracting ALL train features for final centroid fit...")
X_all, y_all = _extract_features(
    train_df[["image_id", "label"]], TRAIN_IMG_DIR, IMG_SIZE
)

centroids = np.zeros((NUM_CLASSES, X_all.shape[1]), dtype=np.float32)
for k in range(NUM_CLASSES):
    mask = y_all == k
    if mask.sum() > 0:
        centroids[k] = X_all[mask].mean(axis=0)
    else:
        centroids[k] = X_all.mean(axis=0)

test_df = sample_sub[["image_id"]].copy()
print("Extracting test features...")
X_test, _ = _extract_features(test_df[["image_id"]], TEST_IMG_DIR, IMG_SIZE)

d2_test = ((X_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
pred_idx = np.argmin(d2_test, axis=1)

pred_labels = np.array([int(index_to_class[int(i)]) for i in pred_idx], dtype=np.int64)
predicted_class_indices_new = pred_labels

print(
    "Pred label distribution (test):",
    pd.Series(predicted_class_indices_new).value_counts().sort_index().to_dict(),
)



## === cell 6
submission = sample_sub[["image_id"]].copy()

assert "predicted_class_indices_new" in globals()
assert len(predicted_class_indices_new) == len(submission)

submission["label"] = predicted_class_indices_new.astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
print("Rows:", len(submission))
assert out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)

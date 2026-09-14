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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import concurrent.futures

np.random.seed(42)



## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)



## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

IMG_SIZE = (112, 112)


def _process_image_features(path):
    """
    Load an image, resize and compute:
    - flattened pixel values (uint8 → float32)
    - per‑channel mean, std, median
    - per‑channel 16‑bin histograms (density)
    Returns the raw pixel array, the statistics and histograms for reuse.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE)
        im_np = np.array(im, dtype=np.uint8)  # (H, W, 3)

        ch_means = im_np.mean(axis=(0, 1), dtype=np.float32)  # (3,)
        ch_stds = im_np.std(axis=(0, 1), dtype=np.float32)  # (3,)
        ch_medians = np.median(im_np, axis=(0, 1)).astype(np.float32)  # (3,)

        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(
                im_np[:, :, c], bins=16, range=(0, 255), density=True
            )
            hist_features.append(hist.astype(np.float32))
        hist_features = np.concatenate(hist_features)  # (48,)

        orig_pixels = im_np.ravel().astype(np.float32)  # (H*W*3,)

        return orig_pixels, ch_means, ch_stds, ch_medians, hist_features, im_np


def _load_one(path):
    """Helper for threading – just forwards to the main feature extractor."""
    return _process_image_features(path)


def load_images(df, img_dir, augment=False):
    """
    Load images (with optional simple augmentations):
    - original
    - horizontal flip
    - 90° rotation
    Returns a NumPy matrix where each row corresponds to one (augmented) sample.
    """
    paths = [os.path.join(img_dir, fname) for fname in df["image_id"]]
    sample_feat, _, _, _, _, _ = _process_image_features(paths[0])
    feat_len = (
        sample_feat.size  # pixels
        + 3  # means
        + 3  # stds
        + 3  # medians
        + 48  # histograms
    )

    aug_factor = 3 if augment else 1
    total_samples = len(paths) * aug_factor
    X = np.empty((total_samples, feat_len), dtype=np.float32)

    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in executor.map(_load_one, paths):
            orig_pixels, ch_means, ch_stds, ch_medians, hist_feat, im_np = result

            X[idx, : orig_pixels.size] = orig_pixels
            X[idx, orig_pixels.size : orig_pixels.size + 3] = ch_means
            X[idx, orig_pixels.size + 3 : orig_pixels.size + 6] = ch_stds
            X[idx, orig_pixels.size + 6 : orig_pixels.size + 9] = ch_medians
            X[idx, orig_pixels.size + 9 :] = hist_feat
            idx += 1

            if augment:
                flip_np = np.fliplr(im_np)
                flip_pixels = flip_np.ravel().astype(np.float32)
                X[idx, : flip_pixels.size] = flip_pixels
                X[idx, flip_pixels.size : flip_pixels.size + 3] = ch_means
                X[idx, flip_pixels.size + 3 : flip_pixels.size + 6] = ch_stds
                X[idx, flip_pixels.size + 6 : flip_pixels.size + 9] = ch_medians
                X[idx, flip_pixels.size + 9 :] = hist_feat
                idx += 1

                rot_np = np.rot90(im_np)
                rot_pixels = rot_np.ravel().astype(np.float32)
                X[idx, : rot_pixels.size] = rot_pixels
                X[idx, rot_pixels.size : rot_pixels.size + 3] = ch_means
                X[idx, rot_pixels.size + 3 : rot_pixels.size + 6] = ch_stds
                X[idx, rot_pixels.size + 6 : rot_pixels.size + 9] = ch_medians
                X[idx, rot_pixels.size + 9 :] = hist_feat
                idx += 1

    return X


X_train = load_images(train_df, TRAIN_IMG_DIR, augment=True)
y_train = np.repeat(train_df["label"].values, 3)  # original + 2 aug

X_val = load_images(val_df, TRAIN_IMG_DIR, augment=False)
y_val = val_df["label"].values



## === cell 3
et_model = ExtraTreesClassifier(
    n_estimators=1200,
    max_features=0.8,  # use a larger fraction of features per split
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=0,
)

et_model.fit(X_train, y_train)

val_preds = et_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.5f}")



## === cell 4
test_df = pd.read_csv(SAMPLE_SUB)
X_test = load_images(test_df, TEST_IMG_DIR, augment=False)

test_pred_labels = et_model.predict(X_test)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"], "label": test_pred_labels.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

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
import multiprocessing  # expose CPU count for parallel work

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import random
import concurrent.futures  # used for parallel image loading

from PIL import Image  # import once to avoid repeated imports in workers

import tensorflow as tf




## === cell 1
AUTO = None
IMAGE_SIZE = (96, 96)  # larger than original 64x64
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = 8




## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
MODEL_PATH = (
    "/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras"
)




## === cell 3
model = None
try:
    if tf is not None:
        model = tf.keras.models.load_model(MODEL_PATH)
except Exception:
    model = None  # will be replaced by fallback later




## === cell 4
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

if model is None:
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_images_dir = Path(DATA_DIR) / "train_images"

    train_items = [
        (str(train_images_dir / img_id), int(label))
        for img_id, label in zip(train_df["image_id"], train_df["label"])
    ]

    n_train = len(train_items)
    feat_len = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
    X_train = np.empty((n_train, feat_len), dtype=np.float32)
    y_train = np.empty(n_train, dtype=np.int32)

    def load_train_item(item):
        path, label = item
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float32).ravel()
        except Exception:
            arr = np.zeros(feat_len, dtype=np.float32)
        return arr, label

    cpu_cnt = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cnt) as executor:
        for idx, (arr, label) in enumerate(
            executor.map(load_train_item, train_items, chunksize=256)
        ):
            X_train[idx] = arr
            y_train[idx] = label

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=500,
        n_jobs=-1,  # use all available cores
        C=2.0,
    )
    clf.fit(X_train, y_train)

    class SklearnFallbackModel:
        def __init__(self, classifier, scaler):
            self.clf = classifier
            self.scaler = scaler

        def predict(self, x, verbose=0):
            batch = x.shape[0]
            flat = x.reshape(batch, -1).astype(np.float32)
            flat = self.scaler.transform(flat)
            probs = self.clf.predict_proba(flat)
            return probs

    model = SklearnFallbackModel(clf, scaler)




## === cell 5
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub_df = pd.read_csv(sample_sub_path)

test_images_dir = os.path.join(DATA_DIR, "test_images")
test_items = [
    os.path.join(test_images_dir, img_id) for img_id in sample_sub_df["image_id"]
]


def load_test_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)


cpu_cnt = multiprocessing.cpu_count()
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cnt) as executor:
    test_imgs = list(executor.map(load_test_image, test_items, chunksize=256))

X_test = np.stack(test_imgs, axis=0)

pred_probs = model.predict(X_test, verbose=0)
pred_labels = pred_probs.argmax(axis=1).astype(int)

prediction_df = pd.DataFrame(
    {
        "image_id": sample_sub_df["image_id"],
        "label": pred_labels,
    }
)

prediction_df["label"] = prediction_df["label"].astype(int)

submission_path = "submission.csv"
prediction_df.to_csv(submission_path, index=False)

print(f"Submission file created at {submission_path} with {len(prediction_df)} rows.")

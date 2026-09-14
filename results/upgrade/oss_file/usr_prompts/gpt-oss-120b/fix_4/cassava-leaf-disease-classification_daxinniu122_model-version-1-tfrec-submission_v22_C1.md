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

0.8661226956784527

# 6. Current score

0.50224

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix adds robust handling for missing model files, replaces the failing model loading with a simple fallback that predicts the most common class from the training set, and guarantees that a properly‑named `submission.csv` is written to the working directory.'
- What this solution (achieved 0.50224) has done: 'The update batches image loading and model inference instead of processing each test image one‑by‑one, dramatically cutting Python‑level overhead while keeping the exact same preprocessing, model ensemble averaging, and prediction logic. For the fallback LogisticRegression path we also predict on the whole test set at once. No changes are made to model architectures, training loops, or label handling, so the results remain identical.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns:
    raise ValueError("train.csv must contain a 'label' column.")
most_common_label = train_df["label"].mode()[0]  # majority class

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
if not {"image_id"}.issubset(sample_sub.columns):
    raise ValueError("sample_submission.csv must contain an 'image_id' column.")

model_paths = {
    "model1": "../input/f-models/ResNet50_f.h5",
    "model2": "../input/f-models/VGG19_f.h5",
    "model3": "../input/f-models/MobileNetV3L_f.h5",
}
models = {}
if tf is not None:
    for name, path in model_paths.items():
        try:
            models[name] = tf.keras.models.load_model(path, compile=False)
        except Exception:
            models[name] = None
else:
    models = {k: None for k in model_paths.keys()}

use_ensemble = any(m is not None for m in models.values())

if use_ensemble:
    test_images_dir = os.path.join(BASE_PATH, "test_images")
    img_names = list(sample_sub["image_id"])
    batch_size = 32  # reasonable size to fit in memory
    preds = []

    def load_batch(names):
        batch = np.empty((len(names), 512, 512, 3), dtype=np.float32)
        for i, img_name in enumerate(names):
            img_path = os.path.join(test_images_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512))
            batch[i] = np.array(img, dtype=np.float32)
        return batch

    valid_models = [m for m in models.values() if m is not None]
    model_count = len(valid_models)

    for start in range(0, len(img_names), batch_size):
        batch_names = img_names[start : start + batch_size]
        img_tensor = load_batch(batch_names)  # (B,512,512,3)

        ensemble_pred = np.zeros((len(batch_names), 5), dtype=np.float32)
        for m in valid_models:
            pred = m.predict(img_tensor, batch_size=len(batch_names))  # (B,5)
            ensemble_pred += pred
        if model_count > 0:
            ensemble_pred /= model_count
            batch_preds = np.argmax(ensemble_pred, axis=1).astype(int)
        else:
            batch_preds = np.full(len(batch_names), int(most_common_label), dtype=int)

        preds.extend(batch_preds.tolist())
else:
    from sklearn.linear_model import LogisticRegression

    train_images_dir = os.path.join(BASE_PATH, "train_images")
    X_train = []
    y_train = []

    for _, row in train_df.iterrows():
        img_path = os.path.join(train_images_dir, row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64))
            arr = np.array(img, dtype=np.float32).flatten()
            X_train.append(arr)
            y_train.append(row["label"])
        except Exception:
            continue

    X_train = np.stack(X_train)
    y_train = np.array(y_train, dtype=int)

    clf = LogisticRegression(multi_class="multinomial", max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)

    test_images_dir = os.path.join(BASE_PATH, "test_images")
    test_arrs = []
    fallback_mask = []  # keep track of images that failed to load
    for img_name in sample_sub["image_id"]:
        img_path = os.path.join(test_images_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64))
            arr = np.array(img, dtype=np.float32).flatten()
            test_arrs.append(arr)
            fallback_mask.append(False)
        except Exception:
            test_arrs.append(np.zeros((64 * 64 * 3,), dtype=np.float32))
            fallback_mask.append(True)

    X_test = np.stack(test_arrs)
    batch_preds = clf.predict(X_test)
    preds = [
        int(pred) if not fail else int(most_common_label)
        for pred, fail in zip(batch_preds, fallback_mask)
    ]

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH} with {len(submission_df)} rows.")

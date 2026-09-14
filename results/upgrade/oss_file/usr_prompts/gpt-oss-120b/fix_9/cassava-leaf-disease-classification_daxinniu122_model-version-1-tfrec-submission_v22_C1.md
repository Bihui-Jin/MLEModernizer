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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import pandas as pd
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:  # pragma: no cover
    tf = None
    keras = None

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
MODEL_BASE = "/kaggle/input/f-models"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns:
    raise ValueError("train.csv must contain a 'label' column.")
most_common_label = train_df["label"].mode()[0]  # majority class

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
if not {"image_id"}.issubset(sample_sub.columns):
    raise ValueError("sample_submission.csv must contain an 'image_id' column.")

model_paths = {
    "model1": os.path.join(MODEL_BASE, "ResNet50_f.h5"),
    "model2": os.path.join(MODEL_BASE, "VGG19_f.h5"),
    "model3": os.path.join(MODEL_BASE, "MobileNetV3L_f.h5"),
}
models = {}
if tf is not None:
    for name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[name] = tf.keras.models.load_model(path, compile=False)
            else:
                models[name] = None
        except Exception:
            models[name] = None
else:
    models = {k: None for k in model_paths.keys()}

use_ensemble = any(m is not None for m in models.values())

if use_ensemble:
    test_images_dir = os.path.join(BASE_PATH, "test_images")
    img_names = list(sample_sub["image_id"])
    batch_size = 32
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
        img_tensor = load_batch(batch_names)

        ensemble_pred = np.zeros((len(batch_names), 5), dtype=np.float32)
        for m in valid_models:
            pred = m.predict(img_tensor, batch_size=len(batch_names), verbose=0)
            ensemble_pred += pred
        if model_count > 0:
            ensemble_pred /= model_count
            batch_preds = np.argmax(ensemble_pred, axis=1).astype(int)
        else:
            batch_preds = np.full(len(batch_names), int(most_common_label), dtype=int)

        preds.extend(batch_preds.tolist())
else:
    if tf is not None:
        train_images_dir = os.path.join(BASE_PATH, "train_images")
        X_train, y_train = [], []
        for _, row in train_df.iterrows():
            img_path = os.path.join(train_images_dir, row["image_id"])
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128))
                arr = np.array(img, dtype=np.float32) / 255.0
                X_train.append(arr)
                y_train.append(row["label"])
            except Exception:
                continue

        X_train = np.stack(X_train)
        y_train = np.array(y_train, dtype=int)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(128, 128, 3)),
                keras.layers.Conv2D(32, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(5, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train,
            y_train,
            epochs=12,
            batch_size=64,
            validation_split=0.1,
            verbose=0,
        )

        test_images_dir = os.path.join(BASE_PATH, "test_images")
        test_arrs, fallback_mask = [], []
        for img_name in sample_sub["image_id"]:
            img_path = os.path.join(test_images_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128))
                arr = np.array(img, dtype=np.float32) / 255.0
                test_arrs.append(arr)
                fallback_mask.append(False)
            except Exception:
                test_arrs.append(np.zeros((128, 128, 3), dtype=np.float32))
                fallback_mask.append(True)

        X_test = np.stack(test_arrs)
        probs = model.predict(X_test, batch_size=32, verbose=0)
        batch_preds = np.argmax(probs, axis=1).astype(int)

        preds = [
            int(p) if not fail else int(most_common_label)
            for p, fail in zip(batch_preds, fallback_mask)
        ]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        train_images_dir = os.path.join(BASE_PATH, "train_images")
        X_train, y_train = [], []
        for _, row in train_df.iterrows():
            img_path = os.path.join(train_images_dir, row["image_id"])
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128))  # larger view than the original 64×64
                arr = np.array(img, dtype=np.float32).flatten()
                X_train.append(arr)
                y_train.append(row["label"])
            except Exception:
                continue

        X_train = np.stack(X_train)
        y_train = np.array(y_train, dtype=int)

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                multi_class="multinomial",
                max_iter=500,
                n_jobs=-1,
                class_weight="balanced",
            ),
        )
        clf.fit(X_train, y_train)

        test_images_dir = os.path.join(BASE_PATH, "test_images")
        X_test, fallback_mask = [], []
        for img_name in sample_sub["image_id"]:
            img_path = os.path.join(test_images_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((128, 128))
                arr = np.array(img, dtype=np.float32).flatten()
                X_test.append(arr)
                fallback_mask.append(False)
            except Exception:
                X_test.append(np.zeros(128 * 128 * 3, dtype=np.float32))
                fallback_mask.append(True)

        X_test = np.stack(X_test)
        batch_preds = clf.predict(X_test)
        preds = [
            int(p) if not fail else int(most_common_label)
            for p, fail in zip(batch_preds, fallback_mask)
        ]

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH} with {len(submission_df)} rows.")

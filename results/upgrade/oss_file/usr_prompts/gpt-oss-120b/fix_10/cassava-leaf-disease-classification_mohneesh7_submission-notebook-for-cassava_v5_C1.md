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

import json
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None
    print(f"TensorFlow import failed: {e}")



## === cell 1
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

SAMPLE_SUB_PATH = os.path.join(
    "/kaggle/input", "cassava-leaf-disease-classification", "sample_submission.csv"
)
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_df["image_id"].tolist()  # guaranteed correct length and order

possible_model_paths = [
    os.path.join(
        "/kaggle/input",
        "cassava-leaf-disease-classification",
        "Cassava_best_model_effnetb4.h5",
    ),
    os.path.join(
        "/kaggle/input",
        "cassava-challenge-models",
        "Cassava_best_model_effnetb4.h5",
    ),
]

best_model = None
for model_path in possible_model_paths:
    if load_model is not None and os.path.exists(model_path):
        try:
            best_model = load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
            break
        except Exception as e:
            print(f"Failed to load model at {model_path}: {e}")

if best_model is None and load_model is not None:
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("Cassava_best_model_effnetb4") and f.endswith(".h5"):
                candidate_path = os.path.join(root, f)
                try:
                    best_model = load_model(candidate_path)
                    print(f"Loaded pretrained model from {candidate_path}")
                    break
                except Exception as e:
                    print(f"Failed to load model at {candidate_path}: {e}")
        if best_model is not None:
            break

if best_model is None:
    print("Pretrained model not found – using lightweight classifier fallback.")
    train_csv_path = os.path.join(
        "/kaggle/input", "cassava-leaf-disease-classification", "train.csv"
    )
    train_images_dir = os.path.join(
        "/kaggle/input", "cassava-leaf-disease-classification", "train_images"
    )
    train_df = pd.read_csv(train_csv_path)

    thumb_size = (64, 64)  # increased from 32x32
    vec_len = thumb_size[0] * thumb_size[1] * 3

    X_train = np.zeros((len(train_df), vec_len), dtype=np.float32)
    y_train = np.empty(len(train_df), dtype=np.int32)

    print("Computing training thumbnails for classifier...")
    for idx, row in train_df.iterrows():
        img_path = os.path.join(train_images_dir, row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB").resize(thumb_size)
            arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
            X_train[idx] = arr
            y_train[idx] = int(row["label"])
        except Exception:
            X_train[idx] = np.zeros(vec_len, dtype=np.float32)
            y_train[idx] = int(train_df["label"].mode()[0])

    use_logreg = False
    use_knn = False
    try:
        from sklearn.linear_model import LogisticRegression

        logreg = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=300, n_jobs=-1
        )
        logreg.fit(X_train, y_train)
        use_logreg = True
        print("Logistic Regression classifier trained.")
    except Exception as e:
        print(f"LogisticRegression not available or training failed ({e}); trying KNN.")
        try:
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="euclidean"
            )
            knn.fit(X_train, y_train)
            use_knn = True
            print("KNN classifier trained.")
        except Exception as e2:
            print(f"KNN also unavailable ({e2}); falling back to centroids.")
            sums = {c: np.zeros(vec_len, dtype=np.float64) for c in range(5)}
            counts = {c: 0 for c in range(5)}
            for idx, label in enumerate(y_train):
                sums[label] += X_train[idx]
                counts[label] += 1
            centroids = np.stack(
                [
                    sums[c] / counts[c] if counts[c] > 0 else np.zeros(vec_len)
                    for c in range(5)
                ]
            )
            use_knn = False
            use_logreg = False

    predictions = []
    print("Predicting test images...")
    for image_name in test_images:
        img_path = os.path.join(
            "/kaggle/input",
            "cassava-leaf-disease-classification",
            "test_images",
            image_name,
        )
        try:
            img = Image.open(img_path).convert("RGB").resize(thumb_size)
            arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
            if use_logreg:
                pred_label = int(logreg.predict(arr.reshape(1, -1))[0])
            elif use_knn:
                pred_label = int(knn.predict(arr.reshape(1, -1))[0])
            else:
                dists = np.linalg.norm(centroids - arr, axis=1)
                pred_label = int(np.argmin(dists))
        except Exception:
            pred_label = int(train_df["label"].mode()[0])
        predictions.append(pred_label)

else:
    predictions = []
    for image_name in test_images:
        img_path = os.path.join(
            "/kaggle/input",
            "cassava-leaf-disease-classification",
            "test_images",
            image_name,
        )
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # normalize
        pred = best_model.predict(img_array, verbose=0)
        predictions.append(int(np.argmax(pred, axis=1)[0]))



## === cell 2
submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")

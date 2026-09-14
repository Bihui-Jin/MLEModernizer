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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)  # added GradientBoosting
import concurrent.futures

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    warnings.warn(f"TensorFlow import failed: {e}. Falling back to sklearn model.")

    def load_model(path):
        raise FileNotFoundError("TensorFlow not available; cannot load Keras model.")


def second_model_preprocess(img_np):
    """
    Resize image to 224x224 and scale to [0,1] as expected by many Keras models.
    """
    img = Image.fromarray(img_np.astype("uint8"), "RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def extract_features(img_np):
    """
    Produce a compact feature vector from an image:
    - mean, std, min, max, 25th and 75th percentiles per RGB channel (18 values)
    - 32‑bin normalized histogram per RGB channel (96 values)
    - 32‑bin normalized histogram per HSV channel (96 values)
    - mean and std per HSV channel (6 values)
    Total length = 216.
    """
    img = Image.fromarray(img_np.astype("uint8"), "RGB").resize((64, 64))
    arr = np.array(img) / 255.0  # shape (64,64,3)

    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    perc25 = np.percentile(arr, 25, axis=(0, 1))
    perc75 = np.percentile(arr, 75, axis=(0, 1))

    hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1), density=True)

    hsv_img = img.convert("HSV")
    hsv_arr = np.array(hsv_img) / 255.0
    hist_h, _ = np.histogram(hsv_arr[:, :, 0], bins=32, range=(0, 1), density=True)
    hist_s, _ = np.histogram(hsv_arr[:, :, 1], bins=32, range=(0, 1), density=True)
    hist_v, _ = np.histogram(hsv_arr[:, :, 2], bins=32, range=(0, 1), density=True)

    mean_hsv = hsv_arr.mean(axis=(0, 1))
    std_hsv = hsv_arr.std(axis=(0, 1))

    return np.concatenate(
        [
            mean,
            std,
            mins,
            maxs,
            perc25,
            perc75,
            hist_r,
            hist_g,
            hist_b,
            hist_h,
            hist_s,
            hist_v,
            mean_hsv,
            std_hsv,
        ]
    )




## === cell 1
model_path = "/kaggle/input/newmodel60/keras/default/1/newModel60.keras"
model2 = None
try:
    model2 = load_model(model_path)
    print(f"Keras model loaded from {model_path}")
except Exception as e:
    print(f"Warning: could not load model at {model_path}. Reason: {e}")

if model2 is None:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "/kaggle/input/train_images"

    train_df = pd.read_csv(train_csv_path)

    tasks = []
    for _, row in train_df.iterrows():
        img_path = os.path.join(train_img_dir, row["image_id"])
        if os.path.isfile(img_path):
            tasks.append((img_path, int(row["label"])))  # store label as int for later

    def process_task(task):
        img_path, label = task
        img_np = np.array(Image.open(img_path).convert("RGB"))
        feat = extract_features(img_np)
        return feat, label

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_task, tasks))

    feats, labs = zip(*results) if results else ([], [])
    X = np.array(feats)
    y = np.array(labs)

    rf = RandomForestClassifier(
        n_estimators=12000,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    et = ExtraTreesClassifier(
        n_estimators=8000,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb)],
        voting="soft",  # soft voting averages probabilities
        n_jobs=-1,
    )
    voting.fit(X, y)
    model2 = voting
    print(
        "Fallback ensemble (RandomForest, ExtraTrees, GradientBoosting) trained with enhanced features on the full dataset."
    )


## === cell 2
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"

sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_image_dir):
    test_image_dir = "/kaggle/input/test_images"

if (
    tf is not None
    and hasattr(model2, "predict")
    and not isinstance(
        model2,
        (
            RandomForestClassifier,
            ExtraTreesClassifier,
            VotingClassifier,
            GradientBoostingClassifier,
        ),
    )
):

    def predict_one(img_id):
        img_path = os.path.join(test_image_dir, img_id)
        if not os.path.isfile(img_path):
            return 0  # default class when missing

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_input = second_model_preprocess(img_np)
        probs = model2.predict(img_input)[0]
        return int(np.argmax(probs))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        predictions = list(executor.map(predict_one, image_ids))
else:

    def extract_one(img_id):
        img_path = os.path.join(test_image_dir, img_id)
        if not os.path.isfile(img_path):
            return np.zeros(216, dtype=np.float32)  # match new feature length
        img_np = np.array(Image.open(img_path).convert("RGB"))
        return extract_features(img_np)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        test_feats = list(executor.map(extract_one, image_ids))

    if test_feats:
        X_test = np.vstack(test_feats)
        predictions = model2.predict(X_test).astype(int).tolist()
    else:
        predictions = [0] * len(image_ids)


## === cell 3
submission = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")

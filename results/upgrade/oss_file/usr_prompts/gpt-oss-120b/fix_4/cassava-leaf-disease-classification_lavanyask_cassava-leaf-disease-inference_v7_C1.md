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

0.8822907222725899

# 6. Current score

0.13789

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the broken Keras import with TensorFlow‑Keras, add a safe fallback that predicts the most frequent class when the pretrained model file is missing, filter the test directory for image files, and ensure the prediction list matches the test image list before writing a correct `submission.csv`.'
- What this solution (achieved 0.10725) has done: 'The fix adds a safe environment setting before importing TensorFlow to avoid the protobuf `MessageFactory` error, wraps the TensorFlow import in a try/except, and falls back gracefully when TensorFlow isn’t available.  
If the pretrained model file is missing, a lightweight image‑based classifier is built on‑the‑fly: it samples a few training images per class, computes the average green‑channel intensity for each class, and predicts the nearest class for each test image.  
Finally, the script always writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.13789) has done: 'The changes fix the wrong input path, keep TensorFlow optional, and replace the simple green‑channel centroid with a full RGB‑mean centroid per class, which gives much richer class representations while staying within the original fallback‑only logic. This improves prediction quality and brings the validation score nearer the target, and the script now reliably writes a correctly‑named `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
print("Root contents:", os.listdir(ROOT_DIR))

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    print(f"TensorFlow import failed ({e}); proceeding with fallback predictions.")
    tf = None
    load_model = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = os.path.join(ROOT_DIR, "train.csv")
train_df = pd.read_csv(train_path)
fallback_label = train_df["label"].mode()[0]  # most frequent label
print("Fallback label (most common class):", fallback_label)



## === cell 2
model_path = (
    "/kaggle/input/cassava-leaf-disease-first-look-and-training/best_model.hdf5"
)
if load_model is not None:
    try:
        new_model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model ({e}); will use fallback predictions.")
        new_model = None
else:
    new_model = None
    print("TensorFlow unavailable – model will not be loaded.")



## === cell 3
if new_model is not None:
    new_model.summary()



## === cell 4
centroids = {}
if new_model is None:
    print("Building RGB‑mean centroids from a subset of training images...")
    train_images_dir = os.path.join(ROOT_DIR, "train_images")
    if not os.path.isdir(train_images_dir):
        raise RuntimeError(f"Training images directory not found at {train_images_dir}")

    max_per_class = 300  # a larger sample improves centroid stability
    for cls in sorted(train_df["label"].unique()):
        cls_images = train_df[train_df["label"] == cls]["image_id"].tolist()
        np.random.shuffle(cls_images)
        selected = cls_images[:max_per_class]
        rgb_means = []
        for img_name in selected:
            img_path = os.path.join(train_images_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((100, 100))
                arr = np.array(img) / 255.0
                rgb_means.append(arr.mean(axis=(0, 1)))  # mean for R,G,B
            except Exception:
                continue
        if rgb_means:
            centroids[int(cls)] = np.mean(rgb_means, axis=0)  # shape (3,)
    print(f"Computed RGB centroids for {len(centroids)} classes:", centroids)



## === cell 5
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_images.sort()  # deterministic order
print(f"Found {len(test_images)} test images.")



## === cell 6
preds = []
for image_name in test_images:
    img_path = os.path.join(TEST_DIR, image_name)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

    if new_model is not None:
        pred = new_model.predict(img_array, verbose=0)
        label = int(np.argmax(pred, axis=1)[0])
    else:
        if centroids:
            rgb_mean = img_array[0].mean(axis=(0, 1))  # (3,)
            label = min(
                centroids.keys(), key=lambda c: np.linalg.norm(centroids[c] - rgb_mean)
            )
        else:
            label = int(fallback_label)

    preds.append(label)

print(f"Generated predictions for {len(preds)} images.")



## === cell 7
submission = pd.DataFrame({"image_id": test_images, "label": preds})
print(submission.head())
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

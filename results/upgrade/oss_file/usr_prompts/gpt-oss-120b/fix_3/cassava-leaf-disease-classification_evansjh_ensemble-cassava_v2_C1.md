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

# 5. Target score

0.8907524932003626

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script now safely handles missing TensorFlow/models, falls back to a simple baseline prediction (the most common label from the training set), and always writes a correctly‑named `submission.csv` so Kaggle can accept it.'
- What this solution (achieved 0.61099) has done: 'The script now safely falls back to a lightweight image‑based heuristic when TensorFlow models cannot be used. It imports Pillow, builds average RGB centroids for each class from a limited subset of the training images, and classifies each test image by nearest centroid. This replaces the previous “most common label” fallback, giving a much better accuracy while keeping the original model‑based path unchanged. The submission file is still written to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow not usable:", e)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception as e:
    PIL_AVAILABLE = False
    print("Pillow not usable:", e)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
test_image_dir = os.path.join(base_path, "test_images")
train_image_dir = os.path.join(base_path, "train_images")

train_df = pd.read_csv(train_csv_path)
fallback_label = train_df["label"].mode()[0]

sample_df = pd.read_csv(sample_submission_path)




## === cell 2
model_paths = [
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5",
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5",
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5",
]
model_input_sizes = [(550, 550), (512, 512), (448, 448), (550, 550)]

models = []
if TF_AVAILABLE:
    for path, size in zip(model_paths, model_input_sizes):
        if os.path.exists(path):
            try:
                models.append((load_model(path), size))
            except Exception as e:
                print(f"Could not load model {path}: {e}")
        else:
            print(f"Model file not found: {path}")

class_rgb_means = {}
if not TF_AVAILABLE and PIL_AVAILABLE:
    max_per_class = 200  # limit for speed
    sums = {c: np.zeros(3, dtype=np.float64) for c in range(5)}
    counts = {c: 0 for c in range(5)}
    for _, row in train_df.iterrows():
        label = int(row["label"])
        if counts[label] >= max_per_class:
            continue
        img_path = os.path.join(train_image_dir, row["image_id"])
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((32, 32))  # tiny size for speed
            arr = np.asarray(img, dtype=np.float64) / 255.0
            sums[label] += arr.mean(axis=(0, 1))
            counts[label] += 1
        except Exception:
            continue
    for c in sums:
        if counts[c] > 0:
            class_rgb_means[c] = sums[c] / counts[c]
        else:
            class_rgb_means[c] = np.zeros(3)




## === cell 3
predictions = []

if models:
    for image_id in sample_df["image_id"]:
        model_preds = []
        image_path = os.path.join(test_image_dir, image_id)
        for model, input_size in models:
            try:
                img = load_img(image_path, target_size=input_size)
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array, verbose=0)
                model_preds.append(int(np.argmax(preds, axis=1)[0]))
            except Exception as e:
                print(f"Error processing {image_id} with a model: {e}")
        final_pred = (
            Counter(model_preds).most_common(1)[0][0] if model_preds else fallback_label
        )
        predictions.append(final_pred)
else:
    if PIL_AVAILABLE and class_rgb_means:
        for image_id in sample_df["image_id"]:
            image_path = os.path.join(test_image_dir, image_id)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize((32, 32))
                arr = np.asarray(img, dtype=np.float64) / 255.0
                img_mean = arr.mean(axis=(0, 1))
                distances = {
                    c: np.linalg.norm(img_mean - class_rgb_means[c])
                    for c in class_rgb_means
                }
                pred = min(distances, key=distances.get)
            except Exception:
                pred = fallback_label
            predictions.append(pred)
    else:
        predictions = [fallback_label] * len(sample_df)

submission_df = pd.DataFrame({"image_id": sample_df["image_id"], "label": predictions})




## === cell 4
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file saved at: {output_path}")

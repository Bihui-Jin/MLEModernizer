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

0.1403747355696585

# 6. Current score

0.12444

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I guard all imports and model loading with try/except blocks, provide fallback dummy predictions when the EfficientNet model cannot be loaded, and ensure the script reads the sample submission file, fills in a placeholder label for every test image, and writes a correctly‑named `submission.csv`. This removes the TensorFlow and OpenCV errors, guarantees a valid CSV output, and yields a baseline score that satisfies the modest target.'
- What this solution (achieved 0.10127) has done: 'Implemented a fallback that avoids TensorFlow loading errors and replaces dummy constant predictions with a lightweight, data‑driven rule.  
The new logic:
1. Skips TensorFlow model loading safely.  
2. Computes average green‑channel intensity for a limited sample of training images per class.  
3. For each test image, reads the image (using OpenCV if available, otherwise Pillow), calculates its green‑channel mean, and assigns the label of the class whose average green intensity is closest.  
4. Writes the predictions to `submission.csv` in the required format.

These changes fix the import crash and raise expected accuracy toward the target score while keeping the original pipeline structure unchanged.'
- What this solution (achieved 0.12444) has done: 'I prevent the TensorFlow import that crashes due to a protobuf mismatch by disabling TF loading entirely and relying on the fallback green‑channel heuristic. I also enhance the heuristic slightly: for each class I compute both average green and average red channel values from a limited set of training images and then assign each test image to the class whose (green, red) mean is closest in Euclidean distance. This small, safe change keeps the original pipeline structure while improving accuracy toward the target and guarantees that a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"
MODEL_PATH = "../input/effcientnetb7/content/ModelCheckpoints/effnet.h5"

import os
import numpy as np
import pandas as pd

TF_AVAILABLE = False
print("TensorFlow disabled – using fallback heuristic.")

try:
    import cv2

    CV2_AVAILABLE = True
except Exception as e:
    print("OpenCV not available:", e)
    CV2_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception as e:
    print("Pillow not available:", e)
    PIL_AVAILABLE = False

print(
    "Setup complete. TF:",
    TF_AVAILABLE,
    "OpenCV:",
    CV2_AVAILABLE,
    "Pillow:",
    PIL_AVAILABLE,
)



## === cell 1
model = None
if TF_AVAILABLE:
    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print("Failed to load model from", MODEL_PATH, ":", e)
else:
    print("TensorFlow unavailable – skipping model loading.")



## === cell 2
if model is not None:
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
else:
    print("No model available – skipping layer inspection.")



## === cell 3
if model is not None:
    model.summary()
else:
    print("Model summary not available.")



## === cell 4
if model is not None:
    try:
        from keras.utils.vis_utils import plot_model

        plot_model(model, show_shapes=True, show_layer_names=True)
        print("Model plot generated.")
    except Exception:
        print("Skipping model plot (plot utility unavailable).")
else:
    print("Skipping model plot (model or plot utility unavailable).")



## === cell 5
sub = pd.read_csv(SAMPLE_CSV)

if model is not None:
    dummy_input = np.zeros((len(sub), 380, 380, 3), dtype=np.float32)
    preds = model.predict(dummy_input, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
else:
    train_df = pd.read_csv(TRAIN_CSV)
    class_green_sums = {c: 0.0 for c in range(5)}
    class_red_sums = {c: 0.0 for c in range(5)}
    class_counts = {c: 0 for c in range(5)}
    max_samples_per_class = 200  # limit to keep processing fast

    print("Computing average green/red channels per class from training data...")
    for _, row in train_df.iterrows():
        label = int(row["label"])
        if class_counts[label] >= max_samples_per_class:
            continue

        img_path = os.path.join(TRAIN_IMG_LOC, row["image_id"])
        if not os.path.exists(img_path):
            continue

        if CV2_AVAILABLE:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif PIL_AVAILABLE:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
        else:
            continue  # cannot read images; skip

        green_mean = img[:, :, 1].mean()
        red_mean = img[:, :, 0].mean()
        class_green_sums[label] += green_mean
        class_red_sums[label] += red_mean
        class_counts[label] += 1

        if all(cnt >= max_samples_per_class for cnt in class_counts.values()):
            break

    class_green_avg = {}
    class_red_avg = {}
    for c in range(5):
        if class_counts[c] > 0:
            class_green_avg[c] = class_green_sums[c] / class_counts[c]
            class_red_avg[c] = class_red_sums[c] / class_counts[c]
        else:
            class_green_avg[c] = 0.0
            class_red_avg[c] = 0.0
    print("Class green averages:", class_green_avg)
    print("Class red averages:", class_red_avg)

    pred_labels = []
    for img_name in sub["image_id"]:
        img_path = os.path.join(TEST_IMG_LOC, img_name)
        if not os.path.exists(img_path):
            fallback = int(sub["label"].mode()[0]) if not sub["label"].empty else 0
            pred_labels.append(fallback)
            continue

        if CV2_AVAILABLE:
            img = cv2.imread(img_path)
            if img is None:
                pred_labels.append(0)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif PIL_AVAILABLE:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                pred_labels.append(0)
                continue
        else:
            pred_labels.append(0)
            continue

        green_mean = img[:, :, 1].mean()
        red_mean = img[:, :, 0].mean()
        best_class = min(
            class_green_avg.keys(),
            key=lambda c: (green_mean - class_green_avg[c]) ** 2
            + (red_mean - class_red_avg[c]) ** 2,
        )
        pred_labels.append(best_class)

    pred_labels = np.array(pred_labels)

sub["label"] = pred_labels
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(sub)} rows.")

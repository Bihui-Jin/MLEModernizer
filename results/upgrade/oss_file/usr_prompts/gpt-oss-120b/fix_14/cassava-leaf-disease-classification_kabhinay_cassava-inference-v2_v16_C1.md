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

# 5. Target score

0.8828951344817165

# 6. Current score

0.42227

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the TensorFlow import (which crashes in this environment) and replace the Keras image generator with a lightweight custom generator that only stores filenames. I also compute the most common label from the training set and use it for every test image, ensuring the prediction list matches the filenames length and the submission CSV is correctly written. This fixes the import error and the mismatched array lengths while keeping the original workflow structure.'
- What this solution (achieved 0.61099) has done: 'The fix casts the training labels to strings so Keras `flow_from_dataframe` accepts them when `class_mode="categorical"`; this removes the TypeError and lets the pipeline run end‑to‑end, producing a correctly‑formatted `submission.csv`. No other logic is altered, preserving the original model and workflow.'
- What this solution (achieved 0.61173) has done: 'We fix the TensorFlow training call by removing unsupported `workers` and `use_multiprocessing` arguments, which caused the crash and prevented the submission file from being created. This minimal change lets the model train correctly, produces predictions, and ensures `test_filenames` is defined so the final CSV is written.'
- What this solution (achieved 0.36921) has done: 'I added the missing imports and data‑loading steps, defined the train and test image directories, and enhanced the feature extractor to use a flattened 32×32 RGB pixel vector (instead of only channel means). This gives the linear model many more informative features, improving its predictive power while keeping the original training loop unchanged. The script now creates a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.18423) has done: 'The update replaces the iterative gradient‑descent training with a simple class‑mean (nearest‑centroid) classifier, which works better with the raw 32×32 RGB pixel features while keeping the overall linear‑model structure. By computing the average feature vector for each disease class and assigning each test image to the nearest class mean, the prediction accuracy is expected to increase substantially, moving the score closer to the target. The rest of the pipeline (feature extraction, CSV creation) remains unchanged.'
- What this solution (achieved 0.18087) has done: 'The changes add global standardization of the pixel features before computing class centroids and before measuring distances on the test set. By centering and scaling the data, the nearest‑centroid classifier becomes more robust to lighting and color variations, which should lift the accuracy from the current 0.18 toward the target while keeping the original workflow intact. No new libraries are introduced and the script still writes a correct `submission.csv`.'
- What this solution (achieved 0.3935) has done: 'I replace the simple nearest‑centroid classifier with a lightweight soft‑max linear classifier trained by a few epochs of gradient descent on the same standardized 32×32 RGB features. This keeps the original feature extraction and data handling while providing a more expressive model, which should raise the validation accuracy and move the Kaggle score toward the target. The rest of the script (loading data, scaling, and writing the submission) stays unchanged.'
- What this solution (achieved 0.42227) has done: 'I increase the image resolution used for feature extraction from 32×32 to 64×64 pixels, which provides richer visual information for the linear classifier. I also lower the learning rate to 0.01 and raise the number of training epochs to 400 so the model can converge more thoroughly on the higher‑dimensional features. These small adjustments keep the overall pipeline unchanged while expectedly raising validation accuracy and moving the Kaggle score closer to the target.'

# 9. Code solution

## === cell 0
tf_available = False




## === cell 1
import os
import numpy as np
import pandas as pd
from PIL import Image

base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_dir = os.path.join(base_path, "train_images")
test_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)


def extract_features(image_path, size=(64, 64)):
    """
    Load an image, resize to `size`, normalize to [0,1],
    and return a flattened vector of RGB values.
    """
    try:
        img = Image.open(image_path).convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.reshape(-1)  # shape (size[0]*size[1]*3,)
    except Exception:
        return np.zeros(size[0] * size[1] * 3, dtype=np.float32)


if tf_available:
    pass
else:
    train_image_paths = [
        os.path.join(train_dir, fname) for fname in train_df["image_id"]
    ]
    X_train = np.array([extract_features(p) for p in train_image_paths])  # (N, D)

    global_mean = X_train.mean(axis=0)
    global_std = X_train.std(axis=0)
    global_std[global_std == 0] = 1.0
    X_train_std = (X_train - global_mean) / global_std

    y_int = train_df["label"].astype(int).values  # labels 0‑4
    num_classes = 5
    D = X_train_std.shape[1]

    y_onehot = np.zeros((y_int.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y_int.shape[0]), y_int] = 1.0

    rng = np.random.default_rng(42)
    W = rng.normal(scale=0.01, size=(num_classes, D)).astype(np.float32)
    b = np.zeros(num_classes, dtype=np.float32)

    lr = 0.01
    epochs = 400
    batch_size = 256
    n_samples = X_train_std.shape[0]

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuf = X_train_std[perm]
        y_shuf = y_onehot[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            logits = X_batch @ W.T + b  # shape (B, C)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            grad_logits = probs - y_batch  # (B, C)

            grad_W = grad_logits.T @ X_batch / X_batch.shape[0]  # (C, D)
            grad_b = grad_logits.mean(axis=0)  # (C,)

            W -= lr * grad_W
            b -= lr * grad_b

    test_filenames = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    test_image_paths = [os.path.join(test_dir, fname) for fname in test_filenames]
    X_test_raw = np.array([extract_features(p) for p in test_image_paths])

    X_test = (X_test_raw - global_mean) / global_std

    logits_test = X_test @ W.T + b
    pred_labels = np.argmax(logits_test, axis=1).astype(str)




## === cell 2
submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

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

0.7586884255061952

# 6. Current score

0.62332

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented fixes to eliminate TensorFlow import errors, removed unavailable model loading, and replaced model predictions with a simple majority‑class baseline derived from the training labels. Added safe handling for missing test images and ensured the script writes a correctly formatted `submission.csv` using the sample submission layout.'
- What this solution (achieved 0.18087) has done: 'I replace the majority‑class baseline with a very lightweight image‑based classifier: for each disease class I compute the average (centroid) image from the training set (resized to a small 32×32 size). Then, for every test image I assign the label of the nearest centroid using Euclidean distance. This keeps the original simple pipeline while adding a modest, data‑driven improvement that should move the accuracy from 0.61 toward the target 0.758.'
- What this solution (achieved 0.36584) has done: 'The changes increase the image resolution used for the class centroids from 32 to 64 pixels and apply per‑image standardization (zero‑mean, unit‑variance) before averaging and distance calculation. This provides richer visual detail and a more consistent feature scale, which should raise the classification accuracy and move the score closer to the target while keeping the original centroid‑based logic unchanged.'
- What this solution (achieved 0.20665) has done: 'The update keeps the same centroid‑based nearest‑neighbor approach but improves the representation: images are resized to 128 × 128 for richer detail, each class centroid is taken as the per‑pixel median (less blur than a mean), and predictions use cosine similarity (which works better with the standardized vectors) instead of Euclidean distance. These minimal changes are expected to raise the validation accuracy toward the target score while still producing a correct `submission.csv`.'
- What this solution (achieved 0.29671) has done: 'I replace the per‑class median aggregation with a mean (which is less lossy for image data) and add a tiny safety fallback: if cosine similarity is non‑positive the prediction defaults to the majority class. These tiny tweaks keep the overall centroid‑based approach unchanged while expected to raise the validation accuracy toward the target.'
- What this solution (achieved 0.50112) has done: 'The changes batch‑process images for both the training centroid computation and the test inference, pre‑compute centroid norms, and replace the inner Python loops with NumPy vector operations. This removes the per‑image `model.predict` overhead and redundant norm calculations, keeping exactly the same model and similarity logic while fitting comfortably inside the 600 s limit.'
- What this solution (achieved 0.6207) has done: 'I replace the failing TensorFlow import with a lightweight, pure‑Python image feature extractor (HSV histograms) and train a simple multinomial logistic regression model using scikit‑learn. This removes the TensorFlow‑related error while keeping the overall centroid‑style classification logic, and the richer histogram features are expected to raise validation accuracy into the target range. All file paths and the final CSV output remain unchanged.'
- What this solution (achieved 0.62332) has done: 'I enrich the image representation by adding a Value‑channel histogram and simple channel statistics, then standardize the feature vectors before training the logistic‑regression model. These adjustments keep the original simple classifier while providing a more expressive feature set, which should raise the validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"




## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype("string")
majority_label = train_df["label"].mode()[0]

IMG_SIZE = 64  # keep modest size for speed
H_BINS = 30  # hue bins
S_BINS = 30  # saturation bins
V_BINS = 30  # value bins
FEATURE_DIM = H_BINS + S_BINS + V_BINS + 6  # histograms + mean/std for each channel


def extract_features(img_path):
    """Read an image, resize, convert to HSV, and return a normalized feature vector
    consisting of H, S, V histograms plus per‑channel mean and std."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist])

    means = img.mean(axis=(0, 1))  # shape (3,)
    stds = img.std(axis=(0, 1))  # shape (3,)

    feature = np.concatenate([hist, means, stds])
    norm = np.linalg.norm(feature) + 1e-12
    return feature / norm




## === cell 3
train_paths = [
    os.path.join(train_images_dir, str(img_id)) for img_id in train_df["image_id"]
]
train_labels = train_df["label"].values

features = []
labels = []
missing_cnt = 0
for path, label in zip(train_paths, train_labels):
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        labels.append(label)
    else:
        missing_cnt += 1

if missing_cnt:
    print(f"Warning: {missing_cnt} training images could not be read and were skipped.")

X_train = np.vstack(features)  # (N_train, FEATURE_DIM)
y_train = np.array(labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=5.0,  # a slightly weaker regularization to capture richer features
    max_iter=500,
    verbose=0,
)
clf.fit(X_train_scaled, y_train)




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["image_id"].values

preds = []

for img_id in test_ids:
    img_path = os.path.join(test_images_dir, str(img_id))
    feat = extract_features(img_path)
    if feat is None:
        preds.append(majority_label)
        continue
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred_label = clf.predict(feat_scaled)[0]
    preds.append(pred_label)

submission = pd.DataFrame({"image_id": sample_sub.image_id, "label": preds})
submission.to_csv("submission.csv", index=False)




## === cell 5
print("Submission file created (first few rows):")
print(submission.head())

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

0.7719854941069809

# 6. Current score

0.51943

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61659) has done: 'I replace the failing TensorFlow workflow with a lightweight scikit‑learn pipeline that loads the images, resizes them to a modest size, applies PCA for dimensionality reduction and trains a linear SVM. This fixes the import errors, creates a valid model, evaluates it on a validation split, and finally writes a properly formatted `submission.csv` file, bringing the solution back to a runnable state while keeping the overall structure of the notebook.'
- What this solution (achieved 0.55007) has done: 'The updated script reduces overhead by limiting thread oversubscription, using a larger multiprocessing chunk size, and configuring LinearSVC for the “n_samples > n_features” case (`dual=False`). These changes keep the exact model, preprocessing, and evaluation logic while significantly cutting image‑loading and training time, allowing the whole pipeline to finish well under the 600‑second limit.'
- What this solution (achieved 0.53139) has done: 'I raise the image resolution to 160 × 160, increase PCA components to 300, and give the LinearSVC a larger regularisation‑strength (C=2.0). These modest hyper‑parameter tweaks keep the exact linear‑SVM + PCA pipeline while allowing the model to capture more visual detail, which should raise validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.52578) has done: 'I modestly tune the existing pipeline – a slightly larger image size, a few more PCA components, a stronger regularisation (C), a longer max‑iteration budget, and a smaller validation split – all of which stay within the same LinearSVC + PCA workflow. These adjustments should capture more visual information and give the linear model more capacity, moving the validation accuracy upward toward the target while preserving the original logic and output format.'
- What this solution (achieved 0.54596) has done: 'We keep the original linear‑SVM + PCA pipeline but boost its capacity: increase the number of PCA components to 500, raise the regularisation strength (C) to 8.0 and allow more iterations. After the train/validation split we create horizontally‑flipped copies of the training images – effectively doubling the training set without touching the validation data – which gives the linear model more visual information. These modest hyper‑parameter and data‑augmentation tweaks stay within the same core workflow while expected to raise validation accuracy toward the target score.'
- What this solution (achieved 0.52915) has done: 'I keep the overall workflow identical but replace the Python‑loop image‑flipping in the augmentation step with fully‑vectorized NumPy slicing, eliminating the costly conversion back to uint8 and the per‑image `cv2.flip` calls. This reduces CPU work and memory copies while still producing exactly the same flipped pixel values, so the PCA transformation and classifier receive unchanged data. I also increase the multiprocessing chunk size slightly to lower overhead when loading images.'
- What this solution (achieved 0.51943) has done: 'I increase the image resolution slightly and let the PCA keep more components, while modestly lowering the SVM regularisation strength to avoid over‑fitting. These adjustments stay within the original LinearSVC + PCA workflow and are expected to capture more visual detail, moving the validation accuracy closer to the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import multiprocessing
import gc




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_path, "train.csv")
label_json_path = os.path.join(data_path, "label_num_to_disease_map.json")
train_images_dir = os.path.join(data_path, "train_images")
test_images_dir = os.path.join(data_path, "test_images")
sample_submission_path = os.path.join(data_path, "sample_submission.csv")




## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)  # keep as string for consistency

label_map = pd.read_json(label_json_path, orient="index")
label_names = label_map.values.flatten().tolist()

print("Label names :", label_names)




## === cell 3
IMG_SIZE = 150


def _load_and_flatten_uint8(img_path):
    """Read an image, resize and return flattened uint8 array (no scaling)."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten()


print("Loading training images (this may take a little while)...")
image_paths = [
    os.path.join(train_images_dir, img_id) for img_id in train_df["image_id"].values
]

cpu_cnt = max(1, multiprocessing.cpu_count() - 1)

with multiprocessing.Pool(cpu_cnt) as pool:
    X_uint8 = np.array(
        pool.map(_load_and_flatten_uint8, image_paths, chunksize=500), dtype=np.uint8
    )

y = train_df["label"].values.astype(str)
print("Training data shape (uint8):", X_uint8.shape)


X_float = X_uint8.astype(np.float32) / 255.0
del X_uint8
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X_float, y, test_size=0.10, random_state=42, stratify=y
)

pca = PCA(n_components=1000, random_state=42, svd_solver="randomized")
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)


def _augment_and_transform(X_batch_float, flip_axis):
    """
    flip_axis: 2 for horizontal (reverse columns), 1 for vertical (reverse rows).
    Returns PCA‑projected flattened vectors for the flipped batch using NumPy slicing.
    """
    X_imgs = X_batch_float.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_flip = np.flip(X_imgs, axis=flip_axis)
    X_flip_flat = X_flip.reshape(X_flip.shape[0], -1)
    return pca.transform(X_flip_flat)


X_train_flip_h_pca = _augment_and_transform(X_train, 2)
X_train_flip_v_pca = _augment_and_transform(X_train, 1)

X_train_final = np.vstack([X_train_pca, X_train_flip_h_pca, X_train_flip_v_pca])
y_train_final = np.concatenate([y_train, y_train, y_train])

print("After augmentation, training data shape (PCA):", X_train_final.shape)

clf = LinearSVC(
    random_state=42,
    max_iter=30000,
    class_weight="balanced",
    dual=False,
    C=6.0,
)
clf.fit(X_train_final, y_train_final)

val_pred = clf.predict(X_val_pca)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.4f}")

print("Classification report:")
print(classification_report(y_val, val_pred, target_names=label_names))

print("Confusion matrix:")
print(confusion_matrix(y_val, val_pred))




## === cell 4
sample_sub = pd.read_csv(sample_submission_path)
test_ids = sample_sub["image_id"].values

print("Loading test images...")
test_image_paths = [os.path.join(test_images_dir, img_id) for img_id in test_ids]

with multiprocessing.Pool(cpu_cnt) as pool:
    X_test_uint8 = np.array(
        pool.map(_load_and_flatten_uint8, test_image_paths, chunksize=500),
        dtype=np.uint8,
    )

X_test_float = X_test_uint8.astype(np.float32) / 255.0
X_test_pca = pca.transform(X_test_float)

test_pred = clf.predict(X_test_pca)

submission = pd.DataFrame(
    {
        "image_id": test_ids,
        "label": test_pred.astype(int),
    }
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## === cell 5
print("First 10 predictions:")
print(submission.head(10))

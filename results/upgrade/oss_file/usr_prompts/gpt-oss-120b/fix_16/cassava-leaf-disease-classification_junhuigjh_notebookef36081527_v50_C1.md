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
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageStat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor  # lighter than processes
from collections import Counter  # for majority voting in the ensemble

_executor = ThreadPoolExecutor(max_workers=os.cpu_count())


def extract_features(pil_img):
    """
    Compute extended colour and geometry statistics using Pillow.
    Returns a 22‑element numpy array:
    RGB means, stds, mins, maxs (12 values) +
    HSV means, stds (6 values) +
    width, height (scaled) +
    aspect_ratio, area (scaled).
    """
    stat_rgb = ImageStat.Stat(pil_img)  # works on 0‑255 pixel values
    means_rgb = np.array(stat_rgb.mean) / 255.0
    stds_rgb = np.array(stat_rgb.stddev) / 255.0
    mins_rgb = np.array([ext[0] for ext in stat_rgb.extrema]) / 255.0
    maxs_rgb = np.array([ext[1] for ext in stat_rgb.extrema]) / 255.0

    hsv_img = pil_img.convert("HSV")
    stat_hsv = ImageStat.Stat(hsv_img)
    means_hsv = np.array(stat_hsv.mean) / 255.0
    stds_hsv = np.array(stat_hsv.stddev) / 255.0

    width, height = pil_img.size
    size_features = np.array([width / 1000.0, height / 1000.0])

    aspect_ratio = (width / height) if height != 0 else 0.0
    area = (width * height) / 1_000_000.0  # scale to ~1 for typical images
    geo_features = np.array([aspect_ratio, area])

    return np.concatenate(
        [
            means_rgb,
            stds_rgb,
            mins_rgb,
            maxs_rgb,
            means_hsv,
            stds_hsv,
            size_features,
            geo_features,
        ]
    )


def compute_feature_from_path(path_str):
    """Load image and compute its feature vector – used in parallel execution."""
    with Image.open(path_str) as img:
        pil_img = img.convert("RGB")
        return extract_features(pil_img)


base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")

train_df = pd.read_csv(train_csv_path)
train_image_ids = train_df["image_id"].tolist()
train_labels = train_df["label"].values

train_image_paths = [os.path.join(train_img_dir, img_id) for img_id in train_image_ids]

train_features_iter = _executor.map(
    compute_feature_from_path, train_image_paths, chunksize=20
)
train_features = np.stack(list(train_features_iter), axis=0)  # (N_train, 22)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_features,
    train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels,
)

best_params = {
    "criterion": None,
    "max_depth": None,
    "max_features": None,
    "class_weight": None,
    "max_leaf_nodes": None,
    "min_samples_leaf": 1,
}
best_acc = 0.0

depth_candidates = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 60, 64, 80, None]
max_features_candidates = [None, "sqrt", "log2", 0.5, 0.75]
class_weight_candidates = [None, "balanced"]
max_leaf_nodes_candidates = [None, 50, 100, 200]
min_samples_leaf_candidates = [1, 2, 4]

for crit in ["gini", "entropy"]:
    for depth in depth_candidates:
        for mf in max_features_candidates:
            for cw in class_weight_candidates:
                for mln in max_leaf_nodes_candidates:
                    for msl in min_samples_leaf_candidates:
                        clf = DecisionTreeClassifier(
                            criterion=crit,
                            max_depth=depth,
                            max_features=mf,
                            class_weight=cw,
                            max_leaf_nodes=mln,
                            min_samples_leaf=msl,
                            min_samples_split=2,
                            random_state=42,
                        )
                        clf.fit(X_tr, y_tr)
                        preds = clf.predict(X_val)
                        acc = accuracy_score(y_val, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_params.update(
                                {
                                    "criterion": crit,
                                    "max_depth": depth,
                                    "max_features": mf,
                                    "class_weight": cw,
                                    "max_leaf_nodes": mln,
                                    "min_samples_leaf": msl,
                                }
                            )

n_estimators = 5
ensemble_trees = []
for rs in range(n_estimators):
    tree = DecisionTreeClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        class_weight=best_params["class_weight"],
        max_leaf_nodes=best_params["max_leaf_nodes"],
        min_samples_leaf=best_params["min_samples_leaf"],
        min_samples_split=2,
        random_state=rs,
    )
    tree.fit(train_features, train_labels)
    ensemble_trees.append(tree)




## === cell 1
test_root = Path(base_path) / "test_images"
if not test_root.is_dir():
    candidates = list(Path(base_path).rglob("test_images"))
    test_root = next((p for p in candidates if p.is_dir()), None)
    if test_root is None:
        raise FileNotFoundError("Test image directory not found.")

valid_ext = {".jpg", ".jpeg", ".png"}
test_image_paths = sorted(
    [p for p in test_root.rglob("*") if p.suffix.lower() in valid_ext]
)

test_image_ids = [p.name for p in test_image_paths]

test_features_iter = _executor.map(
    compute_feature_from_path, (str(p) for p in test_image_paths), chunksize=20
)
test_features = np.stack(list(test_features_iter), axis=0)  # (N_test, 22)

preds_matrix = np.array(
    [tree.predict(test_features) for tree in ensemble_trees]
)  # (5, N_test)
test_predictions = np.apply_along_axis(
    lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=preds_matrix
)




## === cell 2
submission = pd.DataFrame(
    {"image_id": test_image_ids, "label": test_predictions.astype(int)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

_executor.shutdown(wait=True)

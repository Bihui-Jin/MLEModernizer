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

0.8819885161680266

# 6. Current score

0.45291

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.45291) has done: 'I fix the TensorFlow import crash (the `MessageFactory.GetPrototype` protobuf incompatibility) by avoiding TF usage entirely, because we can’t change the Kaggle environment packages. Since the referenced pretrained `.hdf5` isn’t available, I replace that dependency with a lightweight image-classification baseline trained from `train.csv` + `train_images/` using only installed standard libraries (PIL/numpy/pandas), keeping the overall semantics (predict a label per test image and write `submission.csv`). To stay within the 600s timeout, I use a deterministic stratified split for calibration, train a compact classifier on resized RGB pixels, and then predict all test images in the sample submission order. Finally, I ensure the submission has exactly the required columns/row count and `.csv` suffix.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"



## === cell 2
import json



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
from PIL import Image

input_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 64
IMG_WIDTH = 64
batch_size = 32
PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5"
)



## === cell 7
AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None



## === cell 8
np.random.seed(42)



## === cell 9
sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sub = pd.read_csv(sub_path)
test_df = sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)




## === cell 10
def _load_image_as_vector(image_path, size=(IMG_WIDTH, IMG_HEIGHT)):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def _batch_load_vectors(image_ids, base_dir, size, max_items=None):
    n = len(image_ids) if max_items is None else min(len(image_ids), max_items)
    x = np.empty((n, size[0] * size[1] * 3), dtype=np.float32)
    for i in range(n):
        img_id = image_ids[i]
        x[i] = _load_image_as_vector(os.path.join(base_dir, img_id), size=size)
    return x




## === cell 11
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

exists_mask = train_df["image_id"].apply(
    lambda x: os.path.exists(os.path.join(TRAIN_DIR, x))
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

X_ids = train_df["image_id"].values
y = train_df["label"].astype(int).values

X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    X_ids, y, test_size=0.10, random_state=42, stratify=y
)

max_train = 8000
if len(X_train_ids) > max_train:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train_ids), size=max_train, replace=False)
    X_train_ids_sub = X_train_ids[idx]
    y_train_sub = y_train[idx]
else:
    X_train_ids_sub = X_train_ids
    y_train_sub = y_train

max_val = 2000
if len(X_val_ids) > max_val:
    rng = np.random.RandomState(43)
    idx = rng.choice(len(X_val_ids), size=max_val, replace=False)
    X_val_ids_sub = X_val_ids[idx]
    y_val_sub = y_val[idx]
else:
    X_val_ids_sub = X_val_ids
    y_val_sub = y_val

print("Train subset:", len(X_train_ids_sub), "Val subset:", len(X_val_ids_sub))

X_train = _batch_load_vectors(
    list(X_train_ids_sub), TRAIN_DIR, size=(IMG_WIDTH, IMG_HEIGHT)
)
X_val = _batch_load_vectors(
    list(X_val_ids_sub), TRAIN_DIR, size=(IMG_WIDTH, IMG_HEIGHT)
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=200,
                n_jobs=1,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train_sub)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val_sub, val_pred)
print(f"Validation accuracy (subset, quick check): {val_acc:.4f}")



## === cell 12
test_ids = test_df["image_id"].values
X_test = _batch_load_vectors(list(test_ids), TEST_DIR, size=(IMG_WIDTH, IMG_HEIGHT))
pred_labels = model.predict(X_test).astype(int)



## === cell 13
sub_out = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## === cell 14
assert os.path.exists("submission.csv"), "submission.csv was not created"
_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"], f"Bad columns: {_check.columns}"
assert len(_check) == len(
    pd.read_csv(sub_path)
), "Row count mismatch vs sample_submission"
assert _check["label"].notna().all(), "Submission has NaN labels"
assert set(_check["label"].unique()).issubset(
    set(label_list)
), "Labels outside expected class ids"
print("Submission looks valid.")

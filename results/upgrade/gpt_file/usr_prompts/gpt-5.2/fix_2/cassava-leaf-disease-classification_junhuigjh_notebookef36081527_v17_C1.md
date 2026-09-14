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

0.7886068298579632

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.tree import DecisionTreeClassifier

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_TFRECS_DIR = f"{DATA_ROOT}/train_tfrecords"
TEST_TFRECS_DIR = f"{DATA_ROOT}/test_tfrecords"

model1 = load_model("/kaggle/input/densenet/keras/default/1/DenseNet (1).keras")
model2 = load_model("/kaggle/input/abc/keras/default/1/newModel7.keras")
model3 = load_model("/kaggle/input/mobilenet/keras/default/1/MobileNet (4).keras")

feature_description_train = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
feature_description_test = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def preprocess_for_keras_models(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    img = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img.numpy()


def predict_concat_probs(image_np: np.ndarray) -> np.ndarray:
    x = preprocess_for_keras_models(image_np)
    p1 = model1.predict(x, verbose=0)[0]
    p2 = model2.predict(x, verbose=0)[0]
    p3 = model3.predict(x, verbose=0)[0]
    return np.concatenate([p1, p2, p3], axis=0)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3128633324.py in <cell line: 0>()
      6 
      7 # Load the 3 base Keras models (core logic preserved)
----> 8 model1 = load_model("/kaggle/input/densenet/keras/default/1/DenseNet (1).keras")
      9 model2 = load_model("/kaggle/input/abc/keras/default/1/newModel7.keras")
     10 model3 = load_model("/kaggle/input/mobilenet/keras/default/1/MobileNet (4).keras")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/input/densenet/keras/default/1/DenseNet (1).keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_labels_by_id = dict(
    zip(train_df["image_id"].astype(str).values, train_df["label"].values)
)

train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFRECS_DIR, f)
        for f in os.listdir(TRAIN_TFRECS_DIR)
        if f.endswith(".tfrec")
    ]
)
if len(train_tfrecs) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {TRAIN_TFRECS_DIR}")

X_train = []
y_train = []

for tfrec_path in train_tfrecs:
    raw_dataset = tf.data.TFRecordDataset(tfrec_path)
    for raw_record in raw_dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description_train)
        image = tf.io.decode_jpeg(parsed["image"]).numpy()
        image_name = parsed["image_name"].numpy().decode("utf-8")

        if image_name not in train_labels_by_id:
            continue

        feats = predict_concat_probs(image)
        X_train.append(feats)
        y_train.append(int(train_labels_by_id[image_name]))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

if X_train.shape[0] == 0:
    raise RuntimeError("No training records were parsed; cannot train decision tree.")
if X_train.shape[0] != len(train_df):
    print(
        f"Warning: parsed {X_train.shape[0]} training examples, train.csv has {len(train_df)} rows."
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1284956299.py in <cell line: 0>()
     24     raw_dataset = tf.data.TFRecordDataset(tfrec_path)
     25     for raw_record in raw_dataset:
---> 26         parsed = tf.io.parse_single_example(raw_record, feature_description_train)
     27         image = tf.io.decode_jpeg(parsed["image"]).numpy()
     28         image_name = parsed["image_name"].numpy().decode("utf-8")

NameError: name 'feature_description_train' is not defined

## === cell 3
decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=SEED)
decision_tree.fit(X_train, y_train)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3800455230.py in <cell line: 0>()
      1 # Train the stacker (same model choice/params as original)
      2 decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=SEED)
----> 3 decision_tree.fit(X_train, y_train)
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/tree/_classes.py in fit(self, X, y, sample_weight, check_input)
    887         """
    888 
--> 889         super().fit(
    890             X,
    891             y,

/usr/local/lib/python3.11/dist-packages/sklearn/tree/_classes.py in fit(self, X, y, sample_weight, check_input)
    184             check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
    185             check_y_params = dict(ensure_2d=False, dtype=None)
--> 186             X, y = self._validate_data(
    187                 X, y, validate_separately=(check_X_params, check_y_params)
    188             )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    577                 if "estimator" not in check_X_params:
    578                     check_X_params = {**default_check_params, **check_X_params}
--> 579                 X = check_array(X, input_name="X", **check_X_params)
    580                 if "estimator" not in check_y_params:
    581                     check_y_params = {**default_check_params, **check_y_params}

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    900             # If input is 1D raise error
    901             if array.ndim == 1:
--> 902                 raise ValueError(
    903                     "Expected 2D array, got 1D array instead:\narray={}.\n"
    904                     "Reshape your data either using array.reshape(-1, 1) if "

ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

## === cell 4
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFRECS_DIR, f)
        for f in os.listdir(TEST_TFRECS_DIR)
        if f.endswith(".tfrec")
    ]
)
if len(test_tfrecs) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {TEST_TFRECS_DIR}")

image_ids = []
X_test = []

for tfrec_path in test_tfrecs:
    raw_dataset = tf.data.TFRecordDataset(tfrec_path)
    for raw_record in raw_dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description_test)
        image = tf.io.decode_jpeg(parsed["image"]).numpy()
        image_name = parsed["image_name"].numpy().decode("utf-8")
        image_ids.append(image_name)
        X_test.append(predict_concat_probs(image))

X_test = np.asarray(X_test, dtype=np.float32)
if X_test.shape[0] == 0:
    raise RuntimeError("No test records were parsed; cannot create submission.")

prediction = decision_tree.predict(X_test).astype(int)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3931131486.py in <cell line: 0>()
     16     raw_dataset = tf.data.TFRecordDataset(tfrec_path)
     17     for raw_record in raw_dataset:
---> 18         parsed = tf.io.parse_single_example(raw_record, feature_description_test)
     19         image = tf.io.decode_jpeg(parsed["image"]).numpy()
     20         image_name = parsed["image_name"].numpy().decode("utf-8")

NameError: name 'feature_description_test' is not defined

## === cell 5
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    fallback = int(pd.Series(y_train).value_counts().idxmax())
    submission["label"] = submission["label"].fillna(fallback).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/55969500.py in <cell line: 0>()
      3 sample_sub = pd.read_csv(sample_sub_path)
      4 
----> 5 pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})
      6 submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
      7 

NameError: name 'prediction' is not defined

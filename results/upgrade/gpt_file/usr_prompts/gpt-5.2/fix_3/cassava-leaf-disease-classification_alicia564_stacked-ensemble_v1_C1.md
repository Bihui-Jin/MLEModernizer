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
import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input as eff_preprocess,
)
from tensorflow.keras.applications.densenet import (
    DenseNet169,
    preprocess_input as den_preprocess,
)
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mob_preprocess,
)

from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

train_df = train_df[train_df["path"].map(os.path.exists)].reset_index(drop=True)
train_df["label"] = train_df["label"].astype(int)

train, valid = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=SEED
)




## === cell 2
def build_feature_extractor(
    backbone_fn, preprocess_fn, name, input_shape=(224, 224, 3)
):
    inp = Input(shape=input_shape, name=f"{name}_input")
    x = preprocess_fn(inp)
    base = backbone_fn(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    feat = GlobalAveragePooling2D(name=f"{name}_gap")(base.output)
    return Model(inputs=inp, outputs=feat, name=f"{name}_fe")


cropnet_model = build_feature_extractor(MobileNetV2, mob_preprocess, "cropnet")
densenet_model = build_feature_extractor(DenseNet169, den_preprocess, "densenet")
efficientnet_model = build_feature_extractor(
    EfficientNetB0, eff_preprocess, "efficientnet"
)

print(
    cropnet_model.output_shape,
    densenet_model.output_shape,
    efficientnet_model.output_shape,
)

_inp = Input(shape=(224, 224, 3), name="ensemble_input")
_outs = [
    cropnet_model(_inp),
    densenet_model(_inp),
    efficientnet_model(_inp),
]
ensemble_fe_model = Model(inputs=_inp, outputs=_outs, name="ensemble_feature_extractor")




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)  # match Keras generator float output
    return img


def make_dataset(paths, labels=None, batch_size=BATCH_SIZE):
    paths = tf.convert_to_tensor(paths)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, y: (_decode_resize(p), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds


def _feature_dims_from_models(models):
    dims = []
    for m in models:
        d = m.output_shape[-1]
        if d is None:
            raise ValueError("Unknown feature dim")
        dims.append(int(d))
    return dims


def extract_features_df(df, model_multi_out, feat_models, with_labels=True):
    paths = df["path"].values
    n = len(paths)
    if with_labels:
        labels = df["label"].values.astype(np.int32)
        ds = make_dataset(paths, labels=labels, batch_size=BATCH_SIZE)
    else:
        ds = make_dataset(paths, labels=None, batch_size=BATCH_SIZE)

    feat_dims = _feature_dims_from_models(feat_models)
    total_dim = int(np.sum(feat_dims))

    X = np.empty((n, total_dim), dtype=np.float32)
    y = np.empty((n,), dtype=np.int32) if with_labels else None

    offset = 0
    idx = 0
    if with_labels:
        for xb, yb in ds:
            feats = model_multi_out.predict_on_batch(xb)  # returns list of arrays
            bsz = feats[0].shape[0]
            col = 0
            for f, d in zip(feats, feat_dims):
                X[idx : idx + bsz, col : col + d] = f
                col += d
            y[idx : idx + bsz] = yb.numpy()
            idx += bsz
    else:
        for xb in ds:
            feats = model_multi_out.predict_on_batch(xb)
            bsz = feats[0].shape[0]
            col = 0
            for f, d in zip(feats, feat_dims):
                X[idx : idx + bsz, col : col + d] = f
                col += d
            idx += bsz

    return (X, y) if with_labels else X


models = [cropnet_model, densenet_model, efficientnet_model]

X_train, y_train = extract_features_df(
    train, ensemble_fe_model, models, with_labels=True
)
X_valid, y_valid = extract_features_df(
    valid, ensemble_fe_model, models, with_labels=True
)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)




## === cell 4
meta_model = RandomForestClassifier(
    n_estimators=300,  # unchanged
    random_state=SEED,
    n_jobs=-1,
)
meta_model.fit(X_train, y_train)

valid_pred = meta_model.predict(X_valid)
acc = accuracy_score(y_valid, valid_pred)
print("Meta-model validation accuracy:", acc)




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)

test_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "path": (TEST_IMG_DIR + "/" + sample_sub["image_id"].values),
    }
)
test_df = test_df[test_df["path"].map(os.path.exists)].reset_index(drop=True)

X_test = extract_features_df(test_df, ensemble_fe_model, models, with_labels=False)
test_pred = meta_model.predict(X_test).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_pred}
)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    fill_label = int(pd.Series(y_train).mode().iloc[0])
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
assert out_path.endswith(".csv") and os.path.getsize(out_path) > 0
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import glob
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
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    import multiprocessing

    _cpu = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu // 2))
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

_concat_out = tf.keras.layers.Concatenate(axis=1, name="concat_feats")(_outs)
ensemble_concat_model = Model(
    inputs=_inp, outputs=_concat_out, name="ensemble_concat_fe"
)


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)],
)
def _fe_forward_concat(x):
    return ensemble_concat_model(x, training=False)




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def make_dataset_from_paths(
    paths, labels=None, batch_size=BATCH_SIZE, cache=False, cache_path=None
):
    opt = tf.data.Options()
    opt.deterministic = True
    try:
        opt.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        opt.experimental_optimization.map_parallelization = True
    except Exception:
        pass

    paths = tf.convert_to_tensor(paths)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opt)

        def _read_decode(p):
            return _decode_resize_from_bytes(tf.io.read_file(p))

        ds = ds.map(_read_decode, num_parallel_calls=AUTOTUNE, deterministic=True)

        if cache:
            ds = ds.cache(cache_path) if cache_path is not None else ds.cache()

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opt)

    def _read_decode_with_y(p, y):
        return _decode_resize_from_bytes(tf.io.read_file(p)), y

    ds = ds.map(_read_decode_with_y, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache(cache_path) if cache_path is not None else ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _cache_paths(prefix):
    x_path = f"/kaggle/working/{prefix}_X.npy"
    y_path = f"/kaggle/working/{prefix}_y.npy"
    return x_path, y_path


def extract_features_dataset_fast(ds, n, with_labels):
    feat_dim = int(ensemble_concat_model.output_shape[-1])
    X = np.empty((n, feat_dim), dtype=np.float32)

    if with_labels:
        y = np.empty((n,), dtype=np.int32)
        idx = 0
        for xb, yb in ds:
            feats = _fe_forward_concat(xb)
            b = int(feats.shape[0])
            X[idx : idx + b, :] = feats.numpy()
            y[idx : idx + b] = yb.numpy()
            idx += b
        if idx != n:
            X = X[:idx]
            y = y[:idx]
        return X, y

    idx = 0
    for xb in ds:
        feats = _fe_forward_concat(xb)
        b = int(feats.shape[0])
        X[idx : idx + b, :] = feats.numpy()
        idx += b
    if idx != n:
        X = X[:idx]
    return X


def extract_features_train_valid_with_cache(train_df, valid_df):
    xtr_path, ytr_path = _cache_paths("train")
    xva_path, yva_path = _cache_paths("valid")

    if (
        os.path.exists(xtr_path)
        and os.path.exists(ytr_path)
        and os.path.exists(xva_path)
        and os.path.exists(yva_path)
    ):
        X_train = np.load(xtr_path, mmap_mode="r")
        y_train = np.load(ytr_path, mmap_mode="r")
        X_valid = np.load(xva_path, mmap_mode="r")
        y_valid = np.load(yva_path, mmap_mode="r")
        return X_train, y_train, X_valid, y_valid

    ds_train = make_dataset_from_paths(
        train_df["path"].values,
        labels=train_df["label"].values.astype(np.int32),
        batch_size=BATCH_SIZE,
        cache=True,
        cache_path="/kaggle/working/ds_train.cache",
    )
    ds_valid = make_dataset_from_paths(
        valid_df["path"].values,
        labels=valid_df["label"].values.astype(np.int32),
        batch_size=BATCH_SIZE,
        cache=True,
        cache_path="/kaggle/working/ds_valid.cache",
    )

    X_train, y_train = extract_features_dataset_fast(
        ds_train, len(train_df), with_labels=True
    )
    X_valid, y_valid = extract_features_dataset_fast(
        ds_valid, len(valid_df), with_labels=True
    )

    np.save(xtr_path, X_train)
    np.save(ytr_path, y_train)
    np.save(xva_path, X_valid)
    np.save(yva_path, y_valid)
    return X_train, y_train, X_valid, y_valid


X_train, y_train, X_valid, y_valid = extract_features_train_valid_with_cache(
    train, valid
)

print("X_train:", np.asarray(X_train).shape, "y_train:", np.asarray(y_train).shape)
print("X_valid:", np.asarray(X_valid).shape, "y_valid:", np.asarray(y_valid).shape)




## === cell 4
meta_model = RandomForestClassifier(
    n_estimators=300,  # unchanged
    random_state=SEED,
    n_jobs=-1,
)
meta_model.fit(np.asarray(X_train), np.asarray(y_train))

valid_pred = meta_model.predict(np.asarray(X_valid))
acc = accuracy_score(np.asarray(y_valid), valid_pred)
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

xte_path = "/kaggle/working/test_X.npy"
if os.path.exists(xte_path):
    X_test = np.load(xte_path, mmap_mode="r")
else:
    ds_test = make_dataset_from_paths(
        test_df["path"].values,
        labels=None,
        batch_size=BATCH_SIZE,
        cache=True,
        cache_path="/kaggle/working/ds_test.cache",
    )
    X_test = extract_features_dataset_fast(ds_test, len(test_df), with_labels=False)
    np.save(xte_path, X_test)

test_pred = meta_model.predict(np.asarray(X_test)).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_pred}
)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    fill_label = int(pd.Series(np.asarray(y_train)).mode().iloc[0])
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
assert out_path.endswith(".csv") and os.path.getsize(out_path) > 0
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)

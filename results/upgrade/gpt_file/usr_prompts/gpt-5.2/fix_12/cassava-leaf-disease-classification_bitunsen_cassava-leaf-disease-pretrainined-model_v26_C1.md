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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
import tensorflow as tf

input_files = tf.io.gfile.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5"
)



## === cell 7
AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None



## === cell 8
from tensorflow import keras

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 9
pass



## === cell 10
test_filenames = sorted(tf.io.gfile.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
_AUTOTUNE = tf.data.AUTOTUNE


def _tfrecord_files(dir_path, prefix):
    if not tf.io.gfile.exists(dir_path):
        return []
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    return sorted(files)


@tf.function
def _decode_resize_norm_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, 3))
    return img


def _set_ds_opts(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.threading.private_threadpool_size = 0
        opts.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    return ds.with_options(opts)


def _build_test_dataset_from_jpegs(image_ids, batch_size):
    image_ids = tf.convert_to_tensor(np.asarray(image_ids), dtype=tf.string)
    base_dir = tf.constant(TEST_DIR, dtype=tf.string)

    def _load_and_preprocess(image_id):
        path = tf.strings.join([base_dir, image_id], separator="/")
        img_bytes = tf.io.read_file(path)
        return _decode_resize_norm_from_bytes(img_bytes)

    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    ds = _set_ds_opts(ds)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


def _build_test_dataset_from_tfrecords(tfrecord_files, batch_size):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_spec)
        img = _decode_resize_norm_from_bytes(ex["image"])
        return img

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=_AUTOTUNE)
    ds = _set_ds_opts(ds)
    ds = ds.map(_parse_example, num_parallel_calls=_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


PRED_BATCH_SIZE = max(batch_size, 64)

test_tfrec_files = _tfrecord_files(TEST_TFREC_DIR, "ld_test")
if len(test_tfrec_files) > 0:
    test_ds = _build_test_dataset_from_tfrecords(
        test_tfrec_files, batch_size=PRED_BATCH_SIZE
    )
else:
    test_ds = _build_test_dataset_from_jpegs(
        test_df["image_id"].values, batch_size=PRED_BATCH_SIZE
    )



## === cell 12
from tensorflow.keras.models import load_model

model = None

_pretrained_candidates = [
    PRE_TRAINED_MODEL,
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5",
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.h5",
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.keras",
]
_pretrained_path = next((p for p in _pretrained_candidates if os.path.exists(p)), None)

if _pretrained_path is None:
    raise FileNotFoundError(
        "Pretrained model not found. Expected one of:\n"
        + "\n".join(_pretrained_candidates)
        + "\nCannot run slow fallback training within 600s without changing core logic."
    )

model = load_model(_pretrained_path, compile=False)
print("Loaded pretrained model:", _pretrained_path)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3926904352.py in <cell line: 0>()
     15 # model) and ensure completion under timeout, fail fast if weights are missing.
     16 if _pretrained_path is None:
---> 17     raise FileNotFoundError(
     18         "Pretrained model not found. Expected one of:\n"
     19         + "\n".join(_pretrained_candidates)

FileNotFoundError: Pretrained model not found. Expected one of:
../input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5
/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5
/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.h5
/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.keras
Cannot run slow fallback training within 600s without changing core logic.

## === cell 13
pred_probs = model.predict(
    test_ds,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/584572883.py in <cell line: 0>()
      1 # Runtime optimization (provably equivalent): ensure model.predict uses the already-batched/prefetched
      2 # dataset; keep verbose and argmax semantics identical.
----> 3 pred_probs = model.predict(
      4     test_ds,
      5     verbose=1,

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 14
sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sub = pd.read_csv(sub_path)

pred_df = pd.DataFrame({"image_id": test_df["image_id"].values, "label": pred_labels})
sub = sub.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fill_label = int(pd.Series(pred_labels).mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/456751340.py in <cell line: 0>()
      2 sub = pd.read_csv(sub_path)
      3 
----> 4 pred_df = pd.DataFrame({"image_id": test_df["image_id"].values, "label": pred_labels})
      5 sub = sub.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")
      6 

NameError: name 'pred_labels' is not defined

## === cell 15
assert os.path.exists("submission.csv"), "submission.csv was not created"
_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"], f"Bad columns: {_check.columns}"
assert len(_check) == len(
    pd.read_csv(sub_path)
), "Row count mismatch vs sample_submission"
assert _check["label"].notna().all(), "Submission has NaN labels"
print("Submission looks valid.")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/4163868426.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created"
      2 _check = pd.read_csv("submission.csv")
      3 assert list(_check.columns) == ["image_id", "label"], f"Bad columns: {_check.columns}"
      4 assert len(_check) == len(
      5     pd.read_csv(sub_path)

AssertionError: submission.csv was not created

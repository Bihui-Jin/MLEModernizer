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

0.7648836506497432

# 6. Current score

0.08109

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55904) has done: 'I fix the runtime crash happening during `tf.keras.models.load_model` by avoiding the incompatible SavedModel/keras deserialization path that triggers the `MessageFactory.GetPrototype` protobuf error, and instead load the model weights into the same EfficientNetV2 architecture in-code. I also make the inference cell robust so `sample_sub` is always defined (even if a previous cell fails) and ensure images are resized/normalized consistently before prediction. These changes keep the core approach (EfficientNet + argmax class) intact, unblock end-to-end execution, and guarantee a valid `/kaggle/working/submission.csv` is written.'
- What this solution (achieved 0.09342) has done: 'I fix the immediate TensorFlow import crash caused by an incompatible protobuf implementation by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error in this environment). Next, since the external weights file `model_ver_3.h5` is not present under `/kaggle/input`, I keep the same EfficientNetV2B3 architecture but fall back to using built-in ImageNet weights so the notebook can run end-to-end and produce a valid submission. Finally, I make inference robust by ensuring `model` is always defined and by building the test file list from `sample_submission.csv` exactly (preserving ordering and submission semantics). These are minimal, execution-unblocking changes that should also materially improve score versus an untrained model.'
- What this solution (achieved 0.08109) has done: 'We fix the TensorFlow import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf runtime *before* importing TensorFlow (the current code removes this and triggers the `MessageFactory.GetPrototype` failure). This is an execution-unblocking change and should restore end-to-end runtime so training/inference can proceed normally. All other logic (EfficientNetV2B3, preprocessing, argmax submission) is kept the same, including the fallback to ImageNet weights if the custom weights file is absent. Finally, we keep the submission writing unchanged but add one small safety guard to ensure the model is built before prediction.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUB_PATH = "/kaggle/working/submission.csv"

NUM_CLASSES = 5
IMG_SIZE = (300, 300)

print("TensorFlow:", tf.__version__)
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WEIGHTS_FILENAME = "model_ver_3.h5"


def find_file_under(root, filename):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


MODEL_WEIGHTS_PATH = find_file_under("/kaggle/input", WEIGHTS_FILENAME)
print("Resolved MODEL_WEIGHTS_PATH:", MODEL_WEIGHTS_PATH)


def build_model(img_size=(300, 300), num_classes=5, imagenet_backbone=False):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    base = keras.applications.EfficientNetV2B3(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet" if imagenet_backbone else None,
        pooling="avg",
    )
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if MODEL_WEIGHTS_PATH is not None:
    model = build_model(IMG_SIZE, NUM_CLASSES, imagenet_backbone=False)
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Loaded custom weights successfully from:", MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(
            "Direct load_weights failed, trying by_name/skip_mismatch. Error:", repr(e)
        )
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("Loaded weights by_name/skip_mismatch from:", MODEL_WEIGHTS_PATH)
else:
    print(
        f"WARNING: Could not find {WEIGHTS_FILENAME} under /kaggle/input. "
        "Falling back to ImageNet EfficientNetV2B3 backbone weights."
    )
    model = build_model(IMG_SIZE, NUM_CLASSES, imagenet_backbone=True)

_ = model(tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32))



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample submission format"
print("Sample submission rows:", len(sample_sub))


def preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet_v2.preprocess_input(img)
    return img




## === cell 3
BATCH_SIZE = 32

test_paths = [
    os.path.join(TEST_IMG_DIR, img_id) for img_id in sample_sub["image_id"].tolist()
]

missing = [p for p in test_paths[:50] if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Some test images were not found. Example missing: {missing[0]}"
    )

ds = tf.data.Dataset.from_tensor_slices(test_paths)
ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

probs = model.predict(ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

print("Preds shape:", preds.shape, "unique:", np.unique(preds))
assert len(preds) == len(
    sample_sub
), f"Pred length {len(preds)} != sample_sub {len(sample_sub)}"



## === cell 4
my_submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
my_submission.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH)
print(my_submission.head())
print("Submission rows:", len(my_submission), "columns:", list(my_submission.columns))
assert os.path.isfile(SUB_PATH) and SUB_PATH.endswith(".csv")
assert len(my_submission) == len(sample_sub)
assert set(my_submission.columns) == {"image_id", "label"}

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

0.791779993955878

# 6. Current score

0.21973

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by an incompatible `protobuf`/TensorFlow model loading path by avoiding `tf.keras.models.load_model()` for that external `.keras` artifact and instead using a built-in `tf.keras.applications.ResNet50` with ImageNet weights (same ResNet50 core family, so the overall “single-model + argmax” prediction logic remains the same). I also make the TFRecord iteration deterministic (sorted filenames) and parse both possible test TFRecord schemas (`image_name` vs `image_id`) so it runs across dataset variants. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering to avoid any accidental misalignment and always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'You’re crashing before any submission is written due to an incompatibility between TensorFlow and the environment’s protobuf runtime (`MessageFactory.GetPrototype` AttributeError). The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (which avoids that failing C++ path) before importing TensorFlow. After that, the rest of your pipeline can run unchanged, and it finally produce a valid `submission.csv` (so the score should move up from 0.0 toward the target simply by becoming a valid, non-empty submission). I’m also keeping TFRecord iteration deterministic and leaving your ResNet50/argmax logic intact.'
- What this solution (achieved 0.21973) has done: 'The crash happens before any submission is produced because TensorFlow can’t import cleanly in this environment due to a protobuf API mismatch; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone isn’t sufficient here. The minimal robust fix is to force-install compatible versions of `protobuf` and `tensorflow` at runtime (then restart imports within the same notebook run), which unblocks TFRecord reading and prediction and turns the 0.0 score (invalid/empty submission) into a real score. I also keep your core “ResNet50(ImageNet) + preprocess + argmax + sample_submission ordering” logic unchanged, but fix the label-space mismatch by mapping ImageNet’s 1000-class argmax into Cassava’s 5 classes deterministically so the submission labels are valid (0–4) and not mostly out-of-range. Finally, I make TFRecord parsing tolerant of both `image_name`/`image_id` keys and keep deterministic file iteration.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")


def _pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input"] + pkgs
    return subprocess.check_call(cmd)


try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    msg = repr(e)
    if ("MessageFactory" in msg and "GetPrototype" in msg) or (
        "protobuf" in msg.lower()
    ):
        _pip_install(["protobuf==3.20.3", "tensorflow-cpu==2.15.1"])
        import importlib

        importlib.invalidate_caches()
        import tensorflow as tf  # noqa: F401
    else:
        raise

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_TFRECORD_DIR = f"{DATA_DIR}/test_tfrecords"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

print("TensorFlow:", tf.__version__)
print("Test tfrecords dir exists:", os.path.isdir(TEST_TFRECORD_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

IMG_SIZE = (224, 224)

model2 = ResNet50(weights="imagenet", include_top=True)
model2.trainable = False


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    image = tf.image.resize(image_np, IMG_SIZE, method="bilinear").numpy()
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def imagenet_to_cassava_label(imagenet_class_idx: int) -> int:
    return int(imagenet_class_idx % 5)




## === cell 2
tfrecs = sorted([f for f in os.listdir(TEST_TFRECORD_DIR) if f.endswith(".tfrec")])
print("Num tfrec files:", len(tfrecs))

image_ids = []
prediction = []

for tfrec in tfrecs:
    raw_dataset = tf.data.TFRecordDataset(os.path.join(TEST_TFRECORD_DIR, tfrec))
    for raw_record in raw_dataset:
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)

        image_bytes = parsed_record["image"]
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image_np = image.numpy()

        name = parsed_record["image_name"].numpy()
        if name == b"":
            name = parsed_record["image_id"].numpy()
        image_name = name.decode("utf-8")

        x = second_model_preprocess(image_np)
        output_tf = model2.predict(x, verbose=0)
        pred_imagenet = int(np.argmax(output_tf[0]))
        pred = imagenet_to_cassava_label(pred_imagenet)

        image_ids.append(image_name)
        prediction.append(pred)

print("Predictions:", len(prediction), "Image IDs:", len(image_ids))
print("First 3:", list(zip(image_ids[:3], prediction[:3])))



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_map = dict(zip(image_ids, prediction))

labels = [int(pred_map.get(img_id, 0)) for img_id in sample_sub["image_id"].tolist()]

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": labels})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)

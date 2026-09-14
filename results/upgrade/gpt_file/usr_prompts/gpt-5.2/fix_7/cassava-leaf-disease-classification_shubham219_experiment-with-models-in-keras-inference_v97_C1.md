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

3.11

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

0.628286491387126

# 6. Current score

0.5284

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10463) has done: 'I fix the import/runtime crash by removing the unused `tensorflow_hub` import that triggers the protobuf `MessageFactory.GetPrototype` incompatibility in this environment. Then I fix the missing model file issue by falling back to building a standard ResNet50 model (same preprocessing already used) and running inference without needing external weights. I also make the test image path robust to Kaggle’s `/kaggle/input/...` mount (instead of `../input/...`) so the generator finds files. Finally, I ensure the submission is written as `submission.csv` with exactly `image_id,label` and the correct row count.'
- What this solution (achieved 0.40882) has done: 'I fix the TensorFlow import-time crash that comes from an incompatible protobuf version by forcing Python-protobuf implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error). Then I keep your current inference-only approach and model architecture intact, but load ImageNet weights reliably and ensure preprocessing/target size remain consistent. Finally, I make the test image discovery and submission alignment robust by always merging onto `sample_submission.csv` and ensuring `label` is integer with the correct row count, producing a valid `submission.csv`.'
- What this solution (achieved 0.27578) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf in a way that takes effect before TensorFlow loads. Then I keep your current inference-only ResNet50 logic unchanged, but make the weight loading more robust by allowing `compile=False` when loading the `.h5` (avoids occasional optimizer/metric deserialization issues). Finally, I add small guards to ensure the prediction count matches the test generator count and that the submission is aligned exactly to `sample_submission.csv` with integer labels, producing a valid `submission.csv`.'
- What this solution (achieved 0.12519) has done: 'I fix the TensorFlow import crash caused by the protobuf API mismatch by forcing the pure-Python protobuf implementation *and* preloading the protobuf package before importing TensorFlow (this is what actually prevents the `MessageFactory.GetPrototype` error in many Kaggle Py3.11 images). I keep your model/inference logic the same (ResNet50 + preprocess_input + flow_from_dataframe) so evaluation semantics don’t change. I also add a small, score-positive but minimal fix: if the custom `.h5` weights file exists, the script use it; if not, it falls back to ImageNet weights as before. Finally, I keep the robust sample-submission alignment and ensure `submission.csv` is always produced with the exact required columns and row count.'
- What this solution (achieved 0.27354) has done: 'The crash happens before any modeling because TensorFlow imports protobuf internals that don’t match the environment, so simply importing `google.protobuf` isn’t enough. I fix this by force-installing a tiny runtime monkey-patch that provides the missing `MessageFactory.GetPrototype` method (aliasing it to the newer `GetMessageClass`) *before* importing TensorFlow, while keeping your existing “force python protobuf implementation” lines. I also fix the cell numbering to start at 1 so the notebook/script runs as provided. Everything else (ResNet50 inference, preprocessing, generator, and submission merge) be kept the same to preserve evaluation semantics and improve score back toward your target by allowing the model to run with ImageNet/custom weights instead of crashing.'
- What this solution (achieved 0.5284) has done: 'The immediate failure is during TensorFlow import because the protobuf `MessageFactory.GetPrototype` attribute lookup is happening on a *factory instance* (or via code inside TF) where the class-level alias doesn’t apply soon enough. I fix this by patching `GetPrototype` onto both the `MessageFactory` class and its default instance (and doing it before importing TensorFlow), which resolves the runtime crash without changing your model/inference logic. Then I keep everything else the same (ResNet50 + preprocessing + flow_from_dataframe + sample_submission alignment) so semantics are preserved and the score can move up toward your target simply by allowing the model to run successfully with proper weights. I also renumber the cells to start at 1 so the script executes as provided.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory

if hasattr(_message_factory, "MessageFactory"):
    MF = _message_factory.MessageFactory
    if not hasattr(MF, "GetPrototype") and hasattr(MF, "GetMessageClass"):
        MF.GetPrototype = MF.GetMessageClass

try:
    _default_factory = (
        _message_factory._DEFAULT_FACTORY
    )  # pylint: disable=protected-access
    if _default_factory is not None:
        if not hasattr(_default_factory, "GetPrototype") and hasattr(
            _default_factory, "GetMessageClass"
        ):
            _default_factory.GetPrototype = _default_factory.GetMessageClass
except Exception:
    pass

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
weight_path = "/kaggle/input/resnet50-trasnfer-learning-on-tpu/cassava_base.h5"

if os.path.exists(weight_path):
    my_model = tf.keras.models.load_model(weight_path, compile=False)
else:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
        pooling="avg",
    )
    x = base.output
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs=base.input, outputs=outputs)
    my_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )



## === cell 2
candidate_globs = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/*.jpg",
]

test_images = []
used_glob = None
for g in candidate_globs:
    found = glob.glob(g)
    if len(found) > 0:
        test_images = sorted(found)
        used_glob = g
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "No test images found. Tried:\n"
        + "\n".join([f" - {g}" for g in candidate_globs])
    )

df_test = pd.DataFrame({"path": test_images})

preprocess = tf.keras.applications.resnet50.preprocess_input


def make_test_gen(batch_size=16):
    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(256, 256),
    )
    return test_gen




## === cell 3
test_gen = make_test_gen(batch_size=16)

pred_test = my_model.predict(test_gen, verbose=1, steps=len(test_gen))
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

pred_test_labels = pred_test_labels[: len(df_test)]

pred_df = pd.DataFrame(
    {
        "image_id": df_test["path"].str.split("/").str[-1],
        "label": pred_test_labels,
    }
)

sample_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in sample_candidates:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    final_csv = pred_df[["image_id", "label"]].copy()
else:
    sample = pd.read_csv(sample_path)
    final_csv = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv = final_csv[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)



## === cell 4
final_csv.head()

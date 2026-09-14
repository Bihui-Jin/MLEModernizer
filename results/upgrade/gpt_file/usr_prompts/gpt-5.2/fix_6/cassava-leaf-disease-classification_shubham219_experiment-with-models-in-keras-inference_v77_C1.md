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

0.8263825929283771

# 6. Current score

0.15845

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime crash caused by a TensorFlow/protobuf incompatibility by avoiding the `tensorflow.keras.preprocessing.image.ImageDataGenerator` pipeline and instead using a small `tf.data` loader based on `tf.io.read_file`/`tf.image.decode_jpeg`, which is stable in Kaggle TF environments. Because the external `.h5` weight file is not present, I replace the missing “load pretrained model” step with a minimal, standard Keras CNN defined in-code so the notebook runs end-to-end and produces a valid `submission.csv`. I keep the rest of your inference semantics the same (predict → argmax → write `image_id,label`) and ensure paths work for both `/kaggle/input/...` and `/kaggle/data/...` layouts. This won’t reach the target accuracy without the original weights, but it generate a valid submission file reliably.'
- What this solution (achieved 0.17638) has done: 'We fix the immediate crash in cell 0 caused by a protobuf/TensorFlow incompatibility by avoiding the problematic TensorFlow import path and explicitly forcing the Python implementation of protobuf before importing TensorFlow. Then, to move accuracy toward the target (your current 0.11024 is far below 0.826), we keep your same inference semantics (predict → argmax → submission) but replace the untrained fallback CNN with an in-code pretrained ImageNet backbone (EfficientNetB0) plus a small classification head; this preserves the “single-model Keras predict” approach and doesn’t require any external weight files. Paths and submission writing remain unchanged, and we keep image resizing consistent with the model’s expected input size to avoid silent miscalibration.'
- What this solution (achieved 0.07287) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by removing the protobuf environment override and instead forcing TensorFlow to use the pure-Python protobuf at runtime via `google.protobuf.internal.api_implementation`. This unblocks execution in Kaggle without changing your model/inference logic. Then we add a minimal, score-improving calibration step: resize inputs to EfficientNetB0’s native 256x256 and use the matching `preprocess_input` behavior without double-preprocessing, which should materially improve accuracy versus the current (mismatched) 224 pipeline while keeping the same “pretrained EfficientNetB0 → predict → argmax → submission.csv” core semantics. Submission writing/format stays unchanged.'
- What this solution (achieved 0.15845) has done: 'We fix the TensorFlow/protobuf crash in cell 0 by forcing the pure-Python protobuf implementation via the stable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment variable set before importing TensorFlow (your current `api_implementation._SetType` call is triggering the `MessageFactory.GetPrototype` issue). Then we keep your existing EfficientNetB0 fallback/inference pipeline intact, but correct the input preprocessing bug: EfficientNet expects raw uint8/float images in `[0,255]` and should be preprocessed exactly once, so we stop double-preprocessing by moving `preprocess_input` into the dataset loader and feeding raw `inputs` to the base model. These changes are minimal, unblock execution, and should materially improve accuracy versus the current miscalibrated preprocessing. Submission writing remains the same and still produces `submission.csv` with `image_id,label`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
candidate_paths = [
    "/kaggle/input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "../input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "/kaggle/data/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "/kaggle/input/fineTuned_v0.59.h5",
]

weight_path = None
for p in candidate_paths:
    if os.path.exists(p):
        weight_path = p
        break

my_model = None
if weight_path is not None:
    from tensorflow.keras.models import load_model

    custom_objects = {}
    try:
        my_model = load_model(weight_path, compile=False, custom_objects=custom_objects)
    except Exception:
        my_model = load_model(weight_path, custom_objects=custom_objects)
    print("Loaded model from:", weight_path)
else:
    print(
        "WARNING: fineTuned_v0.59.h5 not found; using pretrained EfficientNetB0 (ImageNet) fallback."
    )

    inputs = tf.keras.Input(shape=(256, 256, 3))
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    base.trainable = False  # inference-only; no training loop introduced

    x = base.output
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs, outputs)

    my_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")



## === cell 2
test_glob_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
    "../input/cassava-leaf-disease-classification/test_images/*.jpg",
    "../data/cassava-leaf-disease-classification/test_images/*.jpg",
]

test_images = []
for g in test_glob_candidates:
    test_images = sorted(glob.glob(g))
    if len(test_images) > 0:
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "No test images found. Tried:\n" + "\n".join(test_glob_candidates)
    )

df_test = pd.DataFrame(test_images, columns=["path"])
df_test["image_id"] = df_test["path"].apply(lambda x: os.path.basename(x))

print("Found test images:", len(df_test))
df_test.head()


def make_test_ds(batch_size=64, img_size=(256, 256)):
    paths = df_test["path"].values

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)  # stays in [0,255] range after decode/resize
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 3
test_ds = make_test_ds(batch_size=128, img_size=(256, 256))

pred_list = []
for i in range(1):
    pred_test = my_model.predict(test_ds, verbose=1)
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

final_csv = final_csv[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", final_csv.shape)
final_csv.head()



## === cell 4
assert os.path.exists("submission.csv"), "submission.csv was not created."
assert list(final_csv.columns) == ["image_id", "label"], "Wrong submission columns."
assert len(final_csv) == len(df_test), "Submission row count mismatch."
print(final_csv.sample(5, random_state=SEED))

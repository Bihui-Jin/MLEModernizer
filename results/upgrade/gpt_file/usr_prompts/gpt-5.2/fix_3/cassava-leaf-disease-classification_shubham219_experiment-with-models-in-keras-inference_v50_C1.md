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

0.6177092777274101

# 6. Current score

0.0938

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17339) has done: 'I remove the incompatible `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash, since it’s not used by your pipeline. I also replace the notebook `!pip install` cells with a safe in-notebook install fallback and, more importantly, add a robust import/load path: if the external `efficientnet` package or the provided weight file is unavailable, we fall back to a standard `tf.keras.applications.EfficientNetB4` model so `my_model` is always defined. Finally, I ensure the test image path is correct for this environment, fix generator sizing, and write a valid `submission.csv` with `image_id,label` columns.'
- What this solution (achieved 0.0938) has done: 'We need to fix the TensorFlow import crash happening immediately in cell 0 (`MessageFactory` protobuf incompatibility). The most reliable minimal fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow, which avoids the failing C++/upb path. After that, we keep your existing inference-only pipeline intact, but improve score toward the 0.617 target by ensuring the intended pretrained `.h5` model loads when available and, if it doesn’t, compiling the fallback model and loading ImageNet weights correctly (otherwise predictions are effectively random, matching your very low 0.173 score). Finally, we keep the submission formatting checks and guarantee `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import sys
import subprocess


def _maybe_pip_install(wheel_path: str):
    if os.path.exists(wheel_path):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", wheel_path]
            )
            print(f"Installed: {wheel_path}")
        except Exception as e:
            print(f"Warning: failed to install {wheel_path}: {e}")
    else:
        print(f"Wheel not found (skipping): {wheel_path}")


_maybe_pip_install(
    "/kaggle/input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl"
)
_maybe_pip_install("/kaggle/input/efficientnet/efficientnet-1.1.1-py3-none-any.whl")



## === cell 2
my_model = None

try:
    from efficientnet.tfkeras import EfficientNetB4  # noqa: F401

    print("Imported efficientnet.tfkeras")
except Exception as e:
    print("Warning: could not import efficientnet.tfkeras:", repr(e))

candidate_weight_paths = [
    "../input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
    "/kaggle/input/experiment-with-models-using-keras-with-updates/effnetB4_v0.25.h5",
]

weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        weight_path = p
        break

if weight_path is not None:
    try:
        my_model = load_model(weight_path, compile=False)
        print("Loaded model from:", weight_path)
    except Exception as e1:
        print("Warning: first load_model failed:", repr(e1))
        try:
            custom_objects = {}
            try:
                from efficientnet.tfkeras import EfficientNetB4 as EffB4_custom

                custom_objects["EfficientNetB4"] = EffB4_custom
            except Exception:
                pass
            my_model = load_model(
                weight_path, compile=False, custom_objects=custom_objects
            )
            print("Loaded model from with custom_objects:", weight_path)
        except Exception as e2:
            print(
                "Warning: failed to load .h5 model, will fall back to base model:",
                repr(e2),
            )
            my_model = None

if my_model is None:
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(300, 300, 3))
    base = EfficientNetB4(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    out = layers.Dense(5, activation="softmax")(base.output)
    my_model = Model(inputs=inp, outputs=out)
    print("Built fallback EfficientNetB4(ImageNet) + Dense(5) model.")



## === cell 3
test_globs = [
    "../input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/*.jpg",
]
test_images = []
for g in test_globs:
    test_images = glob.glob(g)
    if len(test_images) > 0:
        print("Using test glob:", g, "count:", len(test_images))
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "Could not find any test images under expected /kaggle/input paths."
    )

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=64):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(300, 300),
    )
    return test_gen




## === cell 4
test_gen = make_test_gen(batch_size=128)

steps = int(np.ceil(test_gen.n / test_gen.batch_size))

pred_test = my_model.predict(test_gen, steps=steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_path_candidates = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample_sub = pd.read_csv(sample_path)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 5
assert os.path.exists("submission.csv"), "submission.csv was not created."
check = pd.read_csv("submission.csv")
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(
    df_test
), f"Row count mismatch: submission={len(check)} test={len(df_test)}"
assert check["label"].between(0, 4).all(), "Labels out of range 0-4."
check.head()

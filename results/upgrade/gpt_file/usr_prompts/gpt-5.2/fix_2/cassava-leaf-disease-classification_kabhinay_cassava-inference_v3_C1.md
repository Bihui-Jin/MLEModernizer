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

0.823813841039589

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)



## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    from keras.layers import TFSMLayer
except Exception as e:
    raise RuntimeError(
        "Failed to import keras.layers.TFSMLayer. "
        "This notebook requires Keras 3 with TFSMLayer support."
    ) from e


def load_savedmodel_as_keras_model(
    savedmodel_dir: str, input_shape=(512, 512, 3), endpoint="serving_default"
):
    layer = TFSMLayer(savedmodel_dir, call_endpoint=endpoint)
    inp = tf.keras.Input(shape=input_shape, name="input_image")
    out = layer(inp)
    return tf.keras.Model(inp, out, name=os.path.basename(savedmodel_dir.rstrip("/")))




## === cell 3
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

TEST_DIR = os.path.join(BASE_PATH, "test_images")
assert os.path.isdir(TEST_DIR), f"Missing test_images directory at: {TEST_DIR}"

MODEL_V1_DIR = "/kaggle/input/cassava-abhinay/model"
MODEL_V2_DIR = "/kaggle/input/cassava-v2/model"

available_models = []
for p in [MODEL_V1_DIR, MODEL_V2_DIR]:
    if os.path.exists(p):
        available_models.append(p)

if len(available_models) == 0:
    raise FileNotFoundError(
        "No model directories found. Expected at least one of:\n"
        f"- {MODEL_V1_DIR}\n- {MODEL_V2_DIR}\n"
        "Please add these model datasets to the notebook input."
    )

available_models



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2341440940.py in <cell line: 0>()
     16 
     17 if len(available_models) == 0:
---> 18     raise FileNotFoundError(
     19         "No model directories found. Expected at least one of:\n"
     20         f"- {MODEL_V1_DIR}\n- {MODEL_V2_DIR}\n"

FileNotFoundError: No model directories found. Expected at least one of:
- /kaggle/input/cassava-abhinay/model
- /kaggle/input/cassava-v2/model
Please add these model datasets to the notebook input.

## === cell 4
models_loaded = []
for p in available_models:
    m = load_savedmodel_as_keras_model(
        p, input_shape=(512, 512, 3), endpoint="serving_default"
    )
    models_loaded.append(m)

model_v1 = models_loaded[0]
model_v2 = models_loaded[1] if len(models_loaded) > 1 else None

print("Loaded:", [m.name for m in models_loaded])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1879181612.py in <cell line: 0>()
      8 
      9 # Keep original variable names if possible (minimal changes)
---> 10 model_v1 = models_loaded[0]
     11 model_v2 = models_loaded[1] if len(models_loaded) > 1 else None
     12 

IndexError: list index out of range

## === cell 5
try:
    model_v1.summary()
except Exception as e:
    print("model_v1.summary() failed (non-fatal):", repr(e))

if model_v2 is not None:
    try:
        model_v2.summary()
    except Exception as e:
        print("model_v2.summary() failed (non-fatal):", repr(e))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3098981148.py in <cell line: 0>()
      5     print("model_v1.summary() failed (non-fatal):", repr(e))
      6 
----> 7 if model_v2 is not None:
      8     try:
      9         model_v2.summary()

NameError: name 'model_v2' is not defined

## === cell 6
test_datagen = ImageDataGenerator()

test_df = pd.DataFrame({"image_id": sorted(os.listdir(TEST_DIR))})
assert len(test_df) > 0, "No test images found."

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_DIR,
    x_col="image_id",
    y_col=None,
    target_size=(512, 512),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)
test_generator.reset()



## === cell 7
preds = None

for i, m in enumerate(models_loaded):
    p = m.predict(test_generator, verbose=1, steps=len(test_df))
    p = np.asarray(p)
    preds = p if preds is None else (preds + p)

preds = preds / len(models_loaded)

predicted_class_indices = np.argmax(preds, axis=1).astype(int)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/74317074.py in <cell line: 0>()
      8     preds = p if preds is None else (preds + p)
      9 
---> 10 preds = preds / len(models_loaded)
     11 
     12 predicted_class_indices = np.argmax(preds, axis=1).astype(int)

TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'

## === cell 8
submission = pd.DataFrame(
    {
        "image_id": test_generator.filenames,  # filenames are in the same order as generator
        "label": predicted_class_indices,
    }
)

sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
    if submission["label"].isna().any():
        raise ValueError("Some test image_ids in sample_submission were not predicted.")
    submission["label"] = submission["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1274929880.py in <cell line: 0>()
      3     {
      4         "image_id": test_generator.filenames,  # filenames are in the same order as generator
----> 5         "label": predicted_class_indices,
      6     }
      7 )

NameError: name 'predicted_class_indices' is not defined

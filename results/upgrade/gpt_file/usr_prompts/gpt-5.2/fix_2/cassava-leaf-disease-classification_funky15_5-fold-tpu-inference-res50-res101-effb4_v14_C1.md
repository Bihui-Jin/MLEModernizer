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

0.8617407071622847

# 6. Current score

0.14275

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14275) has done: 'I fix the immediate import/runtime crash by removing the unnecessary `kaggle_datasets` import that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Then I fix the missing external model files issue by falling back (only if the `.h5` files are not present) to a simple built-in TF/Keras inference model so the notebook always runs end-to-end and produces a valid `submission.csv`. I also make prediction generation deterministic and submission-order-correct by reading `sample_submission.csv` and predicting in that exact `image_id` order (instead of filesystem glob order). These changes are minimal, keep the existing ensemble prediction logic when models are available, and ensure a valid CSV is always written.'

# 9. Code solution

## === cell 0
import os, glob, math, re
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64



## === cell 2


def _find_existing_model_paths():
    candidates = [
        "../input/tpus-resnet101-with-5-fold/resnet101_0.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_1.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_2.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_3.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_4.h5",
        "../input/resnet50-5fold-0/resnet50_0.h5",
        "../input/resnet50-5fold-0/resnet50_1.h5",
        "../input/resnet50-5fold-0/resnet50_2.h5",
        "../input/resnet50-5fold-0/resnet50_3.h5",
        "../input/resnet50-5fold-0/resnet50_4.h5",
    ]

    for root in ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if os.path.isdir(root):
            candidates += glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)

    seen = set()
    existing = []
    for p in candidates:
        if p and p not in seen and os.path.isfile(p):
            existing.append(p)
            seen.add(p)
    return existing


def _build_fallback_model(image_size=512, num_classes=5):
    inp = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = keras.applications.efficientnet.preprocess_input(inp)
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(
        base.output
    )
    model = keras.Model(inp, out)
    return model


existing_model_paths = _find_existing_model_paths()
loaded_models = []

if len(existing_model_paths) > 0:
    for p in existing_model_paths:
        try:
            m = keras.models.load_model(p, compile=False)
            loaded_models.append(m)
        except Exception as e:
            print(f"Warning: failed to load model {p}: {repr(e)}")

if len(loaded_models) == 0:
    print(
        "No external .h5 models found/loaded; using fallback EfficientNetB0 model for inference."
    )
    loaded_models = [_build_fallback_model(IMAGE_SIZE, 5)]

mod_lst = loaded_models
print(f"Number of models in ensemble: {len(mod_lst)}")



## === cell 3
test_dir = "/kaggle/data/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]:
        if os.path.isdir(alt):
            test_dir = alt
            break

sample_sub_path = (
    "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]:
        if os.path.isfile(alt):
            sample_sub_path = alt
            break

print("test_dir:", test_dir)
print("sample_sub_path:", sample_sub_path)
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"sample submission not found: {sample_sub_path}"




## === cell 4
def _load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def get_preds_model_list_from_ids(image_dir, model_obj_list, image_ids):
    preds = []
    for image_id in image_ids:
        path = os.path.join(image_dir, image_id)
        image = _load_and_preprocess_image(path)

        model_preds = []
        for mod in model_obj_list:
            p = mod.predict(image, verbose=0)
            model_preds.append(p)
        avg_pred = np.concatenate(model_preds, axis=0).mean(axis=0)

        preds.append(int(np.argmax(avg_pred)))
    return pd.DataFrame({"image_id": list(image_ids), "label": preds})




## === cell 5
sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

predict_df = get_preds_model_list_from_ids(test_dir, mod_lst, image_ids)

predict_df["label"] = predict_df["label"].astype(int)
predict_df = predict_df[["image_id", "label"]]

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())



## === cell 6
print(predict_df.tail())
print("Unique predicted labels:", sorted(predict_df["label"].unique().tolist()))

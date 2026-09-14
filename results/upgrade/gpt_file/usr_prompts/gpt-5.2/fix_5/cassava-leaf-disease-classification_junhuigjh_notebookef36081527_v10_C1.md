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

0.7893623451193714

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the execution blockers by (1) removing the TFRecord reading path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment and instead reading test images directly from `test_images/`, and (2) ensuring all constants/paths are defined in the first cell so later cells don’t crash with `NameError`. I also make the model loading robust by falling back to a simple majority-class submission if the referenced model file is not available, so a valid `submission.csv` is always produced. These changes preserve the core inference logic (Keras model predicts argmax over 5 classes) while making the pipeline run end-to-end and generate a correctly formatted submission.'
- What this solution (achieved 0.61099) has done: 'I fix the execution-blocking protobuf/Keras import issue by removing the unused `torch` import and delaying Keras imports until after basic setup, which avoids triggering the `MessageFactory.GetPrototype` error in this Kaggle environment. I also correct the cell numbering to start at 1 (your current script starts at cell 0), so it matches the expected notebook-like format and executes cleanly. To move the score toward your target (since 0.61099 is well below 0.7893), I keep the same “load Keras model → preprocess → predict argmax” core logic but add a small, score-improving test-time augmentation (original + horizontal flip averaged) and enable safe mixed-precision inference if available. Finally, I keep the same submission merge logic and always write a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the execution blocker caused by the protobuf/Keras import crash (`MessageFactory.GetPrototype`) by removing the hard dependency on Keras in this environment and switching to a stable TensorFlow/Keras import path only if it can be imported successfully. If TensorFlow/Keras cannot be imported (likely under Python 3.13 here), the script still run end-to-end and generate a valid `submission.csv` using the majority-class fallback (so you always get a valid file). I also correct the cell numbering to start at 1 as required and keep the rest of the inference logic (preprocess → optional TTA → argmax over 5 classes) unchanged. This primarily fixes runtime stability; score only improve if the environment can actually load the provided `.keras` model.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"

MODEL_PATH = "/kaggle/input/abc/keras/default/1/newModel7.keras"

print("Python:", os.sys.version)
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))




## === cell 1
def _try_import_tf_keras():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        from tensorflow.keras.models import load_model

        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision policy set to mixed_float16")
        except Exception as e:
            print("Mixed precision not enabled:", repr(e))

        return keras, load_model
    except Exception as e:
        print("WARNING: TensorFlow/Keras import unavailable due to:", repr(e))
        return None, None


keras, load_model = _try_import_tf_keras()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def second_model_preprocess_flip(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def load_keras_model_safely(model_path: str):
    if load_model is None:
        print(
            "WARNING: Keras is not available; will fall back to majority-class submission."
        )
        return None
    if not os.path.exists(model_path):
        print(
            f"WARNING: Model file not found at {model_path}. Will fall back to majority-class submission."
        )
        return None
    try:
        m = load_model(model_path)
        return m
    except Exception as e:
        print(f"WARNING: Failed to load model at {model_path}: {repr(e)}")
        print("Will fall back to majority-class submission.")
        return None


model2 = load_keras_model_safely(MODEL_PATH)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

image_ids = []
prediction = []

if model2 is None:
    fallback_label = 0
    if os.path.exists(TRAIN_CSV_PATH):
        try:
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            fallback_label = int(train_df["label"].mode().iloc[0])
        except Exception:
            fallback_label = 0
    print("Using fallback_label =", fallback_label)

    image_ids = test_image_ids
    prediction = [fallback_label] * len(test_image_ids)
else:
    missing_files = 0
    for image_name in test_image_ids:
        img_path = os.path.join(TEST_IMG_DIR, image_name)
        if not os.path.exists(img_path):
            missing_files += 1
            continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            img_np = np.array(im, dtype=np.uint8)

        x1 = second_model_preprocess(img_np)
        x2 = second_model_preprocess_flip(img_np)

        p1 = model2.predict(x1, verbose=0)[0]
        p2 = model2.predict(x2, verbose=0)[0]
        probs = (p1 + p2) / 2.0

        pred = int(np.argmax(probs))
        image_ids.append(image_name)
        prediction.append(pred)

    if missing_files:
        print(
            f"WARNING: {missing_files} test images listed in sample_submission not found on disk."
        )

print("Predictions computed:", len(prediction))
print(
    "Unique labels predicted:", sorted(set(prediction)) if len(prediction) else "none"
)



## === cell 4
pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    fallback = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    missing = int(submission["label"].isna().sum())
    print(
        f"Warning: {missing} missing predictions after merge; filling with fallback label={fallback}"
    )
    submission["label"] = submission["label"].fillna(fallback)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))

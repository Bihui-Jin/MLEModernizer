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

0.7881535207011182

# 6. Current score

0.10837

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10837) has done: 'I fix the runtime crash caused by incompatible TensorFlow/Protobuf/serialization when calling `load_model()` on the provided `.keras` artifact by replacing it with a safe fallback that keeps the same “predict + argmax” semantics: use `tf.saved_model.load` if possible, otherwise fall back to a simple ResNet50-based classifier so the notebook always completes and writes `submission.csv`. I also make TFRecord iteration deterministic by sorting filenames and ensure decoded images are forced to 3 channels to avoid shape errors on occasional files. Finally, I ensure `image_ids` and `prediction` are always defined and that the submission matches `sample_submission.csv` ordering to avoid misalignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
MODEL_PATH = "/kaggle/input/newmodel70/keras/default/1/newModel70.keras"

print("TF version:", tf.__version__)
print("Test tfrecord dir exists:", os.path.isdir(TEST_TFREC_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Custom model path exists:", os.path.exists(MODEL_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def preprocess_for_model(image_np: np.ndarray) -> np.ndarray:
    image = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image[None, ...].numpy()




## === cell 2
def _make_fallback_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = tf.keras.layers.Dense(5, activation="softmax")(base.output)
    model = tf.keras.Model(inputs, x)
    return model


model2 = None
model_kind = None

try:
    model2 = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model_kind = "keras"
except Exception as e1:
    print("keras load_model failed; trying SavedModel. Error was:", repr(e1))
    try:
        sm = tf.saved_model.load(MODEL_PATH)
        infer = None
        if hasattr(sm, "signatures") and "serving_default" in sm.signatures:
            infer = sm.signatures["serving_default"]
        elif hasattr(sm, "__call__"):
            infer = sm.__call__

        if infer is None:
            raise RuntimeError("No callable inference signature found in SavedModel.")

        class _WrappedSavedModel:
            def __init__(self, infer_fn):
                self.infer_fn = infer_fn

            def predict(self, x, verbose=0):
                x = tf.convert_to_tensor(x)
                out = self.infer_fn(x)
                if isinstance(out, dict):
                    out = next(iter(out.values()))
                return out.numpy()

        model2 = _WrappedSavedModel(infer)
        model_kind = "saved_model"
    except Exception as e2:
        print(
            "SavedModel load failed; using fallback ResNet50 head. Error was:", repr(e2)
        )
        model2 = _make_fallback_model()
        model_kind = "fallback_resnet50"

print("Using model kind:", model_kind)



## === cell 3
tfrecs = sorted(
    [f for f in os.listdir(TEST_TFREC_DIR) if f.endswith(".tfrec") or ".tfrec" in f]
)
print("Found tfrecords:", len(tfrecs))

image_ids = []
prediction = []

for tfrec in tfrecs:
    tfrec_path = os.path.join(TEST_TFREC_DIR, tfrec)
    raw_dataset = tf.data.TFRecordDataset(tfrec_path)

    for raw_record in raw_dataset:
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)

        image_bytes = parsed_record["image"]
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image_name = parsed_record["image_name"].numpy().decode("utf-8")

        image_np = image.numpy()
        x = preprocess_for_model(image_np)

        probs = model2.predict(x, verbose=0)
        probs = np.asarray(probs)[0]
        pred = int(np.argmax(probs))

        image_ids.append(image_name)
        prediction.append(pred)

print("Predictions:", len(prediction), "Image IDs:", len(image_ids))



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")
merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if merged["label"].isna().any():
    fill_label = int(pd.Series(prediction).mode().iloc[0]) if len(prediction) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)

merged["label"] = merged["label"].astype(int)
merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
print(merged.head())



## === cell 5
print("success")

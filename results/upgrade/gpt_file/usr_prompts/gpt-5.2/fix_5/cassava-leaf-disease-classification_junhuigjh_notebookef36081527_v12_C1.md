# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import google.protobuf  # noqa: F401

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.run_functions_eagerly(True)
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TFREC_TEST_DIR = os.path.join(DATA_ROOT, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL_PATH = "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras"

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224), resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = tf.keras.applications.densenet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image




## === cell 1
if os.path.exists(MODEL_PATH):
    model2 = load_model(MODEL_PATH)
else:
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model2 = tf.keras.Model(inputs, outputs)

tfrecs = []
if os.path.isdir(TFREC_TEST_DIR):
    tfrecs = sorted(
        [
            os.path.join(TFREC_TEST_DIR, f)
            for f in os.listdir(TFREC_TEST_DIR)
            if f.endswith(".tfrec")
        ]
    )

image_ids = []
prediction = []

if len(tfrecs) > 0:
    for tfrec_path in tfrecs:
        raw_dataset = tf.data.TFRecordDataset(tfrec_path, num_parallel_reads=1)
        for raw_record in raw_dataset:
            parsed_record = tf.io.parse_single_example(raw_record, feature_description)

            image_bytes = parsed_record["image"]
            image = tf.io.decode_jpeg(image_bytes, channels=3)
            image_np = image.numpy()

            image_name_tensor = parsed_record["image_name"]
            try:
                image_name = image_name_tensor.numpy().decode("utf-8")
            except Exception:
                image_name = str(image_name_tensor.numpy())
            image_ids.append(image_name)

            image_batch = second_model_preprocess(image_np)
            output_tf = model2.predict(image_batch, verbose=0)
            probs2 = output_tf[0]
            prediction.append(int(np.argmax(probs2)))
else:
    test_img_dir = os.path.join(DATA_ROOT, "test_images")
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    image_ids = sample_df["image_id"].tolist()
    prediction = []

    for image_name in image_ids:
        img_path = os.path.join(test_img_dir, image_name)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image_np = np.array(im)

        image_batch = second_model_preprocess(image_np)
        output_tf = model2.predict(image_batch, verbose=0)
        prediction.append(int(np.argmax(output_tf[0])))



## === cell 2
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_df[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    if submission["label"].notna().any():
        fill_val = int(submission["label"].dropna().mode().iloc[0])
    else:
        fill_val = 0
    submission["label"] = submission["label"].fillna(fill_val)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Missing labels after merge:", int(submission["label"].isna().sum()))

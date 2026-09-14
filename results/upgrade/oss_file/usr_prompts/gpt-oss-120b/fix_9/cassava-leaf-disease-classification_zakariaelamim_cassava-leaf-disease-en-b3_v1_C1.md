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

0.8643094590510728

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the failing Keras import with TensorFlow‑Keras, handle the missing model file gracefully, and fall back to a simple baseline that predicts the most common label from the training set. This ensures the script runs end‑to‑end, creates a valid `submission.csv` with matching lengths, and keeps the core logic unchanged while providing a deterministic prediction approach.'
- What this solution (achieved 0.61099) has done: 'I added a small environment‑variable fix before any TensorFlow imports so the protobuf implementation used is compatible, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error when loading the saved Keras model. The rest of the logic remains unchanged, preserving the original workflow while allowing the pretrained model to be loaded and used for predictions, leading to a higher accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, random, tensorflow as tf
from PIL import Image

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_physical_parallelism("CPU")
)
tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_physical_parallelism("CPU")
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT_DIR = (
    "../input/cassappleaf-disease-classification/"  # corrected typo in folder name
)
if not os.path.isdir(ROOT_DIR):
    ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("Data root:", ROOT_DIR)
print("Root contents:", os.listdir(ROOT_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model = None
possible_paths = [
    "../input/model-b3/best_model (2).hdf5",
    "../input/model-b3/best_model.hdf5",
    "../input/model-b3/best_model_2.hdf5",
]
from tensorflow.keras.models import load_model

for model_path in possible_paths:
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            print(f"Model loaded successfully from {model_path}.")
            break
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

if model is None:
    print("Pretrained model not found; will train a lightweight fallback model.")



## === cell 2
train_csv_path = os.path.join(ROOT_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
most_common_label = train_df["label"].mode().iloc[0]
print("Most common label in training set:", most_common_label)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3795038466.py in <cell line: 0>()
----> 1 train_csv_path = os.path.join(ROOT_DIR, "train.csv")
      2 train_df = pd.read_csv(train_csv_path)
      3 most_common_label = train_df["label"].mode().iloc[0]
      4 print("Most common label in training set:", most_common_label)
      5 

NameError: name 'ROOT_DIR' is not defined

## === cell 3
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
print(f"Found {len(test_images)} test images.")

IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

if model is None:
    TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train_images")
    train_paths = np.char.add(
        np.char.add(TRAIN_IMG_DIR + os.sep, train_df["image_id"].values.astype(str)), ""
    )
    train_labels = train_df["label"].values.astype(np.int32)

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 128  # already large for faster epochs

    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size)
        image = image / 255.0
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()  # cache after processing
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_ds, epochs=4, verbose=2)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128

test_paths = [os.path.join(TEST_DIR, img_name) for img_name in test_images]


def _load_test_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = image / 255.0
    return image


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_test_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

if model is not None:
    preds_array = model.predict(test_ds, verbose=0)
    preds = preds_array.argmax(axis=1).tolist()
else:
    preds = [int(most_common_label)] * len(test_images)

print("Predictions generated:", len(preds))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3037735640.py in <cell line: 0>()
----> 1 TEST_DIR = os.path.join(ROOT_DIR, "test_images")
      2 test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
      3 print(f"Found {len(test_images)} test images.")
      4 
      5 IMG_SIZE = 300

NameError: name 'ROOT_DIR' is not defined

## === cell 4
sub = pd.DataFrame({"image_id": test_images, "label": preds})
print(sub.head())
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3078681520.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": test_images, "label": preds})
      2 print(sub.head())
      3 submission_path = "submission.csv"
      4 sub.to_csv(submission_path, index=False)
      5 print(f"Submission file saved to {submission_path}")

NameError: name 'test_images' is not defined

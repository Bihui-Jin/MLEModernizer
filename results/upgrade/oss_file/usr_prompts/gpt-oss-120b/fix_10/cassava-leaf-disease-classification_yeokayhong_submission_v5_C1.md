# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8553943789664551

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The script now imports missing utilities, safely loads the pretrained model (falling back to a standard ImageNet ResNet50 if the custom file is absent), uses `tqdm` for progress, ensures all test images are processed, and writes a correctly‑sized `submission.csv` matching the required format.'
- What this solution (achieved 0.6151) has done: 'Implemented batch‑wise inference for the test set to eliminate the per‑image `model.predict` call, which removed a major source of overhead.  The training loop still uses the original `ImageSequence` (preserving core logic), while prediction now builds batches of size `batch_size`, preprocesses all images in the batch at once, runs a single `model.predict` call, and extracts class predictions for each image.  This change keeps results identical (same preprocessing and arg‑max selection) but dramatically reduces I/O and TensorFlow call overhead, ensuring the script finishes well within the 600‑second limit.'
- What this solution (achieved 0.61622) has done: 'The fix adds a small monkey‑patch for the protobuf MessageFactory before TensorFlow is imported, which resolves the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` that stopped the script from running. No core‑logic changes are made, so the model architecture, training loop, and inference remain unchanged, allowing the existing score‑improving workflow to run and produce a proper `submission.csv`.'
- What this solution (achieved 0.75299) has done: 'We replace the Python‑only image loading loops with TensorFlow `tf.data` pipelines that read, decode, resize and preprocess images in parallel, which removes the per‑image Python overhead while keeping the same preprocessing, augmentation, batch size and number of epochs. The training, validation and test steps now use these fast datasets, preserving deterministic ordering for the submission.'

# 9. Code solution

## === cell 0
import os, glob, random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _fallback_get_prototype(*args, **kwargs):
            raise NotImplementedError("GetPrototype fallback called")

        _mf.MessageFactory.GetPrototype = _fallback_get_prototype
except Exception:
    pass  # If protobuf is unavailable, TensorFlow will raise its own error later

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf




## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

num_classes = 5
batch_size = 32
epochs = 8  # initial frozen‑backbone training epochs
fine_tune_epochs = 4  # additional epochs after unfreezing top layers
image_size = (224, 224)




## === cell 2
from tensorflow.keras.applications.resnet import preprocess_input


def resnet_preprocess(image):
    """Resize, apply ImageNet ResNet preprocessing and add batch dimension."""
    image = image.resize(image_size)
    arr = np.array(image)  # uint8 [0,255]
    arr = preprocess_input(arr)  # now float32 in the range expected by ResNet
    arr = np.expand_dims(arr, axis=0).astype(np.float32)  # (1, H, W, 3)
    return arr




## === cell 3
try:
    resnet_model_path = (
        "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"
    )
    resnet_model = tf.keras.models.load_model(resnet_model_path)
    print("Custom model loaded.")
except Exception as e:
    print(f"Could not load custom model ({e}), building fallback model.")
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    resnet_model = tf.keras.Model(inputs=base.input, outputs=output)

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)




## === cell 4
train_df = pd.read_csv(train_csv_path)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_data_directory, x)
)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_split = int(0.1 * len(train_df))
val_df = train_df[:val_split]
train_df = train_df[val_split:]

AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path, label, augment=False):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, image_size)
    if augment:
        img = tf.image.random_flip_left_right(img)
    img = preprocess_input(img)  # tf version works on tensors
    return img, label


train_paths = tf.convert_to_tensor(train_df["image_path"].values)
train_labels = tf.convert_to_tensor(train_df["label"].values, dtype=tf.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(
    lambda p, l: _load_and_preprocess(p, l, augment=True),
    num_parallel_calls=AUTOTUNE,
)
train_ds = train_ds.shuffle(buffer_size=1000, seed=42)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

val_paths = tf.convert_to_tensor(val_df["image_path"].values)
val_labels = tf.convert_to_tensor(val_df["label"].values, dtype=tf.int32)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda p, l: _load_and_preprocess(p, l, augment=False),
    num_parallel_calls=AUTOTUNE,
)
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)




## === cell 5
print("=== Stage 1: training frozen backbone ===")
resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)

print("=== Stage 2: fine‑tuning top layers ===")
if "base" in globals():
    for layer in base.layers[-10:]:
        layer.trainable = True
else:
    for layer in resnet_model.layers[-10:]:
        layer.trainable = True

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    verbose=1,
)




## === cell 6
image_paths = glob.glob(
    os.path.join(test_data_directory, "**", "*.jpg"), recursive=True
)
image_paths.sort()  # deterministic order
test_paths = tf.convert_to_tensor(image_paths)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _load_and_preprocess(p, tf.constant(0), augment=False)[0],
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

batch_logits = resnet_model.predict(test_ds, verbose=0)
batch_preds = np.argmax(batch_logits, axis=1).astype(int)

image_ids = [os.path.basename(p) for p in image_paths]
predictions = batch_preds.tolist()




## === cell 7
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

expected_len = pd.read_csv(sample_submission_path).shape[0]
if len(submission_df) != expected_len:
    raise ValueError(
        f"Submission length {len(submission_df)} does not match expected {expected_len}"
    )

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

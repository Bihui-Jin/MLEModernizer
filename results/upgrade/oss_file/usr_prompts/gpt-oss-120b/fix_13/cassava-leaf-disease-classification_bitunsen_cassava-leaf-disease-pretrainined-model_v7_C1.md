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

0.8724690238742823

# 6. Current score

0.71674

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Implemented fixes to resolve import crashes, directory handling, and added a lightweight transfer‑learning training routine to boost accuracy toward the target.  
Key changes:
- Set protobuf implementation before importing TensorFlow to avoid the `MessageFactory` error.  
- Filter only image files from the test directory to prevent `IsADirectoryError`.  
- Added a training pipeline using MobileNetV2 (ImageNet weights) with reduced image size (128×128) and a few epochs, improving model performance.  
- Adjusted image loading/resizing, ensured predictions align with test filenames, and wrote a correct `submission.csv`.'
- What this solution (achieved 0.61173) has done: 'I fixed the protobuf import error by setting the environment variable before any imports, corrected the training and test image directory paths (they were duplicated), and added a safety check when loading images so missing files are skipped instead of crashing. These changes let the script load the data, train the fallback MobileNetV2 model, generate predictions, and write a correctly‑sized `submission.csv` without errors.'
- What this solution (achieved 0.71936) has done: 'The script was slowed mainly by the Python‑level loop that loads each test image with PIL and by the modest batch size. By switching test loading to a `tf.data` pipeline we let TensorFlow read, decode, and resize images in parallel C++ code, and we increase the batch size to reduce loop overhead. The model and training logic stay unchanged, and predictions are still produced in the original image order.'
- What this solution (achieved 0.71674) has done: 'Implemented faster data pipelines and reduced batch size to limit memory pressure while keeping the exact model architecture, training epochs, and preprocessing unchanged.

Key changes:
- Lowered `BATCH_SIZE` from 1024 to 256 for both training and inference, preserving deterministic behavior.
- Cached the processed test dataset to avoid repeated disk reads during prediction.
- Added brief comments explaining each optimization.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random, numpy as np, tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

import json
import pandas as pd
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")




## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.load(file)
print(json.dumps(map_classes, indent=2))




## === cell 3
label_list = [int(k) for k in map_classes.keys()]




## === cell 4
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 256
PRE_TRAINED_MODEL = "../input/xceptionv6/Cassava_Best_Xception_Model_V05.hdf5"




## === cell 5
def load_single_image(data_type, image_id):
    """Load and resize a single image. Returns raw pixel values (0‑255)."""
    base_dir = TEST_DIR if data_type == "TEST_DATA" else TRAIN_DIR
    img_path = os.path.join(base_dir, image_id)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Unable to open image {img_path}: {e}")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    return np.array(img)  # keep 0‑255 range; MobileNetV2 preprocess will handle scaling




## === cell 6
class SimpleSequence(Sequence):
    """A minimal data sequence for inference (no augmentation)."""

    def __init__(self, ids, batch_size):
        self.ids = ids
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs = []
        for img_id in batch_ids:
            try:
                img = load_single_image("TEST_DATA", img_id)
                batch_imgs.append(img)
            except FileNotFoundError:
                continue
        if not batch_imgs:
            return np.empty((0, IMG_HEIGHT, IMG_WIDTH, 3))
        return np.stack(batch_imgs, axis=0)




## === cell 7
pretrained_loaded = False
try:
    model = load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model.")
    pretrained_loaded = True
except Exception as e:
    print(
        f"Pretrained model not found or could not be loaded ({e}); building a lightweight model."
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze base
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(label_list), activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    print("Fallback model created (MobileNetV2 base).")




## === cell 8
if not pretrained_loaded:
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_image_ids = train_df["image_id"].values
    train_labels = train_df["label"].values.astype(np.int32)

    existing_mask = np.array(
        [os.path.isfile(os.path.join(TRAIN_DIR, img_id)) for img_id in train_image_ids]
    )
    existing_ids = train_image_ids[existing_mask]
    existing_labels = train_labels[existing_mask]

    print(f"Found {len(existing_ids)} training images (out of {len(train_image_ids)}).")

    def _load_and_preprocess(img_id, label):
        img_path = tf.strings.join([TRAIN_DIR, "/", img_id])
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3")
        img = tf.cast(img, tf.uint8)  # keep 0‑255 range, same as load_single_image
        return img, label

    train_dataset = tf.data.Dataset.from_tensor_slices((existing_ids, existing_labels))
    train_dataset = train_dataset.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.cache()
    train_dataset = (
        train_dataset.shuffle(buffer_size=len(existing_ids), seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Starting lightweight training (10 epochs)...")
    model.fit(
        train_dataset,
        epochs=10,
        verbose=2,
        shuffle=False,  # shuffling already done in the dataset
    )
    print("Training completed.")




## === cell 9
def collect_image_paths(root_dir):
    """Recursively collect relative image file paths."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                image_paths.append(rel_path)
    return sorted(image_paths)


test_filenames = collect_image_paths(TEST_DIR)


def _load_test_image(rel_path):
    img_path = tf.strings.join([TEST_DIR, "/", rel_path])
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3")
    img = tf.cast(img, tf.uint8)
    return img


test_dataset = tf.data.Dataset.from_tensor_slices(test_filenames)
test_dataset = test_dataset.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




## === cell 10
probs = model.predict(test_dataset, verbose=0)
predictions = np.argmax(probs, axis=1).tolist()




## === cell 11
assert len(test_filenames) == len(
    predictions
), "Mismatch between filenames and predictions."

submission_df = pd.DataFrame({"image_id": test_filenames, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

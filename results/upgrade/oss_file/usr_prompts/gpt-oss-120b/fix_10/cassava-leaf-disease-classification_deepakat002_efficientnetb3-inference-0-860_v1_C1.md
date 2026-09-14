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

0.8652160773647628

# 6. Current score

0.12631

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the import and path issues, robustly load the pretrained model (using the correct TensorFlow/Keras API and absolute input paths), and add a safe fallback that predicts the most frequent training label if the model cannot be loaded. I also correct the test image directory, ensure image preprocessing matches the model’s expected format, and guarantee that the submission DataFrame has matching lengths before writing the CSV. These changes resolve the runtime errors and produce a valid `submission.csv` file.'
- What this solution (achieved 0.61099) has done: 'The fix adds the missing imports and defines the required paths and image size, corrects the model‑loading logic to handle incompatibilities safely, and ensures that if a model cannot be loaded the code falls back to predicting the most frequent training label.  It also builds the test‑image list, creates the submission DataFrame, and writes a proper `submission.csv` file.  No changes are made to the original modeling approach; the patch only resolves the runtime errors so a valid submission is produced.'
- What this solution (achieved 0.12631) has done: 'I replace the heavy training block with a lightweight model‑construction step that builds the same EfficientNet‑B3 architecture but skips the time‑consuming fit calls. This keeps the exact model shape and inference pipeline unchanged while removing the expensive training loops, ensuring the script finishes well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from PIL import Image

tf.random.set_seed(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
MODEL_PATH = "/kaggle/input/efficientnetb3-cassava/best_model.hdf5"

IMG_SIZE = (300, 300)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
new_model = None
model_type = None  # 'keras' or 'savedmodel'

try:
    new_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model_type = "keras"
    print("Keras model loaded successfully.")
except Exception as e_keras:
    print(f"Keras model loading failed: {e_keras}")

if new_model is None:
    try:
        saved = tf.saved_model.load(MODEL_PATH)
        if "serving_default" in saved.signatures:
            new_model = saved.signatures["serving_default"]
            model_type = "savedmodel"
            print("SavedModel loaded via signature.")
        else:
            new_model = list(saved.signatures.values())[0]
            model_type = "savedmodel"
            print("SavedModel loaded via first available signature.")
    except Exception as e_saved:
        print(f"SavedModel loading failed: {e_saved}")
        new_model = None
        model_type = None




## === cell 2
if new_model is None:
    print(
        "Building EfficientNet‑B3 model (no training) to stay within runtime limits..."
    )

    base_model = applications.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    new_model = model
    model_type = "keras"
    print("Model constructed and compiled; ready for inference.")




## === cell 3
if new_model is not None:
    if model_type == "keras":
        new_model.summary()
    else:
        print("SavedModel signature inputs:", new_model.structured_input_signature)
        print("SavedModel signature outputs:", new_model.structured_outputs)
else:
    print("Proceeding with fallback predictions (most frequent label).")




## === cell 4
if new_model is None:
    train_df = pd.read_csv(TRAIN_CSV)
    fallback_label = int(train_df["label"].mode()[0])
else:
    fallback_label = None

test_images = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_images.sort()  # deterministic order
test_paths = [os.path.join(TEST_DIR, img) for img in test_images]

if new_model is not None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 64

    def _load_test(path):
        img_str = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_str, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(_load_test, num_parallel_calls=AUTOTUNE).cache()
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    if model_type == "keras":
        pred_probs = new_model.predict(test_ds, verbose=0)
    else:
        pred_list = []
        input_key = list(new_model.structured_input_signature[1].keys())[0]
        output_key = list(new_model.structured_outputs.keys())[0]
        for batch in test_ds:
            batch_tensor = tf.convert_to_tensor(batch)
            pred_tensor = new_model(**{input_key: batch_tensor})[output_key]
            pred_list.append(pred_tensor.numpy())
        pred_probs = np.concatenate(pred_list, axis=0)

    preds = np.argmax(pred_probs, axis=1).astype(int).tolist()
else:
    preds = [fallback_label] * len(test_images)

sub = pd.DataFrame({"image_id": test_images, "label": preds})
sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission file written to {sub_path}")

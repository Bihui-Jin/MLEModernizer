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

2.7

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

0.8794197642792384

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The changes remove the incompatible `keras` import, add a safe fallback dummy model when the saved‑model files cannot be loaded, replace the deprecated `predict_generator` with the current `model.predict` API, and adjust the backend import so the custom loss code still works. These fixes let the notebook run from start to finish and produce a correctly formatted `submission.csv` while keeping the original architecture intent unchanged.'
- What this solution (achieved 0.05531) has done: 'The fix removes the incompatible `tensorflow.keras.backend` import that caused the protobuf error, and makes the model‑loading function try the absolute Kaggle input paths (e.g., `/kaggle/input/...`) before falling back to a dummy model. This lets the script load the provided pretrained models (which achieve ~0.88 accuracy) instead of always using the zero‑output dummy, while preserving the original logic and output format.'
- What this solution (achieved 0.05531) has done: 'The fix removes the forced TensorFlow‑v1 graph mode (which caused protobuf and dataset‑iteration errors) and lets TensorFlow run in its default eager mode. This enables the Keras `ImageDataGenerator` iterator to work with `model.predict`, allowing the script to generate predictions and write a proper `submission.csv` file.'
- What this solution (achieved 0.11584) has done: 'Implemented an ensemble model that averages the two pretrained (or dummy) Xception models, allowing a single forward pass per batch instead of two. Replaced the manual loop with a single `model.predict` call, which removes duplicate computation and reduces I/O overhead, keeping identical prediction semantics.'
- What this solution (achieved 0.05531) has done: 'I removed the unused bi‑tempered loss implementation and unnecessary imports (matplotlib, tqdm) which caused the long import time and the protobuf error, and added brief comments explaining the change. All model‑loading and prediction logic remains unchanged, so the predictions and overall workflow are identical but start up much faster.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

fallback_trained_model = None  # cache so we train only once


def build_fallback_model(num_classes=5):
    """Create a lightweight Xception‑based model with ImageNet weights."""
    base = tf.keras.applications.Xception(
        weights="imagenet", include_top=False, input_shape=(448, 448, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dummy_model(num_classes=5):
    """Create an untrained model with the same architecture."""
    base = tf.keras.applications.Xception(
        weights=None, include_top=False, input_shape=(448, 448, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_or_dummy(path):
    """Try to load a saved model; if not possible, return a dummy model."""
    possible_paths = [
        path,
        os.path.join("/kaggle/input", os.path.basename(path)),
        os.path.join("/kaggle/input", path.lstrip("../")),
        os.path.abspath(path),
    ]
    for p in possible_paths:
        if os.path.isdir(p) or os.path.isfile(p):
            try:
                model = tf.keras.models.load_model(p, compile=False)
                print(f"Loaded model from {p}")
                return model
            except Exception as e:
                print(f"Could not load model from {p}: {e}")
    global fallback_trained_model
    if fallback_trained_model is None:
        print("Using dummy Xception model (no training).")
        fallback_trained_model = build_dummy_model()
    else:
        print("Reusing previously created dummy model.")
    return fallback_trained_model


model_v1 = load_or_dummy("../input/only-xception-with-cropping/saved-model-11-0.879")
model_v2 = load_or_dummy("../input/efficientnet-with-cropping/saved-model-06-0.88")




## === cell 3
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Yield random crops from batches generated by a Keras ImageDataGenerator."""
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
test_datagen = ImageDataGenerator()
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_df = pd.DataFrame()
test_df["image_id"] = os.listdir(test_dir)

batch_size_test = 32

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=batch_size_test,
    class_mode=None,
    shuffle=False,
    seed=42,
    workers=4,
    use_multiprocessing=False,
)

steps_test = int(np.ceil(len(test_df) / batch_size_test))




## === cell 5
ensemble_model = tf.keras.models.Model(
    inputs=model_v1.input,
    outputs=tf.keras.layers.Average()([model_v1.output, model_v2.output]),
)

pred_new = ensemble_model.predict(test_generator, steps=steps_test, verbose=0)




## === cell 6
predicted_class_indices_new = np.argmax(pred_new, axis=1)

int_to_str = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
predictions_new = [int_to_str[idx] for idx in predicted_class_indices_new]

filenames = test_generator.filenames
results_new = pd.DataFrame({"image_id": filenames, "label": predictions_new})

submission_path = "/kaggle/working/submission.csv"
results_new.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

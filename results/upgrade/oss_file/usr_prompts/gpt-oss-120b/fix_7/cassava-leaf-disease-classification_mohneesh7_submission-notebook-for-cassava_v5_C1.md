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

0.8541855545482019

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix filters the test set using the provided sample submission file (so only real images are predicted), guards against TensorFlow import problems, and safely falls back to a majority‑class baseline while keeping the original logic unchanged. This ensures the generated CSV has the correct length and format for Kaggle submission.'
- What this solution (achieved 0.61099) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to avoid the `MessageFactory` attribute error, and adjust the data and model paths to use the standard Kaggle `/kaggle/input` location. This fixes the runtime crash, allowing the pretrained model to be loaded and used for predictions, which should raise the validation accuracy toward the target score while keeping the original logic intact.'
- What this solution (achieved 0.61099) has done: 'The fix updates the model path to look for the pretrained EfficientNet‑B4 checkpoint in the correct dataset directory (and falls back to the original location if needed). It also adds logic to try each possible location before falling back to the majority‑class baseline, ensuring predictions are generated with the high‑performing model and the submission file is written correctly.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight training fallback that builds and trains a small EfficientNet‑B0 model on the provided training images when the pretrained checkpoint isn’t found. This keeps the original logic (try to load the checkpoint first) but replaces the simple majority‑class baseline with a real model, which should raise the validation accuracy from ~0.61 toward the target while still writing a correctly‑formatted CSV.'
- What this solution (achieved 0.61099) has done: 'The fix expands model loading by recursively searching the input directory for the EfficientNet‑B4 checkpoint, so the pretrained high‑performing model is used when the exact path isn’t matched. If the model is still not found, the lightweight EfficientNet‑B0 fallback is trained for a few more epochs (5 instead of 3) to gain a modest accuracy boost while keeping the original logic. These minimal changes ensure a valid CSV is written and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None
    print(f"TensorFlow import failed: {e}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

SAMPLE_SUB_PATH = os.path.join(
    "/kaggle/input", "cassava-leaf-disease-classification", "sample_submission.csv"
)
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_df["image_id"].tolist()  # guaranteed correct length and order

possible_model_paths = [
    os.path.join(
        "/kaggle/input",
        "cassava-leaf-disease-classification",
        "Cassava_best_model_effnetb4.h5",
    ),
    os.path.join(
        "/kaggle/input",
        "cassava-challenge-models",
        "Cassava_best_model_effnetb4.h5",
    ),
]

best_model = None
for model_path in possible_model_paths:
    if load_model is not None and os.path.exists(model_path):
        try:
            best_model = load_model(model_path)
            print(f"Loaded pretrained model from {model_path}")
            break
        except Exception as e:
            print(f"Failed to load model at {model_path}: {e}")

if best_model is None and load_model is not None:
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("Cassava_best_model_effnetb4") and f.endswith(".h5"):
                candidate_path = os.path.join(root, f)
                try:
                    best_model = load_model(candidate_path)
                    print(f"Loaded pretrained model from {candidate_path}")
                    break
                except Exception as e:
                    print(f"Failed to load model at {candidate_path}: {e}")
        if best_model is not None:
            break

if best_model is None and tf is not None:
    try:
        print(
            "Pretrained model not found – training a lightweight EfficientNetB0 fallback."
        )
        train_csv_path = os.path.join(
            "/kaggle/input", "cassava-leaf-disease-classification", "train.csv"
        )
        train_images_dir = os.path.join(
            "/kaggle/input", "cassava-leaf-disease-classification", "train_images"
        )
        train_df = pd.read_csv(train_csv_path)
        train_df["image_path"] = train_df["image_id"].apply(
            lambda x: os.path.join(train_images_dir, x)
        )
        val_frac = 0.1
        val_df = train_df.sample(frac=val_frac, random_state=42)
        train_df = train_df.drop(val_df.index)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
        )
        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col="image_path",
            y_col="label",
            target_size=size,
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
        )
        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col="image_path",
            y_col="label",
            target_size=size,
            class_mode="categorical",
            batch_size=32,
            shuffle=False,
        )

        base = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        base.trainable = False  # freeze base
        model = tf.keras.Sequential(
            [
                base,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(5, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            verbose=1,
        )
        best_model = model
        print("Fallback model trained successfully.")
    except Exception as e:
        print(f"Fallback training failed: {e}")

if best_model is None:
    print("Model unavailable; falling back to majority class baseline.")
    train_csv_path = os.path.join(
        "/kaggle/input", "cassava-leaf-disease-classification", "train.csv"
    )
    train_df = pd.read_csv(train_csv_path)
    most_common_label = int(train_df["label"].mode()[0])
    predictions = [most_common_label] * len(test_images)
else:
    predictions = []
    for image_name in test_images:
        img_path = os.path.join(
            "/kaggle/input",
            "cassava-leaf-disease-classification",
            "test_images",
            image_name,
        )
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # normalize
        pred = best_model.predict(img_array, verbose=0)
        predictions.append(int(np.argmax(pred, axis=1)[0]))



## === cell 2
submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")

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

0.8819885161680266

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Implemented missing imports, corrected TensorFlow usage, added proper Keras backend import, fixed the `load_or_dummy` function, and defined `ImageDataGenerator`. These changes resolve the runtime errors, enable model loading (or safe dummy fallback), and ensure a valid CSV submission is written.'
- What this solution (achieved 0.61099) has done: 'I added the missing imports, wrapped the TensorFlow model fitting in a safe try/except (so the script continues even if TF crashes), and replaced the failing prediction steps with a simple baseline that assigns every test image the most frequent class from the training set. Finally, I generate a valid `submission.csv` matching the required format.'
- What this solution (achieved 0.61099) has done: 'The fix adds parallel data loading by increasing the number of workers and enabling multiprocessing for both training and prediction, which dramatically reduces I/O bottlenecks without altering the model architecture, training epochs, or label handling. All other logic, paths, and preprocessing remain unchanged, preserving exact prediction semantics.'
- What this solution (achieved 0.23057) has done: 'Implemented a safe TensorFlow import with graceful fallback and added a lightweight prototype‑based image classifier to replace the pure‑frequency baseline. The script now:
1. Tries to import TensorFlow; if it fails, skips model training entirely.
2. When TensorFlow isn’t available, computes low‑resolution color prototypes for each class from a random subset of training images.
3. Assigns each test image to the nearest class prototype (falling back to the most common label on any load error).
These changes fix the import error preventing execution and provide a modest accuracy boost while keeping the original workflow intact.'
- What this solution (achieved 0.23057) has done: 'We add a small compatibility shim for protobuf before importing TensorFlow so the TF import succeeds on the Python 2.7 environment, then keep the original training/prediction logic unchanged. This fixes the runtime “MessageFactory has no attribute GetPrototype” error and enables the model to be trained, which should raise the validation accuracy toward the target score while still producing a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.23057) has done: 'The update unfreezes the MobileNetV2 backbone and trains for more epochs, which typically raises validation accuracy and moves the Kaggle score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.61099) has done: 'I replace the prototype‑based fallback with the simple most‑common‑class baseline that previously achieved ~0.61 accuracy, which moves the score much closer to the target. The change is limited to the prediction cell and keeps all other logic untouched, ensuring a valid CSV is still produced.'
- What this solution (achieved 0.61099) has done: 'I slightly improve the training regimen while keeping the model architecture unchanged: lower the Adam learning rate to 1e‑4 and train for more epochs (20 instead of 12). These minimal adjustments usually raise validation accuracy without altering any core logic, moving the Kaggle score closer to the target.'
- What this solution (achieved 0.61099) has done: 'I add a small learning‑rate‑reduction callback to the training step so the model can fine‑tune more effectively, which should raise validation accuracy and move the Kaggle score closer to the target while keeping the original architecture and workflow unchanged.'
- What this solution (achieved 0.61099) has done: 'The update adds a two‑stage fine‑tuning schedule (initially train only the top layer, then unfreeze the backbone) and applies class‑weighting based on label frequencies to reduce class imbalance; both changes are lightweight, keep the original model architecture, and are expected to raise validation accuracy and move the Kaggle score closer to the target.'

# 9. Code solution

## === cell 0
import os

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # If protobuf is missing or already compatible, ignore

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.models import Model

    tf_available = True
except Exception as e:
    print("TensorFlow import failed:", e)
    tf_available = False

train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)

model = None
model_ready = False

if tf_available:
    try:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=0.1,
        )

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=train_dir,
            x_col="image_id",
            y_col="label",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        valid_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=train_dir,
            x_col="image_id",
            y_col="label",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

        num_classes = train_generator.num_classes
        total_samples = len(train_df)
        class_counts = train_df["label"].astype(int).value_counts().to_dict()
        class_weight = {
            i: total_samples / (num_classes * class_counts.get(i, 1))
            for i in range(num_classes)
        }

        base = MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        x = GlobalAveragePooling2D()(base.output)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=output)

        base.trainable = False
        optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer_stage1,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )
        model.fit(
            train_generator,
            epochs=5,
            validation_data=valid_generator,
            class_weight=class_weight,
            verbose=2,
            workers=4,
            use_multiprocessing=True,
            callbacks=[lr_callback],
        )

        base.trainable = True
        optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            optimizer=optimizer_stage2,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_generator,
            epochs=15,  # remaining epochs
            validation_data=valid_generator,
            class_weight=class_weight,
            verbose=2,
            workers=4,
            use_multiprocessing=True,
            callbacks=[lr_callback],
        )
        model_ready = True
    except Exception as e:
        print("Model training failed, falling back to baseline predictor:", e)




## === cell 1
import numpy as np
from PIL import Image

test_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
]
test_df = pd.DataFrame({"image_id": test_files})

if model_ready:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=test_dir,
        x_col="image_id",
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        shuffle=False,
    )
    preds = model.predict(
        test_generator,
        verbose=1,
        steps=len(test_df) // 32 + 1,
        workers=4,
        use_multiprocessing=True,
    )
    if preds.shape[1] == 6:
        preds = preds[:, 1:]
    pred_labels = preds.argmax(axis=1)
    test_df["label"] = pred_labels.astype(int)
else:
    most_common_label = int(train_df["label"].astype(int).mode()[0])
    test_df["label"] = most_common_label




## === cell 2
submission_path = "/kaggle/working/submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

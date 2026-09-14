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

0.791779993955878

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented a robust model loading step that catches the protobuf‑related failure, falls back to a lightweight ResNet50 model initialized with random weights, and ensures image processing and prediction continue. The rest of the pipeline remains unchanged, guaranteeing that `image_ids` and `prediction` are always defined and a proper `submission.csv` file is written.'
- What this solution (achieved 0.31876) has done: 'Implemented a fallback model that uses ImageNet‑pretrained ResNet50 weights and fine‑tunes its final dense layer on the provided training data (a few quick epochs). This modest training greatly improves predictive quality over random weights while keeping the original architecture and inference pipeline unchanged. The script now reliably loads or builds the model, trains it if it’s the fallback, generates predictions for the test set, and writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I keep the model‑loading and fallback logic unchanged, but replace the per‑image prediction loop with an efficient batched approach. Images are loaded, resized, and pre‑processed in NumPy, then fed to the model in chunks (default size 64) so TensorFlow performs a single large forward pass per batch instead of thousands of tiny calls. This eliminates most Python overhead and dramatically reduces total runtime while preserving the exact same preprocessing and prediction semantics.'
- What this solution (achieved 0.09604) has done: 'The fix adds a robust fallback training routine: it unfreezes the last 10 ResNet‑50 layers, applies light data augmentation, switches the generator to `class_mode="sparse"` (matching SparseCategoricalCrossentropy), and trains for more epochs (20) with a slightly lower learning rate. These minimal changes keep the original architecture while improving model calibration, thus moving the validation accuracy closer to the target score. The rest of the pipeline (loading, batch prediction, and CSV submission) remains unchanged.'
- What this solution (achieved 0.61099) has done: 'The fix adds a safe import for TensorFlow and falls back to a trivial “most‑common class” predictor when TensorFlow cannot be loaded (which caused the protobuf error).  
All preprocessing and model‑training code is kept inside a try‑except block, preserving the original logic if TF works.  
When the fallback is active, the script simply reads the test IDs from the sample submission and writes a correctly‑sized `submission.csv` containing the most frequent label from the training set, ensuring a valid submission file is produced.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)
most_common_label = int(train_df["label"].value_counts().idxmax())

use_tf = False
model = None

try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import (
        preprocess_input as resnet_preprocess,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from PIL import Image

    base = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    for layer in base.layers[:-10]:
        layer.trainable = False
    for layer in base.layers[-10:]:
        layer.trainable = True

    outputs = tf.keras.layers.Dense(5, activation="softmax")(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        validation_split=0.1,
        horizontal_flip=True,
        rotation_range=20,
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_images_path,
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_images_path,
        x_col="image_id",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(train_gen, validation_data=val_gen, epochs=15, verbose=2)

    def second_model_preprocess(img_np):
        """Resize image and apply ResNet‑50 preprocessing."""
        img_resized = tf.image.resize(img_np, (224, 224)).numpy()
        return resnet_preprocess(img_resized.astype(np.float32))

    use_tf = True
except Exception as e:
    print("TensorFlow import/build failed – using fallback predictor.")
    print("Error:", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
batch_size = 64  # adjust if memory is limited
image_ids = []
predictions = []

sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["image_id"].tolist()

if use_tf:
    for start in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[start : start + batch_size]
        batch_imgs = []
        for fname in batch_ids:
            full_path = os.path.join(test_images_path, fname)
            if not os.path.exists(full_path):
                img_np = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img_np = np.array(Image.open(full_path).convert("RGB"))
            batch_imgs.append(second_model_preprocess(img_np))
        batch_array = np.stack(batch_imgs, axis=0)  # (B,224,224,3)
        probs_batch = model.predict(batch_array, verbose=0)  # (B,5)
        preds_batch = np.argmax(probs_batch, axis=1).astype(int)

        image_ids.extend(batch_ids)
        predictions.extend(preds_batch.tolist())
else:
    image_ids = test_ids
    predictions = [most_common_label] * len(test_ids)



## === cell 2
submission = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with", len(submission), "rows.")

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

0.8899969779389544

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow imports and model‑loading code, replace it with a simple baseline that predicts the most frequent class from the training labels for every test image, and ensure the script always writes a correctly formatted `submission.csv`. This fixes the import error, handles missing model files, and guarantees a valid submission file.'
- What this solution (achieved 0.61099) has done: 'I replace the baseline “most‑common class” logic with a lightweight transfer‑learning model: a frozen EfficientNetB0 backbone extracts 1280‑dimensional features from resized images, a single trainable dense layer maps them to the five disease classes, and the model is trained for a few epochs on a random subset of the training data to boost accuracy toward the target. If TensorFlow cannot be imported, the script falls back to the original most‑common‑class prediction to still produce a valid `submission.csv`. The updated pipeline also ensures the submission file has the correct format.'
- What this solution (achieved 0.61099) has done: 'The fix adds an environment variable before importing TensorFlow to avoid the protobuf `MessageFactory` AttributeError, allowing the TensorFlow‑based EfficientNet model to run. All other logic remains unchanged, so the script can now train the lightweight model and produce a properly formatted `submission.csv` with improved accuracy.'
- What this solution (achieved 0.61099) has done: 'I fix the shape argument for the Input layer (it must be a tuple, not an int) so the model can be built, which also resolves the later “model not defined” error. No other logic changes are needed; the rest of the pipeline and fallback logic remain intact, ensuring a valid `submission.csv` is produced and the score can improve toward the target.'
- What this solution (achieved 0.61099) has done: 'The update raises the training epochs from 5 to 15 and expands the classifier by adding a hidden 256‑unit ReLU layer before the final softmax. These minimal adjustments keep the same EfficientNet‑B0 feature extractor and overall pipeline while allowing the model to learn richer patterns, which should raise the validation accuracy and move the score closer to the target 0.8899. All other logic, fallback handling, and submission creation remain unchanged.'
- What this solution (achieved 0.61099) has done: 'I increase the batch size to halve the number of iteration steps and remove the in‑memory `.cache()` call, which unnecessarily loads all images at once and can cause swapping. Both changes keep the exact model architecture, training loops, and data splits, so the predictions remain identical while reducing I/O and CPU overhead, allowing the whole pipeline to finish well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import pandas as pd
import numpy as np
import random
import json

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf_available = True
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

except Exception:
    tf_available = False




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_path = "/kaggle/working/submission.csv"

train_df = pd.read_csv(train_csv_path)
most_common_label = train_df["label"].mode()[0]  # fallback label

train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_image_dir, x)
)




## === cell 2
if tf_available:
    IMG_SIZE = 224
    BATCH_SIZE = 64
    AUTOTUNE = tf.data.AUTOTUNE
    SUBSET_SIZE = len(train_df)  # full set
    VALID_SPLIT = 0.1
    EPOCHS = 30  # unchanged; gives the model more learning time

    subset_df = train_df.sample(n=SUBSET_SIZE, random_state=42)

    val_frac = int(len(subset_df) * VALID_SPLIT)
    val_df = subset_df.iloc[:val_frac]
    train_subset_df = subset_df.iloc[val_frac:]

    def decode_and_resize(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0
        return image, label

    def make_dataset(df):
        paths = df["image_path"].values
        labels = df["label"].values.astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_ds = make_dataset(train_subset_df)
    val_ds = make_dataset(val_df)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    def images_only(ds):
        return ds.map(lambda img, lbl: img, num_parallel_calls=AUTOTUNE)

    train_features = base_model.predict(images_only(train_ds), verbose=0)
    val_features = base_model.predict(images_only(val_ds), verbose=0)

    train_labels = train_subset_df["label"].values.astype(np.int32)
    val_labels = val_df["label"].values.astype(np.int32)

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(train_features.shape[1],)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),  # small regularization boost
            layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_feat_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_feat_ds = train_feat_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_feat_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_feat_ds = val_feat_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    model.fit(train_feat_ds, validation_data=val_feat_ds, epochs=EPOCHS, verbose=2)




## === cell 3
if tf_available:
    test_files = [
        f
        for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    test_paths = [os.path.join(test_image_dir, f) for f in test_files]

    def preprocess_test(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0
        return image

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_features = base_model.predict(test_ds, verbose=0)

    preds = model.predict(test_features, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    submission_df = pd.DataFrame(
        {"image_id": test_files, "label": pred_labels.astype(int)}
    )
else:
    image_predictions = []
    for image_name in os.listdir(test_image_dir):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_predictions.append(
                {"image_id": image_name, "label": int(most_common_label)}
            )
    if not image_predictions:  # fallback to sample submission IDs
        sample_df = pd.read_csv(sample_submission_path)
        image_predictions = [
            {"image_id": row["image_id"], "label": int(most_common_label)}
            for _, row in sample_df.iterrows()
        ]
    submission_df = pd.DataFrame(image_predictions)

submission_df.to_csv(submission_path, index=False)




## === cell 4
submission_df.head()

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

0.8948322756119673

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented a lightweight fix that removes the failing TensorFlow pipeline and replaces it with a simple baseline: the model now predicts the most frequent disease label from the training set for every test image. This resolves import errors, eliminates undefined‑variable issues, and guarantees creation of a correctly formatted `submission.csv` in the expected location.'
- What this solution (achieved 0.61099) has done: 'The fix removes the heavy in‑memory cache and shrinks the shuffle buffer, which cuts RAM pressure and reduces dataset‑pipeline overhead while keeping the same image preprocessing, model, and training epochs. The batch size is increased to 128 to halve the number of training steps without affecting accuracy. These changes keep the core logic identical but make data loading and training fast enough to finish well under the 600‑second limit.'
- What this solution (achieved 0.61099) has done: 'The changes add dataset caching and increase the batch size so that images are read and processed only once and fewer training steps are required, dramatically cutting I/O and CPU time while keeping the same model, training epochs, and evaluation logic.'
- What this solution (achieved 0.61099) has done: 'We add dataset caching so that after the first epoch the already‑decoded and resized images are kept in memory, eliminating the costly file‑I/O and JPEG decoding on the second epoch. This change preserves the exact training logic (same model, epochs, batch size) while cutting the total runtime enough to stay under the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, json
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)

most_common_label = int(train_df["label"].mode()[0])



## === cell 1
try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    print("TensorFlow import failed:", e)
    tf_available = False



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
if tf_available:
    tf.random.set_seed(42)
    np.random.seed(42)

    train_filepaths = (
        train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x)).values
    )
    train_labels = train_df["label"].astype(np.int32).values

    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = 224
    BATCH_SIZE = 64  # reduced to keep memory usage moderate

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0  # normalize to [0,1]
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()  # <-- cache parsed images in memory
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # freeze base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(train_ds, epochs=2, verbose=2)

else:
    from PIL import Image

    SAMPLE_SIZE = 3000  # limit to keep runtime modest
    if len(train_df) > SAMPLE_SIZE:
        train_sample = train_df.sample(SAMPLE_SIZE, random_state=42)
    else:
        train_sample = train_df

    train_paths = (
        train_sample["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x)).values
    )
    train_labels_np = train_sample["label"].astype(np.int32).values

    def _mean_rgb(path):
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32)
            return arr.mean(axis=(0, 1))
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    train_means = np.vstack([_mean_rgb(p) for p in train_paths])



## === cell 3
test_filenames = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
test_filenames.sort()
test_filepaths = [os.path.join(TEST_IMG_DIR, f) for f in test_filenames]

if tf_available:

    def _parse_test(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0
        return image

    test_ds = tf.data.Dataset.from_tensor_slices(test_filepaths)
    test_ds = test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    preds = model.predict(test_ds, verbose=0)
    pred_labels = np.argmax(preds, axis=1).astype(int)
else:
    test_means = np.vstack([_mean_rgb(p) for p in test_filepaths])

    dists = np.linalg.norm(
        train_means[:, np.newaxis, :] - test_means[np.newaxis, :, :], axis=2
    )
    nearest_idx = np.argmin(dists, axis=0)
    pred_labels = train_labels_np[nearest_idx]

submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file created at: {SUBMISSION_PATH}")
print(submission.head())

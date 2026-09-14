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

0.7889090359625265

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0867) has done: 'The update keeps the same model architecture and data handling but speeds up I/O by enabling multiprocessing for the ImageDataGenerators and skips the costly training loop when no GPU is available (the sandbox runs on CPU only). Skipping training only occurs when a pretrained weight file cannot be loaded, preserving the original behavior when weights exist. All other logic, including prediction and submission creation, remains unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.config.threading.set_intra_op_parallelism_threads(tf.data.AUTOTUNE)
tf.config.threading.set_inter_op_parallelism_threads(tf.data.AUTOTUNE)

NUM_WORKERS = min(4, max(1, os.cpu_count() - 1))

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMG_PATH = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
WEIGHT_PATH = "/kaggle/input/model-v16/clf_new_26 (3).h5"  # may not exist

train_df_raw = pd.read_csv(TRAIN_CSV_PATH)
train_df_raw["path"] = train_df_raw["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_PATH, x)
)
train_df_raw["label"] = train_df_raw["label"].astype(str)

label_encoder = {
    lbl: idx for idx, lbl in enumerate(sorted(train_df_raw["label"].unique()))
}
train_df_raw["label_idx"] = train_df_raw["label"].map(label_encoder)

train_df, val_df = train_test_split(
    train_df_raw,
    test_size=0.1,
    stratify=train_df_raw["label"],
    random_state=SEED,
)

BATCH_SIZE = 256  # larger batch reduces steps per epoch


def preprocess_image(path, augment=False):
    """Read, decode, resize and optionally augment an image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    if augment:
        image = tf.image.random_flip_left_right(image, seed=SEED)
        image = tf.image.random_flip_up_down(image, seed=SEED)
        image = tf.image.random_brightness(image, max_delta=0.1, seed=SEED)
        image = (
            tf.image.random_zoom(image, [0.9, 1.1], seed=SEED)
            if hasattr(tf.image, "random_zoom")
            else image
        )
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image


def make_dataset(df, shuffle, augment):
    paths = df["path"].values
    labels = df["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=False)
    ds = ds.map(
        lambda p, l: (preprocess_image(p, augment), tf.one_hot(l, depth=5)),
        num_parallel_calls=NUM_WORKERS,
    )
    ds = ds.cache()  # cache processed images for faster reuse
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, shuffle=True, augment=True)
val_ds = make_dataset(val_df, shuffle=False, augment=False)




## === cell 1
fallback_label = None  # will hold the majority class index if needed

try:
    my_model = load_model(WEIGHT_PATH)
    print("Loaded pretrained model.")
except Exception as e:
    print("Pretrained model not found or failed to load:", e)
    base = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False  # freeze base for a quick training loop

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Starting training on CPU (head only)...")
    try:
        model.fit(
            train_ds,
            epochs=5,
            validation_data=val_ds,
            verbose=2,
        )
        print("Initial training completed. Fine‑tuning the base model...")
        base.trainable = True
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            epochs=3,
            validation_data=val_ds,
            verbose=2,
        )
        print("Fine‑tuning completed.")
        my_model = model
    except Exception as train_err:
        print("Training failed:", train_err)
        fallback_label = int(train_df_raw["label_idx"].mode()[0])
        my_model = None
        print(
            f"Falling back to majority class label {fallback_label} for all predictions."
        )
    try:
        if my_model is not None:
            my_model.save("tmp_cassava_model.h5")
    except Exception:
        pass




## === cell 2
test_images = glob.glob(os.path.join(TEST_IMG_PATH, "*.jpg"))
df_test = pd.DataFrame({"path": test_images})


def make_test_dataset(df):
    paths = df["path"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: preprocess_image(p, augment=False), num_parallel_calls=NUM_WORKERS
    )
    ds = ds.cache()  # cache test images for faster inference
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(df_test)




## === cell 3
if my_model is not None:
    preds = my_model.predict(
        test_ds,
        verbose=1,
    )
    pred_labels = np.argmax(preds, axis=1)
else:
    pred_labels = np.full(len(df_test), fallback_label, dtype=int)

submission = pd.DataFrame(
    {
        "image_id": df_test["path"].apply(lambda p: os.path.basename(p)),
        "label": pred_labels,
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written. Sample:")
print(submission.head())

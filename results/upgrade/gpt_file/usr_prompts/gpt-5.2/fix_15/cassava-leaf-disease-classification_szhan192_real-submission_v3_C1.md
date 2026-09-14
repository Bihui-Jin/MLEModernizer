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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.609398609851919

# 6. Current score

0.75299

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.75972) has done: 'You’re failing because the notebook expects a pre-trained `.h5` model in `/kaggle/input/tryagain`, but that dataset isn’t present—so `model` never loads and inference crashes. I remove the brittle external-model dependency and instead train the same kind of Keras image classifier end-to-end using the provided Cassava `train_images/` and `train.csv`, then run predictions on `test_images/`. To keep changes minimal and stable, I use a standard transfer-learning Keras model (ImageNet backbone + small head), correct preprocessing, and ensure the submission exactly matches `sample_submission.csv` ordering and columns. This run within the Kaggle environment and reliably write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.75972) has done: 'The crash happens immediately because importing `google.protobuf` triggers an incompatible protobuf API (`MessageFactory.GetPrototype`) in this Kaggle environment; that import isn’t needed for training/inference, so I remove it and keep the TensorFlow version print. Since your current score (0.75972) is already much higher than the target (0.6094), I avoid any score-improving changes and only make stability fixes. I also remove an unused `image_dataset_from_directory` creation that can waste time/resources, but keep the exact same data pipeline, model, training loop, and submission formatting. The script run end-to-end and write `/kaggle/working/submission.csv` with the required columns and row order.'
- What this solution (achieved 0.2201) has done: 'Your current run isn’t yielding a Kaggle score because it can fail before training/inference due to the protobuf downgrade/restart logic; TensorFlow 2.18 in this environment already ships with a compatible protobuf, so I remove that brittle install+execv block to guarantee an end-to-end run and a valid `submission.csv`. To move accuracy upward (higher-is-better) toward your target (0.6094) with minimal semantic changes, I keep the same EfficientNetB0 transfer-learning setup but add standard, lightweight image augmentation and label smoothing—both typically improve generalization on Cassava without changing the overall training approach. I also add `.cache()` to the tf.data pipelines for stability/speed (no approximation) and keep the submission aligned exactly to `sample_submission.csv` order and columns. All paths remain unchanged and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.2201) has done: 'I fix the two runtime failures without changing the core model/training approach: (1) the protobuf `MessageFactory.GetPrototype` crash triggered during `import tensorflow` by forcing TensorFlow to use the pure-Python protobuf implementation, and (2) the `label_smoothing` argument error by switching to the TF-supported `SparseCategoricalCrossentropy` API in this environment. I keep the same EfficientNetB0 transfer-learning pipeline, augmentation, splits, and training loop. Finally, I ensure the script always writes `/kaggle/working/submission.csv` with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.75336) has done: 'I fix the TensorFlow import crash caused by forcing the pure-Python protobuf implementation; in this environment TF 2.18 expects the default C++ protobuf, so removing that env override restores a clean import. Next, I fix the `SparseCategoricalCrossentropy(label_smoothing=...)` incompatibility by switching to `CategoricalCrossentropy` with one-hot labels, preserving the same model/training approach and keeping label smoothing behavior (which should help move accuracy up toward your 0.609 target). I make the minimal necessary dataset change to output one-hot labels and keep the rest of the pipeline (EfficientNetB0 backbone, augmentation, training loop, and submission formatting) intact. The script then run end-to-end and always write `/kaggle/working/submission.csv` in the required format.'
- What this solution (achieved 0.75336) has done: 'The failure happens at `import tensorflow` due to a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. To fix it without changing your model/training logic, I force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the incompatible C++ protobuf path. I keep the EfficientNetB0 transfer-learning pipeline, one-hot labels + label smoothing, training loop, and submission formatting unchanged, so score behavior should remain essentially the same while the notebook runs end-to-end and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.75336) has done: 'The crash is caused by forcing pure-Python protobuf (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`), which is incompatible with TensorFlow 2.18 in this environment and triggers the `MessageFactory.GetPrototype` error during `import tensorflow`. I remove that env override (keeping everything else the same) so TensorFlow imports cleanly and the pipeline runs end-to-end. Since your current score (0.75336) is already well above the target (0.6094), I not make any model/training changes that would further improve performance; the fix is intended to be score-neutral and purely for correctness/stability. The script still train the same EfficientNetB0-based classifier and write `/kaggle/working/submission.csv` in the exact required format.'
- What this solution (achieved 0.75299) has done: 'Your current code doesn’t yield a Kaggle score because it can restart the kernel and/or break TensorFlow by downgrading `protobuf`, so I remove that install+execv “protobuf fix” block and keep TensorFlow’s environment intact. To move accuracy upward toward your target with minimal semantic change, I make the train/valid split and shuffle operate on `image_id` groups (avoids any accidental leakage/duplication issues) and add standard normalization safety (`clip_by_value`) while keeping the same EfficientNetB0 model, loss, augmentation, epochs, and submission formatting. I also ensure the submission is always written to `/kaggle/working/submission.csv` and matches `sample_submission.csv` row order exactly. These are stability/correctness tweaks that typically improve (or at least stop harming) accuracy without changing the core approach.'
- What this solution (achieved 0.75299) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring we do not import the standalone `protobuf` package (which can shadow TensorFlow’s bundled protobuf) and by removing `/kaggle/input` from `sys.path` before importing TensorFlow, since Kaggle datasets sometimes contain conflicting modules. This is a runtime-stability fix and should be score-neutral (it doesn’t change the model/training/inference logic). I also keep all paths and the EfficientNetB0 training/inference pipeline intact so the notebook runs end-to-end and always writes `/kaggle/working/submission.csv` in the required format.'
- What this solution (achieved 0.75299) has done: 'The crash happens before any training because TensorFlow’s protobuf dependency is mismatched at import time, causing `MessageFactory.GetPrototype` to be missing. I fix this by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (and clearing any already-imported protobuf modules), which is the minimal change needed to make the pipeline run end-to-end. Since your current score (0.75299) is already well above the target (0.6094), I not change the model/training/inference logic or hyperparameters to avoid further score shifts. The script then train and always write a valid `/kaggle/working/submission.csv` in the exact required format and row order.'
- What this solution (achieved 0.75299) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment variables, which are incompatible with TF 2.18 here and directly cause `MessageFactory.GetPrototype` to be missing. I keep the same EfficientNetB0 transfer-learning model, preprocessing, augmentation, training loop, and submission formatting to avoid unnecessary score shifts (your current score is already above the target, so no score-improving changes are needed). I also keep the existing paths and ensure the script always writes `/kaggle/working/submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import numpy as np
import pandas as pd
from tensorflow import keras

SEED = 42
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled (ok):", repr(e))

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train_df:", train_df.shape, "sample_sub:", sample_sub.shape)
print(train_df.head())
print(sample_sub.head())



## === cell 2
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep modest to fit 600s; no early stopping or approximation used

train_df = train_df.copy()
train_df = train_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)
assert (
    train_df["filepath"].map(os.path.exists).all()
), "Some training image paths do not exist."

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

print("Train split:", tr_df.shape, "Valid split:", va_df.shape)


def make_ds(df, training):
    paths = df["filepath"].values
    labels = df["label"].values.astype("int32")
    labels_oh = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_oh))

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32)

        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.10, seed=SEED)
            img = tf.image.random_contrast(img, lower=0.90, upper=1.10, seed=SEED)

        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


tr_ds = make_ds(tr_df, training=True)
va_ds = make_ds(va_df, training=False)



## === cell 3
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    tr_ds,
    validation_data=va_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 4
test_paths = (
    sample_sub["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x)).values
)
assert all(os.path.exists(p) for p in test_paths), "Some test image paths do not exist."

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": preds}
)

out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(my_submission.head())
print("Submission shape:", my_submission.shape)
assert my_submission.shape[0] == sample_sub.shape[0]
assert list(my_submission.columns) == ["image_id", "label"]

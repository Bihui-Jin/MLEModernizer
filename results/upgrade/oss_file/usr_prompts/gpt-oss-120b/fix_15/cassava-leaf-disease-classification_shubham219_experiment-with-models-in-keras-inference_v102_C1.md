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

3.11

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

0.6589604110003022

# 6. Current score

0.34006

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0639) has done: 'I remove the unused tensorflow_hub import that triggers a protobuf error, replace the missing pre‑trained .h5 model with a lightweight ResNet‑50 model built from ImageNet weights, and add a short training loop using the provided train.csv and image folder. Then I generate predictions on the test images with the same preprocessing, create the required ``submission.csv`` file, and ensure all variables are defined so the notebook runs end‑to‑end without errors.'
- What this solution (achieved 0.1349) has done: 'The changes add lightweight TensorFlow configuration (thread limits and mixed‑precision) and enable multiprocessing data loading during model training. These adjustments keep the exact model architecture, augmentation, and training epochs, while reducing I/O and compute overhead, allowing the script to finish well within the 600‑second limit without affecting result accuracy.'
- What this solution (achieved 0.2074) has done: 'The script is sped up by (1) increasing the data‑loading parallelism (workers = 8) and (2) using a larger batch size (128) which cuts the number of training and inference steps roughly in half while keeping the same model, epochs, and preprocessing. All other logic—including the ResNet‑50 architecture, augmentation, and label handling—remains unchanged, so the results stay equivalent.'
- What this solution (achieved 0.11099) has done: 'We replace the slow ImageDataGenerator pipeline with a native tf.data pipeline that does image loading, resizing, ResNet‑50 preprocessing, and the same augmentations (random flip, rotation ±15°, translation ±10%) using TensorFlow’s built‑in preprocessing layers. tf.data runs in parallel, batches efficiently, and prefetches, cutting the data‑loading overhead dramatically while keeping the model architecture, loss, optimizer, and training epochs unchanged. The test‑time pipeline is similarly streamlined, preserving the original order for correct submission generation.'
- What this solution (achieved 0.12818) has done: 'The changes fix the TensorFlow dataset shuffle call (using the correct argument name), reduce the batch size for more stable training, and add a checkpoint callback that saves the best‑validation model and restores it before inference. These fixes allow the script to run end‑to‑end and improve validation accuracy, moving the score toward the target while preserving the original architecture and training logic.'
- What this solution (achieved 0.34006) has done: 'We fix the TensorFlow dataset shuffle argument (use buffer_size instead of buffer) and ensure the training dataset is created before fitting. Wealso lower the epoch count to keep runtime safe and add a modest early‑stopping callback, preserving the original model architecture and preprocessing. These minimal changes resolve the runtime errors and should improve validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_SUPPORTED"] = "1"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    stratify=df_train["label"],
    random_state=SEED,
)

IMG_SIZE = 256
BATCH_SIZE = 64  # smaller batch for more stable training
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 5

preprocess_fn = resnet50.preprocess_input

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.0416667),  # ≈15°
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]
)


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_fn(image)
    label = tf.strings.to_number(label, out_type=tf.int32)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label


def load_and_preprocess_test(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_fn(image)
    return image


train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (train_df["path"].values, train_df["label"].values)
    )
    .shuffle(buffer_size=len(train_df), seed=SEED, reshuffle_each_iteration=True)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .map(
        lambda img, lbl: (augmenter(img, training=True), lbl),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_df["path"].values, val_df["label"].values))
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 2
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 3
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    save_weights_only=True,
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stop_cb],
    verbose=2,
)

if os.path.exists("best_model.h5"):
    model.load_weights("best_model.h5")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3567528510.py in <cell line: 0>()
----> 1 checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
      2     "best_model.h5",
      3     monitor="val_accuracy",
      4     save_best_only=True,
      5     mode="max",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=best_model.h5

## === cell 4
test_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
df_test = pd.DataFrame(test_paths, columns=["path"])

test_ds = (
    tf.data.Dataset.from_tensor_slices(df_test["path"].values)
    .map(load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(preds, axis=1)

submission = pd.DataFrame(
    {
        "image_id": df_test["path"].apply(lambda p: os.path.basename(p)),
        "label": pred_labels.astype(int),
    }
)

submission.to_csv("submission.csv", index=False)

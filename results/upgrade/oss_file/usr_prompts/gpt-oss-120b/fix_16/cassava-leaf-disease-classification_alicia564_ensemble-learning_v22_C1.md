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

0.9073738289513448

# 6. Current score

0.5441

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The changes add mixed‑precision training (speeding up GPU compute), enable parallel data loading in `model.fit`, and replace the per‑image prediction loop with a single batched prediction, which together keep the model architecture and training semantics identical while fitting comfortably inside the 600 s limit.'
- What this solution (achieved 0.05531) has done: 'The changes speed up data loading and training by increasing the batch size to 64, enabling parallel data loading with multiple workers, and passing these settings to `model.fit`. This reduces the number of steps per epoch and utilizes CPU cores for preprocessing, cutting total runtime while keeping the model architecture, loss, and training schedule unchanged.'
- What this solution (achieved 0.0852) has done: 'The changes replace the Python‑level ImageDataGenerator pipelines with an efficient `tf.data` pipeline that reads, decodes, augments, and batches images using TensorFlow’s native ops and prefetching. This eliminates the heavy multiprocessing overhead of `flow_from_dataframe` while keeping the same model, epochs, and augmentations (rotation, shifts, zoom, flips). The label encoding is done once with scikit‑learn and the dataset yields one‑hot vectors, preserving the original training semantics. All other logic, including model definition, mixed‑precision usage, callbacks, and submission creation, remains unchanged.'
- What this solution (achieved 0.09417) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission is generated:

- **Cell 0:** Removed the protocol‑buffers environment override that caused an import‑time `AttributeError`.
- **Cell 1:** Corrected the `shuffle` call signature (`buffer` → `buffer_size`) so the dataset pipeline builds correctly.
- **Cell 1‑2:** Minor reordering to guarantee `train_ds` and `valid_ds` are defined before model training.
- **Cell 3:** No logic changes; the prediction and CSV export now run on a successfully trained model.'
- What this solution (achieved 0.5441) has done: 'The fix keeps the exact model, training loops, and evaluation logic but reduces memory pressure and I/O cost during data loading. The `decode_image` function now casts images to `float16` (matching the mixed‑precision policy) before they are cached, halving the RAM needed for the cached tensors. The batch size is reduced to 128 so each batch fits comfortably in memory, preventing swapping that caused the 10‑minute timeout. All other code—including augmentation, model architecture, and callbacks—remains unchanged, preserving result accuracy.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tf.config.optimizer.set_jit(True)

try:
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)
train_csv["label"] = train_csv["label"].astype(str)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

le = LabelEncoder()
le.fit(train_csv["label"])
train_labels_int = le.transform(train_df["label"])
valid_labels_int = le.transform(valid_df["label"])

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128  # reduced to lower memory pressure
IMG_SIZE = (224, 224)
NUM_CLASSES = 5


def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    img = tf.cast(
        img, tf.float16
    )  # cast to float16 to match mixed precision and halve cache size
    return img


def augment(image):
    image = tf.image.random_flip_left_right(image, seed=42)
    image = tf.image.random_flip_up_down(image, seed=42)
    image = tf.image.random_brightness(image, max_delta=0.1, seed=42)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=42)
    return image


train_raw_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["path"].values, train_labels_int)
)

train_ds = (
    train_raw_ds.map(lambda p, l: (decode_image(p), l), num_parallel_calls=AUTOTUNE)
    .cache()  # cache decoded (and now half‑size) images in RAM
    .map(
        lambda img, l: (
            augment(img),
            tf.one_hot(l, depth=NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    .shuffle(buffer_size=1024, seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_df["path"].values, valid_labels_int))
    .map(
        lambda p, l: (
            decode_image(p),
            tf.one_hot(l, depth=NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 2
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3))
)
base_model.trainable = False  # freeze backbone for the first stage

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation="softmax", dtype="float32")(x)  # ensure float32 logits
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
lr_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

model.fit(
    train_ds,
    epochs=12,
    validation_data=valid_ds,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=12,
    validation_data=valid_ds,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2351568861.py in <cell line: 0>()
     21 )
     22 
---> 23 model.fit(
     24     train_ds,
     25     epochs=12,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node adjust_contrast defined at (most recent call last):
<stack traces unavailable>
No OpKernel was registered to support Op 'AdjustContrastv2' used by {{node adjust_contrast}} with these attrs: [T=DT_HALF]
Registered devices: [CPU]
Registered kernels:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]

	 [[adjust_contrast]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16395]

## === cell 3
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(test_dir, f) for f in test_filenames]

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(
        lambda p: preprocess_input(
            tf.cast(
                tf.image.resize(
                    tf.image.decode_jpeg(tf.io.read_file(p), channels=3), IMG_SIZE
                ),
                tf.float16,
            )
        ),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)
pred_classes = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_classes})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created at {submission_path}")
print(submission_df.head())

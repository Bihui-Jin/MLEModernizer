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

0.8170142036869145

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script was failing because it tried to load a non‑existent SavedModel with `TFSMLayer`, which also triggered protobuf errors. I removed that part, added a lightweight EfficientNet‑B0 classifier that trains briefly on the provided training images, and then uses it for inference on the test set. The new pipeline creates a proper submission CSV matching the required format.'
- What this solution (achieved 0.61099) has done: 'The script is now given multiprocessing workers for image loading and for the fit calls, which removes the single‑threaded bottleneck when reading and augmenting thousands of JPEGs. By passing workers=4, use_multiprocessing=True, and max_queue_size=20 to flow_from_dataframe and the fit functions, data‑preprocessing runs in parallel while the model architecture, training schedule, and all other logic remain unchanged.'
- What this solution (achieved 0.61099) has done: 'We replace the slow Keras ImageDataGenerator pipelines with tf.data datasets that read, resize, rescale and (lightly) augment images using TensorFlow’s native ops, then batch and prefetch them. This keeps the model architecture, loss and training loops unchanged while removing the Python‑level per‑image overhead that caused the 10‑minute timeout. All other cells remain the same, preserving the original semantics and results.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf.message_factory import MessageFactory

    def _get_prototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)




## === cell 2
img_size = (224, 224)  # EfficientNetB0 default size
batch_size = 64
num_classes = 5

tf.random.set_seed(42)

label_names = sorted(train_df["label"].unique())
label_to_index = {name: idx for idx, name in enumerate(label_names)}

train_paths = (
    train_df["image_id"].apply(lambda x: os.path.join(train_img_dir, x)).values
)
train_labels = train_df["label"].map(label_to_index).values.astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))


def _load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # rescale

    img = tf.image.random_flip_left_right(img, seed=42)

    label_onehot = tf.one_hot(label, depth=num_classes)
    return img, label_onehot


train_ds = train_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer=1024, seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2651270427.py in <cell line: 0>()
     36 
     37 train_ds = train_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
---> 38 train_ds = train_ds.shuffle(buffer=1024, seed=42, reshuffle_each_iteration=True)
     39 train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
     40 

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 3
base_model = keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=img_size + (3,)
)
base_model.trainable = False  # freeze backbone initially

model = keras.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 4
epochs_initial = 5  # increased from 3
steps_per_epoch = int(np.ceil(len(train_df) / batch_size))

model.fit(
    train_ds,
    epochs=epochs_initial,
    steps_per_epoch=steps_per_epoch,
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2722273147.py in <cell line: 0>()
      2 steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
      3 
----> 4 model.fit(
      5     train_ds,
      6     epochs=epochs_initial,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(224, 224, 3), dtype=float32). Expected shape (None, 224, 224, 3), but input has incompatible shape (224, 224, 3)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(224, 224, 3), dtype=float32)
  • training=True
  • mask=None

## === cell 5
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs_finetune = 5  # increased from 2
model.fit(
    train_ds,
    epochs=epochs_finetune,
    steps_per_epoch=steps_per_epoch,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1727941523.py in <cell line: 0>()
      7 
      8 epochs_finetune = 5  # increased from 2
----> 9 model.fit(
     10     train_ds,
     11     epochs=epochs_finetune,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(224, 224, 3), dtype=float32). Expected shape (None, 224, 224, 3), but input has incompatible shape (224, 224, 3)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(224, 224, 3), dtype=float32)
  • training=True
  • mask=None

## === cell 6
test_filenames = sorted(
    [f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(test_img_dir, f) for f in test_filenames]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img


test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)




## === cell 7
pred_probs = model.predict(test_ds, verbose=1)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = pred_indices.astype(str).tolist()




## === cell 8
submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

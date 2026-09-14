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

3.10

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

0.6024478694469628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18572) has done: 'The changes keep the same model, data splits, and training schedule, but eliminate an unnecessary extra scaling step in the image generators and enable multiprocessing data loading, which markedly reduces the I/O‑and‑preprocessing overhead that caused the 10‑minute timeout.  Removing the duplicate `rescale=1.0/255` does not alter the final pixel values because `tf.keras.applications.inception_v3.preprocess_input` already performs the required scaling.  Adding `workers` and `use_multiprocessing=True` to `model.fit` speeds up batch preparation without changing any training logic or accuracy.'
- What this solution (achieved 0.13378) has done: 'The changes add parallel data loading by setting worker processes for the ImageDataGenerator flows and for model fitting, which removes the main thread bottleneck of reading and preprocessing images from disk. This keeps the exact same model, training schedule, and augmentation logic, so results remain unchanged while reducing total runtime.'
- What this solution (achieved 0.24402) has done: 'The changes increase data‑loading parallelism (more workers, enable multiprocessing) and explicitly set `steps_per_epoch` and `validation_steps` so TensorFlow does not waste time iterating over empty batches. This keeps the same model, augmentations, training loops, and early‑stopping logic, but speeds up each epoch by reducing I/O bottlenecks.'

# 9. Code solution

## === cell 0
import os, warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
tf.random.set_seed(2020)
try:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
except Exception:
    pass

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMIT_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)

BATCH_SIZE = 64
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
EPOCHS = 30  # a few more epochs for better learning
N_SPLITS = 1




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.AUTOTUNE


def _parse_image(filename, label=None, augment=False):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_zoom(img, [0.8, 1.2])
        img = tf.image.random_shift(img, 0.1, 0.1)
    if label is None:
        return img
    else:
        return img, label


def make_dataset(df, shuffle=False, augment=False):
    filepaths = tf.strings.join([TRAIN_IMG_DIR, "/", df["image_id"]])
    int_labels = tf.cast(df["label"], tf.int32)
    one_hot = tf.one_hot(int_labels, depth=5, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, one_hot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=2020, reshuffle_each_iteration=False)
    ds = ds.map(lambda f, l: _parse_image(f, l, augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def make_test_dataset(df):
    filepaths = tf.strings.join([TEST_IMG_DIR, "/", df["image_id"]])
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(
        lambda f: _parse_image(f, label=None, augment=False),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds




## === cell 2
def create_inception():
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    inputs = Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(5, activation="softmax", dtype="float32")(x)
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model




## === cell 3
split = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.1, random_state=2020)
for train_idx, val_idx in split.split(df_train["image_id"], df_train["label"]):
    train_df = df_train.iloc[train_idx]
    val_df = df_train.iloc[val_idx]
    train_ds = make_dataset(train_df, shuffle=True, augment=True)
    val_ds = make_dataset(val_df, shuffle=False, augment=False)

    steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

    model = create_inception()
    checkpoint_path = "/kaggle/working/inception_best.h5"
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )
    model = load_model(checkpoint_path)
    break  # only one split




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_11/2904361845.py in <cell line: 0>()
      3     train_df = df_train.iloc[train_idx]
      4     val_df = df_train.iloc[val_idx]
----> 5     train_ds = make_dataset(train_df, shuffle=True, augment=True)
      6     val_ds = make_dataset(val_df, shuffle=False, augment=False)
      7 

/tmp/ipykernel_11/2634529571.py in make_dataset(df, shuffle, augment)
     24     filepaths = tf.strings.join([TRAIN_IMG_DIR, "/", df["image_id"]])
     25     # Convert string labels to integer then one‑hot
---> 26     int_labels = tf.cast(df["label"], tf.int32)
     27     one_hot = tf.one_hot(int_labels, depth=5, dtype=tf.float32)
     28     ds = tf.data.Dataset.from_tensor_slices((filepaths, one_hot))

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnimplementedError: {{function_node __wrapped__Cast_device_/job:localhost/replica:0/task:0/device:CPU:0}} Cast string to int32 is not supported [Op:Cast] name: 

## === cell 4
sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV)
test_ds = make_test_dataset(sample_submit)

pred_probs = model.predict(
    test_ds,
    verbose=1,
)

pred_labels = np.argmax(pred_probs, axis=1)

sample_submit["label"] = pred_labels
output_path = os.path.join("/kaggle/working", "submission.csv")
sample_submit.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1914823985.py in <cell line: 0>()
      2 test_ds = make_test_dataset(sample_submit)
      3 
----> 4 pred_probs = model.predict(
      5     test_ds,
      6     verbose=1,

NameError: name 'model' is not defined

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

0.1000302206104563

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.30306) has done: 'The timeout is overwhelmingly dominated by slow Python-side image loading/augmentation via `ImageDataGenerator.flow_from_dataframe` and by running validation every epoch with deterministic ops enabled. To preserve identical model/training logic while speeding up, I switch the input pipeline to a cached, prefetched `tf.data` pipeline that performs the same Keras preprocessing/augmentation ops, and I keep the same train/validation split and steps. I also remove unnecessary per-epoch Python overhead by using `tf.data` deterministically with parallel mapping + AUTOTUNE prefetch, and I keep all seeds/determinism settings intact. Model architecture, optimizer, loss, epochs, steps, checkpointing, and prediction semantics remain the same.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import InceptionResNetV2

plt.rcParams["figure.figsize"] = (17, 6)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

TFDATA_CACHE_DIR = "./tfdata_cache"
os.makedirs(TFDATA_CACHE_DIR, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAINING_DIR = "../input/cassava-leaf-disease-classification/train_images"
TRAINING_CSV = "../input/cassava-leaf-disease-classification/train.csv"
JSON_LABELS = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"

TRAIN_TFREC_DIR = "../input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "../input/cassava-leaf-disease-classification/test_tfrecords"




## === cell 2
train_df = pd.read_csv(TRAINING_CSV)
train_df["label"] = train_df["label"].astype(
    "string"
)  # required by flow_from_dataframe for categorical class_mode
train_df.head()




## === cell 3
total_images_count = len(train_df.index)
total_train_img_count = int(total_images_count * 0.8)
total_val_img_count = total_images_count - total_train_img_count
print("Expected images counts:")
print(f"\nTotal Images from original directory: {total_images_count}")
print(f"Training Images: {total_train_img_count}")
print(f"Validation Images: {total_val_img_count}")




## === cell 4
label_df = pd.read_json(JSON_LABELS, orient="index")
label_df = label_df.values.flatten().tolist()
label_df




## === cell 5
print("Per-class counts:", train_df["label"].value_counts().sort_index().tolist())




## === cell 6
if False:
    training_images_dir = os.path.join(TRAINING_DIR, "*.jpg")
    print(training_images_dir)
    training_images = glob.glob(training_images_dir)

    if len(training_images) > 0:
        plt.figure(figsize=(12, 12))
        for i in range(1, 10):
            training_image = np.random.choice(training_images)
            training_image_RGB = cv2.imread(training_image)[..., ::-1]
            plt.subplot(3, 3, i)
            plt.imshow(training_image_RGB)
            plt.axis("off")
        plt.show()




## === cell 7
training_datagen = None
validation_datagen = None




## === cell 8
BATCH_SIZE = 48
IMG_WIDTH = 300
IMG_HEIGHT = 300
CHANNEL = 3

train_df_local = train_df.copy()
labels_int = train_df_local["label"].astype("int32").to_numpy()
image_ids = train_df_local["image_id"].to_numpy()

rng = np.random.RandomState(SEED)
indices = np.arange(len(train_df_local))
rng.shuffle(indices)
val_size = int(round(0.2 * len(indices)))
val_idx = indices[:val_size]
train_idx = indices[val_size:]

train_ids = image_ids[train_idx]
train_labels = labels_int[train_idx]
val_ids = image_ids[val_idx]
val_labels = labels_int[val_idx]

n_train = len(train_ids)
n_val = len(val_ids)

print("\nTraining Dataset")
print(f"Found {n_train} training images belonging to 5 classes.")
print("\nValidation Dataset")
print(f"Found {n_val} validation images belonging to 5 classes.")

class_indices = {str(i): i for i in range(5)}
print("\nClass Indices:")
print(class_indices)

train_id_set = tf.constant(train_ids)
val_id_set = tf.constant(val_ids)

train_ids_key = tf.constant(train_ids, dtype=tf.string)
train_labels_val = tf.constant(train_labels, dtype=tf.int32)
val_ids_key = tf.constant(val_ids, dtype=tf.string)
val_labels_val = tf.constant(val_labels, dtype=tf.int32)

train_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(train_ids_key, train_labels_val),
    default_value=tf.constant(-1, tf.int32),
)
val_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids_key, val_labels_val),
    default_value=tf.constant(-1, tf.int32),
)

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=100.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomBrightness(factor=0.4, value_range=(0.0, 1.0), seed=SEED),
    ],
    name="augmenter",
)

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _one_hot(label):
    return tf.one_hot(label, depth=5, dtype=tf.float32)


@tf.function
def _augment(img, label):
    img = augmenter(img, training=True)
    return img, label


def _tfrecord_dataset(filenames, training):
    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(options)

    def _parse(example_proto):
        ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
        img = _decode_resize_from_bytes(ex["image"])
        image_id = ex["image_id"]
        label = ex["label"]
        return img, image_id, tf.cast(label, tf.int32)

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)

    if training:

        def _is_in_train(img, image_id, label):
            lbl = train_label_table.lookup(image_id)
            return tf.greater_equal(lbl, 0)

        def _attach_train_label(img, image_id, label):
            lbl = train_label_table.lookup(image_id)
            return img, _one_hot(lbl)

        ds = ds.filter(_is_in_train).map(
            _attach_train_label, num_parallel_calls=AUTOTUNE
        )
    else:

        def _is_in_val(img, image_id, label):
            lbl = val_label_table.lookup(image_id)
            return tf.greater_equal(lbl, 0)

        def _attach_val_label(img, image_id, label):
            lbl = val_label_table.lookup(image_id)
            return img, _one_hot(lbl)

        ds = ds.filter(_is_in_val).map(_attach_val_label, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

train_dataset = _tfrecord_dataset(train_tfrecs, training=True)
validation_dataset = _tfrecord_dataset(train_tfrecs, training=False)




## === cell 9
class theCallBacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (logs.get("val_accuracy", 0) > 0.92) and (logs.get("accuracy", 0) > 0.92):
            print(
                "\nTraining Accuracy> 0.92 & Validation Accuracy> 0.92\nCancelling training!"
            )
            self.model.stop_training = True


callback_on_metrics = theCallBacks()

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    cooldown=1,
    min_lr=0.0001,
)




## === cell 10
loss_func = CategoricalCrossentropy()




## === cell 11
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)




## === cell 12
new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))




## === cell 13
DROPOUT_RATE = 0.5

base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=new_input,
    pooling="avg",
)

for layer in base_model.layers:
    layer.trainable = False

x = Dropout(DROPOUT_RATE)(base_model.output)
x = Dense(4096, activation="relu", name="fc6")(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation="relu", name="fc7")(x)
x = Dropout(DROPOUT_RATE)(x)

out = Dense(5, activation="softmax", name="classifier")(x)
model = Model(inputs=base_model.input, outputs=out)

SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001
sgd = tf.keras.optimizers.SGD(
    learning_rate=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True
)

model.compile(
    optimizer=sgd,
    loss=loss_func,
    metrics=["accuracy"],
    steps_per_execution=32,
)
model.summary()




## === cell 14
num_epochs = 10

steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))
validation_steps = int(np.ceil(n_val / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 15
model_checkpoint_path = "./best_model.keras"
checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint,
        reduce_lr,
        callback_on_metrics,
    ],
    verbose=1,
)

model = tf.keras.models.load_model(model_checkpoint_path)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1422399974.py in <cell line: 0>()
      8 )
      9 
---> 10 history = model.fit(
     11     train_dataset,
     12     epochs=num_epochs,

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

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::ShuffleAndRepeat::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNextAsOptional]] [Op:__inference_multi_step_on_iterator_36401]

## === cell 16
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)


def _test_tfrecord_dataset(filenames):
    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(options)

    def _parse_test(example_proto):
        ex = tf.io.parse_single_example(
            example_proto,
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "image_id": tf.io.FixedLenFeature([], tf.string),
            },
        )
        img = _decode_resize_from_bytes(ex["image"])
        return img, ex["image_id"]

    ds = ds.map(_parse_test, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds_with_ids = _test_tfrecord_dataset(test_tfrecs)

test_images_ds = test_ds_with_ids.map(
    lambda x, image_id: x, num_parallel_calls=AUTOTUNE
)
test_ids_ds = test_ds_with_ids.map(
    lambda x, image_id: image_id, num_parallel_calls=AUTOTUNE
)

test_ids = []
for b in test_ids_ds:
    test_ids.append(b.numpy())
test_ids = np.concatenate(test_ids, axis=0).astype("U")

test_steps = int(np.ceil(len(sample_submission) / BATCH_SIZE))

probs = model.predict(
    test_images_ds,
    steps=test_steps,
    verbose=1,
)

predicted = np.argmax(probs, axis=1)[: len(sample_submission)].astype(int)

pred_map = dict(zip(test_ids[: len(predicted)], predicted.tolist()))
ordered_pred = sample_submission["image_id"].map(pred_map).astype(int).tolist()

submission = pd.DataFrame(
    {"image_id": sample_submission.image_id, "label": ordered_pred}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1217383570.py in <cell line: 0>()
     48 # Materialize ids in the same order as predict consumes batches (deterministic pipeline).
     49 test_ids = []
---> 50 for b in test_ids_ds:
     51     test_ids.append(b.numpy())
     52 test_ids = np.concatenate(test_ids, axis=0).astype("U")

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:

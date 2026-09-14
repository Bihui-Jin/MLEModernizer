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

0.7842248413417952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras import Model

print("TensorFlow:", tf.__version__)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images/"
test_images_dir_data_path = data_path + "test_images/"
train_tfrecords_dir = data_path + "train_tfrecords/"
test_tfrecords_dir = data_path + "test_tfrecords/"




## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 4
train_csv.head()




## === cell 5
BATCH_SIZE = 18
IMG_SIZE = 224
SEED = 42

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

_WORKERS = max(2, (os.cpu_count() or 4) // 2)
_USE_MPROC = True
_MAX_QSIZE = 32




## === cell 6
NUM_CLASSES = 5

df_all = train_csv.copy()
df_all["image_id"] = df_all["image_id"].astype(str)
df_all = df_all.sort_values("image_id", kind="mergesort").reset_index(drop=True)

n_total = len(df_all)
n_val = int(np.floor(0.1 * n_total))
n_train = n_total - n_val

df_train = df_all.iloc[:n_train].reset_index(drop=True)
df_val = df_all.iloc[n_train:].reset_index(drop=True)

y_train = df_train["label"].astype(np.int32).values
y_val = df_val["label"].astype(np.int32).values

x_train_ids = df_train["image_id"].values
x_val_ids = df_val["image_id"].values

AUTOTUNE = tf.data.AUTOTUNE


def _feature_description():
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }


def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label = tf.cast(ex["target"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


def _parse_tfrecord_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description())
    name = ex["image_name"]
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label = tf.cast(ex["target"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return name, img, label_oh


def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=20.0 / 360.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


def _augment(img, label):
    return augmenter(img, training=True), label


options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.apply_default_optimizations = True

train_tfrec_files = sorted(
    tf.io.gfile.glob(os.path.join(train_tfrecords_dir, "*.tfrec"))
)
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(test_tfrecords_dir, "*.tfrec")))

if not train_tfrec_files:
    raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecords_dir}")
if not test_tfrec_files:
    raise FileNotFoundError(f"No TFRecord files found in: {test_tfrecords_dir}")

val_id_set = tf.constant(x_val_ids.tolist(), dtype=tf.string)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        val_id_set, tf.ones_like(val_id_set, dtype=tf.int32)
    ),
    default_value=0,
)


def _is_val(name, img, label_oh):
    return tf.equal(val_table.lookup(name), 1)


def _drop_name(name, img, label_oh):
    return img, label_oh


full_ds = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)
full_ds = full_ds.map(_parse_tfrecord_with_name, num_parallel_calls=AUTOTUNE)

val_ds = full_ds.filter(_is_val).map(_drop_name, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_ds = full_ds.filter(
    lambda name, img, label_oh: tf.logical_not(_is_val(name, img, label_oh))
)
train_ds = train_ds.map(_drop_name, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_steps = int(np.ceil(len(x_train_ids) / BATCH_SIZE))
valid_steps = int(np.ceil(len(x_val_ids) / BATCH_SIZE))

base = applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # train head first; keeps runtime low and stable

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model_model = Model(inputs=base.input, outputs=outputs)

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

EPOCHS_HEAD = 3
history1 = model_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=valid_steps,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

EPOCHS_FT = 1
history2 = model_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=valid_steps,
    epochs=EPOCHS_FT,
    verbose=1,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_10/2603760845.py in <cell line: 0>()
     99 options.experimental_optimization.map_parallelization = True
    100 options.experimental_optimization.parallel_batch = True
--> 101 options.experimental_optimization.autotune_buffers = True
    102 options.experimental_optimization.apply_default_optimizations = True
    103 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 7
model_model.summary()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/817582770.py in <cell line: 0>()
----> 1 model_model.summary()
      2 
      3 

NameError: name 'model_model' is not defined

## === cell 8
ss_preview = pd.read_csv(data_path + "sample_submission.csv")
test_img_path = test_images_dir_data_path + ss_preview.image_id.iloc[0]

img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {test_img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = (
    cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

plt.figure(figsize=(8, 4))
plt.title(f"TEST IMAGE: {ss_preview.image_id.iloc[0]}")
plt.imshow(resized_img[0])
plt.axis("off")
plt.show()




## === cell 9
ss = ss_preview

test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

test_steps = int(np.ceil(len(ss) / BATCH_SIZE))

probs = model_model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)

preds = np.argmax(probs, axis=1).astype(int)
preds = preds[: len(ss)]

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1545744148.py in <cell line: 0>()
      1 ss = ss_preview
      2 
----> 3 test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
      4 test_ds = test_ds.with_options(options)
      5 test_ds = test_ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE)

NameError: name 'test_tfrec_files' is not defined

## === cell 10
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
print("Shape:", my_submission.shape)
print("Label value counts:\n", my_submission["label"].value_counts().sort_index())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4140135350.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nSaved to: submission.csv")
      4 print("Shape:", my_submission.shape)
      5 print("Label value counts:\n", my_submission["label"].value_counts().sort_index())

NameError: name 'my_submission' is not defined

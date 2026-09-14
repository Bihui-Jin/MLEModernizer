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

0.8516168026594138

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14873) has done: 'I remove the forced pure-Python protobuf implementation (it significantly slows TF graph/IO) and keep deterministic settings otherwise. I switch the data pipeline from `ImageDataGenerator` (Python-side JPEG decode/augment, major bottleneck) to an equivalent `tf.data` pipeline that uses the same images, same preprocessing, and the same augmentation semantics, but runs inside TensorFlow with parallelism, caching, prefetch, and optional XLA compilation. I also ensure we don’t do any redundant work (e.g., avoid repeated dataframe shuffles beyond what’s needed) and use efficient `tf.data` options to overlap CPU input with GPU training. The model, architecture, losses, training schedule (5 epochs + 1 fine-tune epoch), image size, and optimizer settings remain unchanged.'
- What this solution (achieved 0.14873) has done: 'We fix two execution blockers that prevent training/inference from running: (1) the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning TensorFlow to use the pure-Python protobuf implementation (compatibility fix for this environment), and (2) the `tf.data` augmentation crash caused by creating a `RandomRotation` layer inside a traced `map()` function (variable creation in `tf.function`). The augmentation logic remain the same (flip, rotation, translation, zoom), but implemented using stateless TF ops only so it can run inside `Dataset.map` without creating variables. Once those are fixed, `train_gen/val_gen` be defined so the existing training schedule (5 epochs + 1 fine-tune epoch) and submission writing run end-to-end and should substantially improve accuracy versus the previously broken pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFRECORD_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFRECORD_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [
    TRAIN_CSV,
    SAMPLE_SUB,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_TFRECORD_DIR,
    TEST_TFRECORD_DIR,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["image_id", "label"]).issubset(train_df.columns)
assert set(["image_id", "label"]).issubset(sample_df.columns)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print(train_df.head())

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(str)

val_frac = 0.1
train_parts = []
val_parts = []
for lbl, g in train_df.groupby("label", sort=False):
    g = g.sample(frac=1.0, random_state=SEED)
    n_val = max(1, int(round(len(g) * val_frac)))
    val_parts.append(g.iloc[:n_val])
    train_parts.append(g.iloc[n_val:])

train_df_split = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
val_df_split = pd.concat(val_parts, axis=0).reset_index(drop=True)

print("Train split:", len(train_df_split), "Val split:", len(val_df_split))




## === cell 2
IMG_SIZE = 448
BATCH_SIZE = 16  # unchanged

preprocess = tf.keras.applications.efficientnet.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

labels_sorted = sorted(train_df_split["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(labels_sorted)}
print("Class indices:", class_to_index)

train_ids = train_df_split["image_id"].to_numpy()
train_labels = train_df_split["label"].map(class_to_index).astype(np.int32).to_numpy()

val_ids = val_df_split["image_id"].to_numpy()
val_labels = val_df_split["label"].map(class_to_index).astype(np.int32).to_numpy()

train_tfrec_files = sorted(
    [
        os.path.join(TRAIN_TFRECORD_DIR, f)
        for f in os.listdir(TRAIN_TFRECORD_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(TEST_TFRECORD_DIR, f)
        for f in os.listdir(TEST_TFRECORD_DIR)
        if f.endswith(".tfrec")
    ]
)
if not train_tfrec_files:
    raise RuntimeError("No train TFRecord files found.")
if not test_tfrec_files:
    raise RuntimeError("No test TFRecord files found.")


def _decode_resize_from_jpeg_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # faster than tf.image.decode_jpeg
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    return img


augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.055, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
    ],
    name="augmenter",
)


def _preprocess(img):
    return preprocess(img)


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    return img


def _dataset_options_deterministic():
    opt = tf.data.Options()
    opt.deterministic = True
    opt.experimental_optimization.map_parallelization = True
    opt.experimental_optimization.parallel_batch = True
    opt.experimental_optimization.autotune_buffers = True
    return opt


def _interleave_tfrecords(files):
    return tf.data.Dataset.from_tensor_slices(files).interleave(
        lambda f: tf.data.TFRecordDataset(
            f, num_parallel_reads=AUTOTUNE, compression_type=""
        ),
        cycle_length=AUTOTUNE,
        block_length=16,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )


def make_train_ds_from_tfrecords(tfrec_files):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)

    def _map_aug(img, label):
        img = augmenter(img, training=True)
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_images(val_ids_np, val_labels_np):
    paths = np.char.add(TRAIN_DIR + os.sep, val_ids_np.astype(str))
    ds = tf.data.Dataset.from_tensor_slices((paths, val_labels_np))
    ds = ds.with_options(_dataset_options_deterministic())

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_from_jpeg_bytes(img_bytes)
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds_from_tfrecords(train_tfrec_files)
val_gen = make_val_ds_from_images(val_ids, val_labels)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2846877315.py in <cell line: 0>()
    154 
    155 
--> 156 train_gen = make_train_ds_from_tfrecords(train_tfrec_files)
    157 val_gen = make_val_ds_from_images(val_ids, val_labels)
    158 

/tmp/ipykernel_11/2846877315.py in make_train_ds_from_tfrecords(tfrec_files)
    113 def make_train_ds_from_tfrecords(tfrec_files):
    114     ds = _interleave_tfrecords(tfrec_files)
--> 115     ds = ds.with_options(_dataset_options_deterministic())
    116     ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    117 

/tmp/ipykernel_11/2846877315.py in _dataset_options_deterministic()
     94     opt.experimental_optimization.map_parallelization = True
     95     opt.experimental_optimization.parallel_batch = True
---> 96     opt.experimental_optimization.autotune_buffers = True
     97     return opt
     98 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = base(inp, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
model_v4 = tf.keras.Model(inp, out, name="cassava_efficientnetb0")

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_v4.summary()




## === cell 4
EPOCHS = 5

history = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1887502146.py in <cell line: 0>()
      2 
      3 history = model_v4.fit(
----> 4     train_gen,
      5     validation_data=val_gen,
      6     epochs=EPOCHS,

NameError: name 'train_gen' is not defined

## === cell 5
test_df = sample_df[["image_id"]].copy()
test_ids = test_df["image_id"].to_numpy()
test_paths = np.char.add(TEST_DIR + os.sep, test_ids.astype(str))


def make_test_ds_from_images(paths_np, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(paths_np)
    ds = ds.with_options(_dataset_options_deterministic())

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_from_jpeg_bytes(img_bytes)
        img = _preprocess(img)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds_from_images(test_paths, batch_size=32)

pred_v4 = model_v4.predict(test_gen, verbose=1)
predicted_class_indices_v4 = np.argmax(pred_v4, axis=1).astype(int)

sub = sample_df.copy()
if len(predicted_class_indices_v4) != len(sub):
    raise RuntimeError(
        f"Prediction rows ({len(predicted_class_indices_v4)}) != sample rows ({len(sub)})"
    )

sub["label"] = predicted_class_indices_v4

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", sub.columns.tolist())
print("Label distribution:", sub["label"].value_counts().to_dict())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3050833920.py in <cell line: 0>()
     21 
     22 
---> 23 test_gen = make_test_ds_from_images(test_paths, batch_size=32)
     24 
     25 pred_v4 = model_v4.predict(test_gen, verbose=1)

/tmp/ipykernel_11/3050833920.py in make_test_ds_from_images(paths_np, batch_size)
      7 def make_test_ds_from_images(paths_np, batch_size=32):
      8     ds = tf.data.Dataset.from_tensor_slices(paths_np)
----> 9     ds = ds.with_options(_dataset_options_deterministic())
     10 
     11     def _load(path):

/tmp/ipykernel_11/2846877315.py in _dataset_options_deterministic()
     94     opt.experimental_optimization.map_parallelization = True
     95     opt.experimental_optimization.parallel_batch = True
---> 96     opt.experimental_optimization.autotune_buffers = True
     97     return opt
     98 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

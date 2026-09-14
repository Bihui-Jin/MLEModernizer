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

train_id_to_label = dict(zip(train_ids.tolist(), train_labels.tolist()))
val_id_to_label = dict(zip(val_ids.tolist(), val_labels.tolist()))

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
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            0.055, fill_mode="reflect", seed=SEED
        ),  # ~10 degrees
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
    ],
    name="augmenter",
)


def _preprocess(img):
    return preprocess(img)


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_id"]
    return image_id, img, label


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    image_id = ex["image_id"]
    return image_id, img


def _make_lookup_table(py_dict):
    keys = tf.constant(list(py_dict.keys()), dtype=tf.string)
    vals = tf.constant(list(py_dict.values()), dtype=tf.int32)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    return tf.lookup.StaticHashTable(init, default_value=-1)


train_lookup = _make_lookup_table(train_id_to_label)
val_lookup = _make_lookup_table(val_id_to_label)


def _dataset_options_deterministic():
    opt = tf.data.Options()
    opt.deterministic = True
    return opt


def _interleave_tfrecords(files):
    return tf.data.Dataset.from_tensor_slices(files).interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )


def make_train_ds_from_tfrecords(tfrec_files):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.filter(lambda image_id, img, label: train_lookup.lookup(image_id) >= 0)

    ds = ds.map(
        lambda image_id, img, label: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()

    ds = ds.shuffle(
        buffer_size=len(train_ids), seed=SEED, reshuffle_each_iteration=True
    )

    def _map_aug(img, label):
        img = augmenter(img, training=True)
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords(tfrec_files):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.filter(lambda image_id, img, label: val_lookup.lookup(image_id) >= 0)
    ds = ds.map(
        lambda image_id, img, label: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    def _map_val(img, label):
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds_from_tfrecords(train_tfrec_files)
val_gen = make_val_ds_from_tfrecords(train_tfrec_files)




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
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1887502146.py in <cell line: 0>()
      1 EPOCHS = 5
      2 
----> 3 history = model_v4.fit(
      4     train_gen,
      5     validation_data=val_gen,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Shuffle::MemoryCacheImpl::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16928]

## === cell 5
test_df = sample_df[["image_id"]].copy()
test_ids = test_df["image_id"].to_numpy()


def make_test_ds_from_tfrecords(tfrec_files, batch_size=32):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    test_lookup = _make_lookup_table({k: 1 for k in test_ids.tolist()})
    ds = ds.filter(lambda image_id, img: test_lookup.lookup(image_id) >= 0)

    def _map_img(image_id, img):
        img = _preprocess(img)
        return image_id, img

    ds = ds.map(_map_img, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds_from_tfrecords(test_tfrec_files, batch_size=32)

test_pred_ids = []
test_pred_probs = []

for batch in test_gen:
    batch_ids, batch_imgs = batch
    probs = model_v4(batch_imgs, training=False)
    test_pred_ids.append(batch_ids.numpy())
    test_pred_probs.append(probs.numpy())

test_pred_ids = np.concatenate(test_pred_ids, axis=0).astype("S")  # bytes
pred_v4 = np.concatenate(test_pred_probs, axis=0)

id_to_idx = {bid.decode("utf-8"): i for i, bid in enumerate(test_pred_ids.tolist())}
order = np.array([id_to_idx[i] for i in test_ids], dtype=np.int32)
pred_v4 = pred_v4[order]

pred_v4 = np.asarray(pred_v4)
if pred_v4.ndim == 1:
    pred_v4 = pred_v4.reshape(-1, 1)

print("pred_v4 shape:", pred_v4.shape, "expected rows:", len(test_df))




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2865812314.py in <cell line: 0>()
     30 test_pred_probs = []
     31 
---> 32 for batch in test_gen:
     33     batch_ids, batch_imgs = batch
     34     probs = model_v4(batch_imgs, training=False)

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:30 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 6
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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3586121840.py in <cell line: 0>()
----> 1 predicted_class_indices_v4 = np.argmax(pred_v4, axis=1).astype(int)
      2 
      3 sub = sample_df.copy()
      4 if len(predicted_class_indices_v4) != len(sub):
      5     raise RuntimeError(

NameError: name 'pred_v4' is not defined

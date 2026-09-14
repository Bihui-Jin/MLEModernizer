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

0.8485947416137806

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE, "train_images")
test_dir = os.path.join(BASE, "test_images")
train_csv_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
train_tfrecord_dir = os.path.join(BASE, "train_tfrecords")
test_tfrecord_dir = os.path.join(BASE, "test_tfrecords")

assert os.path.exists(train_dir), f"Missing train_dir: {train_dir}"
assert os.path.exists(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv: {sample_sub_path}"
assert os.path.exists(
    train_tfrecord_dir
), f"Missing train_tfrecords: {train_tfrecord_dir}"
assert os.path.exists(test_tfrecord_dir), f"Missing test_tfrecords: {test_tfrecord_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.shape, sample_sub.shape)
print(train.head())
print(sample_sub.head())




## === cell 2
IMG_SIZE = (512, 512)
BATCH_SIZE = 8  # preserve
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE

paths = (train_dir + "/" + train["image_id"].astype(str)).to_numpy()
labels = train["label"].to_numpy(np.int32)

n = len(paths)
val_count = int(round(n * 0.1))  # preserve behavior
train_count = n - val_count

rng = np.random.RandomState(SEED)
idx = np.arange(n)
rng.shuffle(idx)

train_idx = idx[:train_count]
val_idx = idx[train_count:]

train_paths = paths[train_idx]
train_labels = labels[train_idx]
val_paths = paths[val_idx]
val_labels = labels[val_idx]


@tf.function
def _decode_resize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


train_label_by_id = dict(
    zip(
        train["image_id"].astype(str).tolist(), train["label"].astype(np.int32).tolist()
    )
)

train_ids_set = set(os.path.basename(p) for p in train_paths.tolist())
val_ids_set = set(os.path.basename(p) for p in val_paths.tolist())

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrec_files) > 0, "No train tfrecords found"
assert len(test_tfrec_files) > 0, "No test tfrecords found"

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_tfrec(example_proto, need_label=True):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    if not need_label:
        return img, ex["image_id"]
    lbl = tf.cast(ex["label"], tf.int32)
    lbl = tf.where(lbl >= 0, lbl, tf.cast(ex["target"], tf.int32))
    return img, lbl, ex["image_id"]


def _make_split_ds(tfrec_files, ids_keep_set, training: bool):
    ids_keep = tf.constant(sorted(list(ids_keep_set)), dtype=tf.string)

    def _keep_id(image_id):
        return tf.reduce_any(tf.equal(image_id, ids_keep))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.map(lambda x: _parse_tfrec(x, need_label=True), num_parallel_calls=AUTOTUNE)

    ds = ds.filter(lambda img, lbl, image_id: _keep_id(image_id))

    ds = ds.map(lambda img, lbl, image_id: (img, lbl), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(
            buffer_size=min(8192, train_count), seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    else:
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_split_ds(train_tfrec_files, train_ids_set, training=True)
val_ds = _make_split_ds(train_tfrec_files, val_ids_set, training=False)

backbone = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = backbone(x, training=False)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
backbone.trainable = False
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=2,
)

backbone.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=2,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3426720632.py in <cell line: 0>()
      1 backbone.trainable = False
----> 2 history1 = model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=2,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_51129]

## === cell 4
def sample_df(sample_size=50, seed=SEED):
    df = train.sample(sample_size, random_state=seed).reset_index(drop=True)
    return df


dfs = sample_df(sample_size=20)
y_true = dfs["label"].to_numpy(np.int32)

sample_paths = (train_dir + "/" + dfs["image_id"].astype(str)).to_numpy()

sample_ds = tf.data.Dataset.from_tensor_slices(sample_paths).map(
    lambda p: _decode_resize(p, None), num_parallel_calls=AUTOTUNE
)
sample_ds = sample_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(sample_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int)

sample_test = pd.DataFrame(
    {
        "Prediction": preds.tolist(),
        "Actual": y_true.tolist(),
        "image_id": dfs["image_id"],
    }
)
print(sample_test.head(10))
print("Sample accuracy:", (sample_test["Prediction"] == sample_test["Actual"]).mean())




## === cell 5
test_image_ids = sample_sub["image_id"].astype(str).tolist()

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
except Exception:
    pass

test_ds = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(opts)
test_ds = test_ds.map(
    lambda x: _parse_tfrec(x, need_label=False), num_parallel_calls=AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

all_probs = []
all_ids = []
for batch_imgs, batch_ids in test_ds:
    all_probs.append(model(batch_imgs, training=False).numpy())
    all_ids.append(batch_ids.numpy())

probs = np.concatenate(all_probs, axis=0)
ids = np.concatenate(all_ids, axis=0).astype(
    "U"
)  # bytes->str via unicode conversion where possible
if ids.dtype.kind != "U":
    ids = np.array([x.decode("utf-8") for x in ids], dtype="U")

preds_by_id = dict(zip(ids.tolist(), probs.argmax(axis=1).astype(int).tolist()))
predictions = [preds_by_id[iid] for iid in test_image_ids]

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3832572199.py in <cell line: 0>()
     22 all_probs = []
     23 all_ids = []
---> 24 for batch_imgs, batch_ids in test_ds:
     25     all_probs.append(model(batch_imgs, training=False).numpy())
     26     all_ids.append(batch_ids.numpy())

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_tfrec_66512}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:

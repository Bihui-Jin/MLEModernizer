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

0.12706

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14873) has done: 'I remove the forced pure-Python protobuf implementation (it significantly slows TF graph/IO) and keep deterministic settings otherwise. I switch the data pipeline from `ImageDataGenerator` (Python-side JPEG decode/augment, major bottleneck) to an equivalent `tf.data` pipeline that uses the same images, same preprocessing, and the same augmentation semantics, but runs inside TensorFlow with parallelism, caching, prefetch, and optional XLA compilation. I also ensure we don’t do any redundant work (e.g., avoid repeated dataframe shuffles beyond what’s needed) and use efficient `tf.data` options to overlap CPU input with GPU training. The model, architecture, losses, training schedule (5 epochs + 1 fine-tune epoch), image size, and optimizer settings remain unchanged.'
- What this solution (achieved 0.14873) has done: 'We fix two execution blockers that prevent training/inference from running: (1) the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning TensorFlow to use the pure-Python protobuf implementation (compatibility fix for this environment), and (2) the `tf.data` augmentation crash caused by creating a `RandomRotation` layer inside a traced `map()` function (variable creation in `tf.function`). The augmentation logic remain the same (flip, rotation, translation, zoom), but implemented using stateless TF ops only so it can run inside `Dataset.map` without creating variables. Once those are fixed, `train_gen/val_gen` be defined so the existing training schedule (5 epochs + 1 fine-tune epoch) and submission writing run end-to-end and should substantially improve accuracy versus the previously broken pipeline.'
- What this solution (achieved 0.12706) has done: 'The timeout is dominated by reading/decoding/resizing ~18.7k JPEGs from `train_images/` in Python via `tf.io.read_file` for every epoch, while the competition already provides TFRecords with embedded JPEG bytes and labels. To preserve the exact same model and training scheme, the main speed fix is to switch the train/val pipelines to TFRecords (still doing the same decode→resize→preprocess→onehot steps), and to deterministically filter TFRecord examples by the existing `train_df_split/val_df_split` using a lookup table. Additionally, we cache the decoded+preprocessed datasets in memory (safe because base is frozen for most training and images are reused across epochs) and tune tf.data options for throughput without changing semantics. Everything else (architecture, losses, epochs, learning rates, augmentation, evaluation) remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")

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
    tf.config.optimizer.set_jit(False)
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

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.055, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
    ],
    name="augmenter",
)


@tf.function
def _decode_resize_from_jpeg_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _preprocess(img):
    return preprocess(img)


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    return img


def _safe_set(obj, name, value):
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False


def _dataset_options_deterministic():
    opt = tf.data.Options()
    _safe_set(opt, "deterministic", True)

    eo = getattr(opt, "experimental_optimization", None)
    if eo is not None:
        _safe_set(eo, "apply_default_optimizations", True)
        _safe_set(eo, "autotune", True)
        _safe_set(eo, "map_fusion", True)
        _safe_set(eo, "map_parallelization", True)
        _safe_set(eo, "parallel_batch", True)
        _safe_set(eo, "filter_fusion", True)
        _safe_set(eo, "inject_prefetch", True)
        _safe_set(eo, "autotune_buffers", True)

    th = getattr(opt, "threading", None)
    if th is not None:
        _safe_set(th, "private_threadpool_size", 0)
        _safe_set(th, "max_intra_op_parallelism", 0)

    return opt


def _interleave_tfrecords(files):
    return tf.data.Dataset.from_tensor_slices(files).interleave(
        lambda f: tf.data.TFRecordDataset(
            f, num_parallel_reads=AUTOTUNE, compression_type=""
        ),
        cycle_length=AUTOTUNE,
        block_length=64,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )


def _make_split_lookup(df):
    keys = df["image_id"].astype(str).values
    vals = np.ones(len(keys), dtype=np.int8)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(vals, dtype=tf.int8),
        ),
        default_value=tf.constant(0, dtype=tf.int8),
    )


_train_lookup = _make_split_lookup(train_df_split)
_val_lookup = _make_split_lookup(val_df_split)


def _filter_by_lookup(lookup: tf.lookup.StaticHashTable):
    @tf.function
    def _fn(img, label, image_name):
        is_missing = tf.equal(tf.strings.length(image_name), 0)
        in_split = tf.equal(lookup.lookup(image_name), tf.constant(1, tf.int8))
        return tf.logical_or(is_missing, in_split)

    return _fn


def make_train_ds_from_tfrecords_filtered(tfrec_files, lookup):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(_filter_by_lookup(lookup))
    ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _map_pre(img, label, image_name):
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_pre, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords_filtered(tfrec_files, lookup):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(_filter_by_lookup(lookup))

    @tf.function
    def _map_pre(img, label, image_name):
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_pre, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds_from_tfrecords_filtered(train_tfrec_files, _train_lookup)
val_gen = make_val_ds_from_tfrecords_filtered(train_tfrec_files, _val_lookup)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1812723305.py in <cell line: 0>()
    144 
    145 
--> 146 _train_lookup = _make_split_lookup(train_df_split)
    147 _val_lookup = _make_split_lookup(val_df_split)
    148 

/tmp/ipykernel_11/1812723305.py in _make_split_lookup(df)
    135     keys = df["image_id"].astype(str).values
    136     vals = np.ones(len(keys), dtype=np.int8)
--> 137     return tf.lookup.StaticHashTable(
    138         tf.lookup.KeyValueTensorInitializer(
    139             tf.constant(keys, dtype=tf.string),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in __call__(cls, *args, **kwargs)
    101       previous_getter = _make_getter(getter, previous_getter)
    102 
--> 103     return previous_getter(*args, **kwargs)
    104 
    105 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in <lambda>(*a, **kw)
     96       return obj
     97 
---> 98     previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
     99     resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    100     for getter in resource_creator_stack[cls._resource_type()]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in default_resource_creator(next_creator, *a, **kw)
     93       assert next_creator is None
     94       obj = cls.__new__(cls, *a, **kw)
---> 95       obj.__init__(*a, **kw)
     96       return obj
     97 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, initializer, default_value, name, experimental_is_anonymous)
    351     self._name = name or "hash_table"
    352     self._table_name = None
--> 353     super(StaticHashTable, self).__init__(default_value, initializer)
    354     self._value_shape = self._default_value.get_shape()
    355 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, default_value, initializer)
    199       self._initializer = self._track_trackable(initializer, "_initializer")
    200     with ops.init_scope():
--> 201       self._resource_handle = self._create_resource()
    202     if (not context.executing_eagerly() and
    203         ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in _create_resource(self)
    361           name=self._name)
    362     else:
--> 363       table_ref = gen_lookup_ops.hash_table_v2(
    364           shared_name=self._shared_name,
    365           key_dtype=self._initializer.key_dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_lookup_ops.py in hash_table_v2(key_dtype, value_dtype, container, shared_name, use_node_name_sharing, name)
    483       return _result
    484     except _core._NotOkStatusException as e:
--> 485       _ops.raise_from_not_ok_status(e, name)
    486     except _core._FallbackException:
    487       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_STRING, shared_name="21", use_node_name_sharing=false, value_dtype=DT_INT8]
All kernels registered for op HashTableV2:
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_DOUBLE]
 [Op:HashTableV2] name: hash_table

## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = augmenter(inp)
x = base(x, training=False)
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


def make_test_ds_from_tfrecords(tfrec_files, batch_size=32):
    ds = _interleave_tfrecords(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds_from_tfrecords(test_tfrec_files, batch_size=32)

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

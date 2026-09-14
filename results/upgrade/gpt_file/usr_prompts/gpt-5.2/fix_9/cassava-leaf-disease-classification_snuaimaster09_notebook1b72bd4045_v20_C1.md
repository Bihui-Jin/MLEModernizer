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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SUBMISSION_MODE = 1

import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import InceptionV3, Xception

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except Exception:
    StratifiedShuffleSplit = None

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _CPU_COUNT = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(max(2, _CPU_COUNT))
    tf.config.threading.set_inter_op_parallelism_threads(max(2, _CPU_COUNT // 2))
except Exception:
    pass

CPU_AUTOTUNE = tf.data.AUTOTUNE

df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_train["label"] = df_train["label"].astype(str)

batch_size = 32
image_size = 300
input_shape = (image_size, image_size, 3)
target_size = (image_size, image_size)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def create_Inception():
    base_model = InceptionV3(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    inception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    inception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return inception


def create_Xception():
    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(5, activation="softmax", name="dense", dtype="float32")(x)

    xception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2, from_logits=False
    )

    xception.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return xception




## === cell 2
preprocess = tf.keras.applications.inception_v3.preprocess_input

TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"

NUM_CLASSES = 5


def _build_paths_and_labels(df: pd.DataFrame, directory: str):
    paths = (directory.rstrip("/") + "/" + df["image_id"].astype(str)).to_numpy()
    labels = df["label"].astype(np.int32).to_numpy()
    return paths, labels


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 0], tf.int32),
        minval=-20.0,
        maxval=20.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)

    img = (
        tf.keras.layers.RandomRotation(factor=0.0, fill_mode="nearest")(
            img, training=True
        )
        if False
        else img
    )

    img = tf.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="nearest"
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=0.8,
        maxval=1.2,
        dtype=tf.float32,
    )
    h = tf.cast(tf.round(tf.cast(image_size, tf.float32) / z), tf.int32)
    w = tf.cast(tf.round(tf.cast(image_size, tf.float32) / z), tf.int32)

    h = tf.maximum(1, h)
    w = tf.maximum(1, w)

    img = tf.image.resize_with_crop_or_pad(img, h, w)
    img = tf.image.resize(
        img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


def make_train_ds(df, batch, shuffle=True, augment=True):
    paths, labels = _build_paths_and_labels(df, TRAIN_DIR)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    idx = tf.data.Dataset.range(len(df))
    ds = tf.data.Dataset.zip((ds, idx))

    def _map_fn(pl, i):
        (p, lab) = pl
        img = _decode_resize(p)
        if augment:
            seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
            img = _augment(img, seed_pair)
        img = preprocess(img)
        y = tf.one_hot(tf.cast(lab, tf.int32), NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds


def make_val_ds(df, batch):
    paths, labels = _build_paths_and_labels(df, TRAIN_DIR)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(p, lab):
        img = _decode_resize(p)
        img = preprocess(img)
        y = tf.one_hot(tf.cast(lab, tf.int32), NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds


def make_test_ds(image_ids, batch):
    paths = (TEST_DIR.rstrip("/") + "/" + image_ids.astype(str)).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = _decode_resize(p)
        img = preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(CPU_AUTOTUNE)
    return ds




## === cell 3
if SUBMISSION_MODE == 0:
    fold_number = 0
    n_splits = 3
    epochs = 8

    tf.keras.backend.clear_session()

    if StratifiedShuffleSplit is None:
        raise ImportError(
            "scikit-learn is required for SUBMISSION_MODE==0 (StratifiedShuffleSplit missing)."
        )

    KFoldSplit = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.1, random_state=SEED
    )

    for train_index, val_index in KFoldSplit.split(
        df_train["image_id"], df_train["label"]
    ):
        train_set = df_train.loc[train_index].copy()
        val_set = df_train.loc[val_index].copy()

        train_ds = make_train_ds(
            train_set.assign(label=train_set["label"].astype(int)),
            batch_size,
            shuffle=True,
            augment=True,
        )
        val_ds = make_val_ds(
            val_set.assign(label=val_set["label"].astype(int)), batch_size
        )

        model = create_Inception()
        print("Training fold no.: " + str(fold_number + 1))

        model_name = "inception "
        fold_name = "fold.h5"
        filepath = model_name + str(fold_number + 1) + fold_name
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=3),
            ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True),
        ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
        )
        fold_number += 1
        if fold_number == n_splits:
            print("Training finished!")

if SUBMISSION_MODE == 1:
    sample_path = os.path.join(
        "../input/cassava-leaf-disease-classification", "sample_submission.csv"
    )
    SampleSubmit = pd.read_csv(sample_path)

    candidate_model_paths = [
        "/kaggle/input/inception2fold/inception 2fold.h5",
        "/kaggle/input/inception2fold/inception2fold.h5",
        "../input/inception2fold/inception 2fold.h5",
        "../input/inception2fold/inception2fold.h5",
        "/kaggle/working/inception_fallback.h5",
    ]
    model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)

    if model_path is not None:
        model = load_model(model_path)
        try:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9, nesterov=True
            )
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.2, from_logits=False
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        except Exception:
            pass
    else:
        epochs = 3

        if StratifiedShuffleSplit is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.1, random_state=SEED
            )
            tr_idx, va_idx = next(
                splitter.split(df_train["image_id"], df_train["label"])
            )
        else:
            perm = np.random.RandomState(SEED).permutation(len(df_train))
            cut = int(len(df_train) * 0.9)
            tr_idx, va_idx = perm[:cut], perm[cut:]

        train_set = df_train.iloc[tr_idx].reset_index(drop=True).copy()
        val_set = df_train.iloc[va_idx].reset_index(drop=True).copy()

        tf.keras.backend.clear_session()
        model = create_Inception()

        train_set["label"] = train_set["label"].astype(int)
        val_set["label"] = val_set["label"].astype(int)
        train_ds = make_train_ds(train_set, batch_size, shuffle=True, augment=True)
        val_ds = make_val_ds(val_set, batch_size)

        local_model_path = "/kaggle/working/inception_fallback.h5"
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", patience=1, verbose=1, factor=0.2),
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(
                filepath=local_model_path, monitor="val_loss", save_best_only=True
            ),
        ]

        model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(local_model_path)
        model_path = local_model_path

    infer_batch_size = 64
    test_ds = make_test_ds(SampleSubmit["image_id"], infer_batch_size)

    preds = model.predict(
        test_ds,
        verbose=1,
    )
    results = np.argmax(preds, axis=1).astype(int).tolist()

    SampleSubmit["label"] = results[: len(SampleSubmit)]
    SampleSubmit["label"] = SampleSubmit["label"].astype(int)

    out_path = os.path.join("/kaggle/working", "submission.csv")
    SampleSubmit.to_csv(out_path, index=False)

    print("Loaded model from:", model_path)
    print("Wrote:", out_path)
    print(SampleSubmit.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_12/2818015678.py in <cell line: 0>()
    105         train_set["label"] = train_set["label"].astype(int)
    106         val_set["label"] = val_set["label"].astype(int)
--> 107         train_ds = make_train_ds(train_set, batch_size, shuffle=True, augment=True)
    108         val_ds = make_val_ds(val_set, batch_size)
    109 

/tmp/ipykernel_12/2704544771.py in make_train_ds(df, batch, shuffle, augment)
     88         return img, y
     89 
---> 90     ds = ds.map(_map_fn, num_parallel_calls=CPU_AUTOTUNE, deterministic=True)
     91     ds = ds.batch(batch, drop_remainder=False)
     92     ds = ds.prefetch(CPU_AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileparghjcu.py in tf___map_fn(pl, i)
     28                     pass
     29                 seed_pair = ag__.Undefined('seed_pair')
---> 30                 ag__.if_stmt(ag__.ld(augment), if_body, else_body, get_state, set_state, ('img',), 1)
     31                 img = ag__.converted_call(ag__.ld(preprocess), (ag__.ld(img),), None, fscope)
     32                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(lab), ag__.ld(tf).int32), None, fscope), ag__.ld(NUM_CLASSES)), dict(dtype=ag__.ld(tf).float32), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_fileparghjcu.py in if_body()
     22                     nonlocal img
     23                     seed_pair = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 24                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed_pair)), None, fscope)
     25 
     26                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filehx7wnb01.py in tf___augment(img, seed_pair)
     12                 angle = ag__.ld(angle) * (ag__.ld(np).pi / 180.0)
     13                 img = ag__.if_exp(False, lambda: ag__.converted_call(ag__.converted_call(ag__.ld(tf).keras.layers.RandomRotation, (), dict(factor=0.0, fill_mode='nearest'), fscope), (ag__.ld(img),), dict(training=True), fscope), lambda: ag__.ld(img), 'False')
---> 14                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR', fill_mode='nearest'), fscope)
     15                 z = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_pair) + ag__.converted_call(ag__.ld(tf).constant, ([2, 0], ag__.ld(tf).int32), None, fscope), minval=0.8, maxval=1.2, dtype=ag__.ld(tf).float32), fscope)
     16                 h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image_size), ag__.ld(tf).float32), None, fscope) / ag__.ld(z),), None, fscope), ag__.ld(tf).int32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_12/2704544771.py", line 85, in _map_fn  *
        img = _augment(img, seed_pair)
    File "/tmp/ipykernel_12/2704544771.py", line 47, in _augment  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'

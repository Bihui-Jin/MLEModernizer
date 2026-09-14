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

0.7963130855243276

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10762) has done: 'The timeout is dominated by slow Python-side image loading/augmentation (Keras `ImageDataGenerator` + `from_generator`) and an extra, redundant rescaling layer that effectively double-normalizes inputs when training from scratch. I keep the same model architecture and training loop semantics, but make the pipeline faster by (1) using `tf.data` with vectorized file reads/decoding/resizing, caching, and deterministic parallelism for train/valid, and (2) switching to TFRecords if present (same pixels/labels, much faster I/O on Kaggle). For inference, I also avoid building an intermediate Python list of paths and use the already-built dataset with optimal tf.data settings; predictions and submission formatting remain identical.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config, "optimizer") and hasattr(tf.config.optimizer, "set_jit"):
        tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TF version:", tf.__version__)

_HAS_GPU = False
try:
    _HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0
except Exception:
    _HAS_GPU = False
print("GPU available:", _HAS_GPU)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
weight_path_candidates = [
    "/kaggle/input/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
    "../input/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
    "/kaggle/data/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
    "/kaggle/working/model-ensembling-with-k-fold/fineTuned_v0.39.h5",
]
weight_path = next((p for p in weight_path_candidates if os.path.exists(p)), None)

my_model = None
if weight_path is not None:
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False)
else:
    print(
        "Pretrained .h5 not found; will train a model from scratch/fine-tune locally to produce submission."
    )




## === cell 2
base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def find_file(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.exists(p):
            return p
    return None


train_csv_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")

test_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]
train_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
]


def first_existing_dir(cands, desc):
    for d in cands:
        if os.path.isdir(d):
            print(f"Using {desc} dir:", d)
            return d
    raise FileNotFoundError(f"Could not find {desc} directory. Checked: {cands}")


test_dir = first_existing_dir(test_dir_candidates, "test_images")
train_dir = first_existing_dir(train_dir_candidates, "train_images")

if train_csv_path is None:
    raise FileNotFoundError("Could not locate train.csv under known base paths.")
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known base paths."
    )

test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found in {test_dir}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = [os.path.basename(p) for p in test_images]


def make_test_gen(batch_size=128):
    my_test_idg = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(512, 512),
    )
    return test_gen


@tf.function
def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [512, 512], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def make_test_ds(paths, batch_size=128):
    autotune = tf.data.AUTOTUNE
    paths_tf = tf.convert_to_tensor(paths, dtype=tf.string)

    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds = tf.data.Dataset.from_tensor_slices(paths_tf).with_options(options)
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            _load_test_image,
            batch_size=batch_size,
            num_parallel_calls=autotune,
            drop_remainder=False,
            deterministic=True,
        )
    )
    if _HAS_GPU:
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
        except Exception:
            ds = ds.prefetch(autotune)
    else:
        ds = ds.prefetch(autotune)
    return ds


def find_tfrecords(kind: str):
    patterns = []
    for b in base_candidates:
        patterns.append(os.path.join(b, f"{kind}_tfrecords", "*.tfrec"))
        patterns.append(
            os.path.join(
                b, "cassava-leaf-disease-classification", f"{kind}_tfrecords", "*.tfrec"
            )
        )
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(list(set(files)))


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [512, 512], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def make_train_valid_ds_from_tfrecords(
    tfrec_files, batch_size, seed=SEED, total_count=None
):
    autotune = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    if total_count is None:
        raise ValueError(
            "total_count must be provided to avoid an O(N) scan that can cause timeouts."
        )
    card = int(total_count)

    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=autotune,
    ).with_options(options)

    ds = ds.map(_parse_tfrec, num_parallel_calls=autotune, deterministic=True)

    ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)

    split = int(0.9 * card)
    ds_tr = ds.take(split)
    ds_va = ds.skip(split)

    ds_tr = ds_tr.apply(
        tf.data.experimental.map_and_batch(
            lambda x, y: (x, y),
            batch_size=batch_size,
            num_parallel_calls=autotune,
            drop_remainder=False,
            deterministic=True,
        )
    )
    ds_va = ds_va.batch(batch_size, drop_remainder=False)

    if _HAS_GPU:
        try:
            ds_tr = ds_tr.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
            ds_va = ds_va.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
        except Exception:
            ds_tr = ds_tr.prefetch(autotune)
            ds_va = ds_va.prefetch(autotune)
    else:
        ds_tr = ds_tr.prefetch(autotune)
        ds_va = ds_va.prefetch(autotune)

    return ds_tr, ds_va, card


def make_train_valid_ds_from_paths(df_tr, df_va, batch_size, seed=SEED):
    autotune = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    tr_paths = tf.convert_to_tensor(df_tr["path"].values, dtype=tf.string)
    tr_labels = tf.convert_to_tensor(df_tr["label_int"].values, dtype=tf.int32)
    va_paths = tf.convert_to_tensor(df_va["path"].values, dtype=tf.string)
    va_labels = tf.convert_to_tensor(df_va["label_int"].values, dtype=tf.int32)

    @tf.function
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, [512, 512], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        return img, label

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(
        options
    )
    va_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_labels)).with_options(
        options
    )

    tr_ds = tr_ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    tr_ds = tr_ds.apply(
        tf.data.experimental.map_and_batch(
            _load,
            batch_size=batch_size,
            num_parallel_calls=autotune,
            drop_remainder=False,
            deterministic=True,
        )
    )
    va_ds = va_ds.apply(
        tf.data.experimental.map_and_batch(
            _load,
            batch_size=batch_size,
            num_parallel_calls=autotune,
            drop_remainder=False,
            deterministic=True,
        )
    )

    if _HAS_GPU:
        try:
            tr_ds = tr_ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
            va_ds = va_ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
        except Exception:
            tr_ds = tr_ds.prefetch(autotune)
            va_ds = va_ds.prefetch(autotune)
    else:
        tr_ds = tr_ds.prefetch(autotune)
        va_ds = va_ds.prefetch(autotune)
    return tr_ds, va_ds




## === cell 3
if my_model is None:
    df_train = pd.read_csv(train_csv_path)

    df_train["path"] = train_dir.rstrip("/") + "/" + df_train["image_id"].astype(str)

    sample_check = df_train["path"].iloc[:: max(1, len(df_train) // 2000)].tolist()
    if not all(os.path.exists(p) for p in sample_check):
        missing = [p for p in sample_check if not os.path.exists(p)][:5]
        raise FileNotFoundError(
            f"Some training image paths do not exist (sample check); examples: {missing}"
        )

    df_train["label"] = df_train["label"].astype(str)
    df_train["label_int"] = df_train["label"].astype(np.int32)

    idx = np.arange(len(df_train))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train.iloc[va_idx].reset_index(drop=True)

    batch_size = 32 if not DEBUG else 8

    tfrec_train_files = find_tfrecords("train")
    use_tfrec = len(tfrec_train_files) > 0
    if use_tfrec:
        print(f"Using TFRecords for training: {len(tfrec_train_files)} files")
        train_ds, valid_ds, card = make_train_valid_ds_from_tfrecords(
            tfrec_train_files,
            batch_size=batch_size,
            seed=SEED,
            total_count=len(df_train),
        )
        steps_per_epoch = int(np.ceil((0.9 * len(df_train)) / batch_size))
        validation_steps = int(np.ceil((0.1 * len(df_train)) / batch_size))
        train_data = train_ds
        valid_data = valid_ds
    else:
        print("TFRecords not found; falling back to ImageDataGenerator (slower).")
        train_idg = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
        )
        valid_idg = ImageDataGenerator(rescale=1.0 / 255.0)

        train_gen = train_idg.flow_from_dataframe(
            df_tr,
            x_col="path",
            y_col="label",
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=True,
            seed=SEED,
        )
        valid_gen = valid_idg.flow_from_dataframe(
            df_va,
            x_col="path",
            y_col="label",
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=False,
            seed=SEED,
        )
        steps_per_epoch = len(train_gen)
        validation_steps = len(valid_gen)
        train_data = train_gen
        valid_data = valid_gen

    inputs = tf.keras.Input(shape=(512, 512, 3))

    x = inputs

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 2 if not DEBUG else 1

    my_model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/922663270.py in <cell line: 0>()
     29     if use_tfrec:
     30         print(f"Using TFRecords for training: {len(tfrec_train_files)} files")
---> 31         train_ds, valid_ds, card = make_train_valid_ds_from_tfrecords(
     32             tfrec_train_files,
     33             batch_size=batch_size,

/tmp/ipykernel_11/3210960782.py in make_train_valid_ds_from_tfrecords(tfrec_files, batch_size, seed, total_count)
    197 
    198     ds_tr = ds_tr.apply(
--> 199         tf.data.experimental.map_and_batch(
    200             lambda x, y: (x, y),
    201             batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    381               'in a future version' if date is None else ('after %s' % date),
    382               instructions)
--> 383       return func(*args, **kwargs)
    384 
    385     doc_controls.set_deprecated(new_func)

TypeError: map_and_batch() got an unexpected keyword argument 'deterministic'

## === cell 4
batch_size = 128

test_paths = df_test["path"].values.astype(str)
test_ds = make_test_ds(test_paths, batch_size=batch_size)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test[["image_id"]].copy()
final_submission["label"] = pred_test_labels

sample_sub = pd.read_csv(sample_sub_path)
final_csv = sample_sub[["image_id"]].merge(final_submission, on="image_id", how="left")

if final_csv["label"].isna().any():
    mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print(f"Wrote submission.csv with shape: {final_csv.shape}")
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3468071117.py in <cell line: 0>()
      2 
      3 test_paths = df_test["path"].values.astype(str)
----> 4 test_ds = make_test_ds(test_paths, batch_size=batch_size)
      5 
      6 pred_test = my_model.predict(

/tmp/ipykernel_11/3210960782.py in make_test_ds(paths, batch_size)
    107     ds = tf.data.Dataset.from_tensor_slices(paths_tf).with_options(options)
    108     ds = ds.apply(
--> 109         tf.data.experimental.map_and_batch(
    110             _load_test_image,
    111             batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    381               'in a future version' if date is None else ('after %s' % date),
    382               instructions)
--> 383       return func(*args, **kwargs)
    384 
    385     doc_controls.set_deprecated(new_func)

TypeError: map_and_batch() got an unexpected keyword argument 'deterministic'

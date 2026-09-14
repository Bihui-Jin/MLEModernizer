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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.0507

# 6. Current score

0.42937

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the `MessageFactory.GetPrototype` crash by pinning TensorFlow to use the pure-Python protobuf implementation (a known TF2.18 + protobuf 6.x issue) before TensorFlow/Keras imports happen. I also fix the submission-length error by ensuring the test generator is created with `shuffle=False` and that the submission rows are taken directly from `sample_submission.csv` (the authoritative test image list/order). Finally, I make the training generator use a compatible preprocessing function (albumentations expects numpy images, while Keras passes float tensors) by switching to Keras-native augmentations via `ImageDataGenerator` arguments, keeping the overall training approach (ImageDataGenerator + EfficientNetB0 training loop) intact and ensuring the pipeline produces a valid `submission.csv`.'
- What this solution (achieved 0.43236) has done: 'The timeout is dominated by training EfficientNetB0 from scratch for 10 epochs on ~18.7k images with slow Python-side image loading/augmentation via `ImageDataGenerator`. To keep the exact same model and training semantics while speeding up, I (1) switch to using the provided TFRecords with an equivalent decode+augment pipeline, (2) compile the model with XLA for faster GPU/CPU execution, and (3) ensure the input pipeline is fully optimized (parallel reads, map parallelism, caching, prefetch, deterministic order). Prediction also read TFRecords (or fall back to images if needed) with prefetching to reduce overhead while preserving identical outputs up to negligible floating-point differences.'
- What this solution (achieved 0.41293) has done: 'The crash happens before TensorFlow loads because TF 2.18 with protobuf 6.x can still hit `MessageFactory.GetPrototype` despite the environment variables alone. I fix this by force-importing the pure-Python protobuf implementation *before* importing TensorFlow, and by keeping the env vars as a fallback. I also keep your TFRecord-based pipeline and model/training logic intact, only making the import order robust so the notebook runs end-to-end and writes a valid `submission.csv`. Since your current score (0.43236) is already far above the target (0.0507) and higher-is-better, I not make score-improving changes—only stability fixes.'
- What this solution (achieved 0.60613) has done: 'The runtime crash is coming from an incompatibility between TensorFlow 2.18 and protobuf 6.x where forcing the pure-Python protobuf implementation via env vars is not always sufficient; we need to hard-pin the protobuf runtime to the Python implementation *before* TensorFlow is imported. I fix this by importing `google.protobuf` early and forcibly setting the implementation type (with safe fallbacks), and by disabling C++ descriptors via env vars before any protobuf/tensorflow import occurs. I not change your model, TFRecord pipeline, training loop, epochs, or prediction logic (your score is already far above the target, so only stability/valid-submission fixes are appropriate). The script then run end-to-end and write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.45179) has done: 'The crash happens before any of your training code runs: TensorFlow 2.18 + protobuf 6.x can import in a state where `MessageFactory.GetPrototype` is missing, and the current environment-variable approach isn’t reliably taking effect early enough. I fix this by forcing protobuf to use the pure-Python implementation *before* TensorFlow is imported, including a safe monkey-patch that restores `GetPrototype` if it’s missing (this is score-neutral and only unblocks execution). I keep your TFRecord pipeline, EfficientNetB0-from-scratch model, training loop, epochs, and prediction logic unchanged. I also keep the submission generation aligned to `sample_submission.csv` to guarantee correct row count and order.'
- What this solution (achieved 0.44768) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying the monkey-patch at the correct scope (instance vs class) and doing it before importing TensorFlow, which is what currently prevents the notebook from running at all. I keep your TFRecord pipeline, EfficientNetB0 model, compile settings, and training/prediction logic unchanged so the score behavior remains effectively the same (your current score is already far above the target). I also keep submission generation aligned to `sample_submission.csv` (authoritative order/length) and ensure the file written is exactly `submission.csv` with the required columns.'
- What this solution (achieved 0.23767) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf runtime *and* monkey-patching the correct MessageFactory class used by protobuf 6.x (the current patch targets a module that no longer matches what TF ends up using). This change is purely to unblock execution and is score-neutral. I keep your TFRecord pipeline, EfficientNetB0-from-scratch model, training loop, and submission construction identical so the score behavior remains essentially unchanged (your current score is already far above the target, so no score-improving changes are needed). The script then run end-to-end and write a valid `submission.csv` with the required columns and row count/order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.45478) has done: 'I fix the protobuf `MessageFactory.GetPrototype` AttributeError by applying a safe monkey-patch to the *instance* created by protobuf’s `symbol_database` (the object TensorFlow actually ends up using), before importing TensorFlow. This unblocks the notebook so it runs end-to-end again, and is score-neutral (no model/training logic changes). I also remove the unused `descriptor_pool` touch that can trigger the failing path early, while keeping your TFRecord pipeline, EfficientNetB0-from-scratch model, training loop, and submission construction unchanged. The script still write a valid `submission.csv` with the correct columns and row count aligned to `sample_submission.csv`.'
- What this solution (achieved 0.45441) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime early and applying a safe compatibility patch to the exact `MessageFactory` object that protobuf’s `symbol_database.Default().message_factory` uses (the one TF ends up touching). This is purely an execution/unblocking change and does not alter your TFRecord pipeline, EfficientNetB0-from-scratch model, training loop, epochs, or prediction logic—so it should be score-neutral (your current score is already far above the target, and higher-is-better). I also make the patch resilient across protobuf 6.x variants by handling both missing `GetPrototype` and legacy access patterns without triggering the failing path. The script then run end-to-end and write a valid `submission.csv` with the correct columns and row count/order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.42937) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early and applying a correct compatibility shim for `MessageFactory.GetPrototype` on the actual class used by protobuf 6.x (`google.protobuf.message_factory.MessageFactory`). This is an execution-only fix and won’t change your model, TFRecord pipeline, training loop, epochs, or prediction logic, so score behavior should remain essentially unchanged (and since your current score is far above the target, we avoid any score-improving changes). I also remove the incorrect patch attempt against a non-existent `message_factory.message_factory` instance attribute, which is what currently leads to the `GetPrototype` AttributeError. The script then run end-to-end and write a valid `submission.csv` with the required columns and correct row count/order aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    from google.protobuf.internal import api_implementation  # type: ignore

    try:
        api_implementation._SetType("python")  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import google.protobuf.message_factory as message_factory  # type: ignore

        MF = getattr(message_factory, "MessageFactory", None)
        if (
            MF is not None
            and (not hasattr(MF, "GetPrototype"))
            and hasattr(MF, "GetMessageClass")
        ):
            try:
                setattr(MF, "GetPrototype", MF.GetMessageClass)
            except Exception:
                pass
    except Exception:
        pass

    try:
        from google.protobuf import symbol_database as _symbol_database  # type: ignore

        _db = _symbol_database.Default()
        _db_mf = getattr(_db, "message_factory", None)
        if (
            _db_mf is not None
            and (not hasattr(_db_mf, "GetPrototype"))
            and hasattr(_db_mf, "GetMessageClass")
        ):
            try:
                setattr(_db_mf, "GetPrototype", _db_mf.GetMessageClass)
            except Exception:
                pass

        _maybe_factory = getattr(_db_mf, "_factory", None)
        if (
            _maybe_factory is not None
            and (not hasattr(_maybe_factory, "GetPrototype"))
            and hasattr(_maybe_factory, "GetMessageClass")
        ):
            try:
                setattr(_maybe_factory, "GetPrototype", _maybe_factory.GetMessageClass)
            except Exception:
                pass
    except Exception:
        pass

except Exception:
    pass

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(str)

sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_df = sub_df.copy()
test_df["label"] = "0"

print("TF version:", tf.__version__)
print("Train rows:", len(train_df), "Test rows:", len(test_df))

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    import tensorflow_addons as tfa  # optional, often not installed

    _HAS_TFA = True
except Exception:
    _HAS_TFA = False

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_tfrec_pattern = os.path.join(INPUT_DIR, "train_tfrecords", "*.tfrec")
test_tfrec_pattern = os.path.join(INPUT_DIR, "test_tfrecords", "*.tfrec")
train_tfrec_files = tf.io.gfile.glob(train_tfrec_pattern)
test_tfrec_files = tf.io.gfile.glob(test_tfrec_pattern)


def _parse_tfrec(example_proto, labeled=True):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    if labeled:
        feature_description["target"] = tf.io.FixedLenFeature([], tf.int64)
    x = tf.io.parse_single_example(example_proto, feature_description)

    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0

    if labeled:
        y = tf.cast(x["target"], tf.int32)
        y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
        return img, y
    return img


def _augment(img, y):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED + 1)

    if _HAS_TFA:
        angle = tf.random.stateless_uniform(
            [], seed=[SEED, 7], minval=-np.pi / 2, maxval=np.pi / 2
        )
        img = tfa.image.rotate(img, angles=angle, interpolation="BILINEAR")
    else:
        k = tf.random.stateless_uniform(
            [], seed=[SEED, 9], minval=0, maxval=4, dtype=tf.int32
        )
        img = tf.image.rot90(img, k=k)

    return img, y


def make_train_ds(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(
        lambda x: _parse_tfrec(x, labeled=True),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if len(train_tfrec_files) > 0:
    train_gen = make_train_ds(sorted(train_tfrec_files))
    steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
else:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
    )
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(INPUT_DIR, "train_images"),
        x_col="image_id",
        y_col="label",
        class_mode="categorical",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    steps_per_epoch = None



## === cell 2
model = EfficientNetB0(
    include_top=True,
    weights=None,
    classes=5,
    input_shape=(224, 224, 3),
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

fit_kwargs = dict(epochs=10, verbose=2)
if steps_per_epoch is not None:
    fit_kwargs["steps_per_epoch"] = steps_per_epoch

model.fit(train_gen, **fit_kwargs)




## === cell 3
def make_test_ds(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(
        lambda x: _parse_tfrec(x, labeled=False),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if len(test_tfrec_files) > 0:
    test_gen = make_test_ds(sorted(test_tfrec_files))
    y_pred = model.predict(test_gen, verbose=1)
else:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=os.path.join(INPUT_DIR, "test_images"),
        x_col="image_id",
        y_col=None,
        class_mode=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    y_pred = model.predict(test_gen, verbose=1)

test_df["label"] = np.argmax(y_pred, axis=1).astype(int)

submission = test_df[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print("Done. Wrote submission.csv with", len(submission), "rows.")
print(submission.head())

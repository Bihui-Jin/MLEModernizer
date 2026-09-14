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

3.14

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

0.1403747355696585

# 6. Current score

0.60314

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The immediate blocker is that the notebook fails before inference because (1) TensorFlow/Protobuf incompatibility triggers `MessageFactory.GetPrototype`, and (2) the referenced pre-trained model file doesn’t exist in your `/kaggle/input`. I fix this by forcing TF to use the pure-Python protobuf implementation (a common Kaggle workaround) and by adding a safe fallback path: if the external model can’t be loaded, train a small baseline CNN on `train_images` to still generate a valid `submission.csv`. This keeps the inference pipeline semantics the same (image preprocessing → model → argmax labels) and guarantees an end-to-end run producing a correctly formatted submission file.'
- What this solution (achieved 0.61809) has done: 'You’re failing immediately on importing/initializing TensorFlow due to a protobuf API mismatch (`MessageFactory.GetPrototype`) that happens before any model code runs; setting the env var inside the notebook is too late because protobuf is imported during TensorFlow import. I fix this by moving the protobuf environment variables to the very top (before any TF/protobuf-related imports) and also adding a safe fallback to force the pure-Python protobuf backend at runtime if needed. Since your current score (0.61099) is far above the very low target (0.14037) and higher-is-better, I keep the modeling/training logic intact and only make stability fixes (no score-improving changes). The script then run end-to-end and always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.61099) has done: 'The timeout is dominated by slow JPEG decode/resize on CPU for 2676 test images at 512×512, plus extra tf.data overhead. I keep the exact preprocessing semantics and model/prediction logic, but reduce input pipeline cost by using the TFRecord test set (same images) with efficient parallel reads, deterministic options, and the same decode→float32/255→resize pipeline. I also enable dataset-level caching for the (single-pass) test pipeline and set TF threading explicitly to avoid oversubscription overhead. These changes are equivalent in outputs (same images, same preprocessing) but substantially reduce wall time.'
- What this solution (achieved 0.61248) has done: 'The main timeout driver is forcing Protocol Buffers to use the pure-Python implementation, which makes TFRecord parsing extremely slow; switching back to the default C++ protobuf preserves exact semantics while dramatically reducing input pipeline time. The second bottleneck is the TFRecord “reorder-by-image_name” pipeline (hash lookup + large batch sort + gather + unbatch), which is unnecessary if we instead iterate test images in the already-correct order from `sample_submission.csv` (same evaluation semantics, same images, just faster). Finally, we keep determinism and the same preprocessing/model logic, but tighten the `tf.data` pipeline to avoid expensive caching/unbatching and reduce overhead.'
- What this solution (achieved 0.61286) has done: 'The crash happens before any of your code runs because TensorFlow imports protobuf, and the environment’s protobuf version lacks `MessageFactory.GetPrototype`. The minimal, standard Kaggle workaround is to force the pure-Python protobuf implementation **before** importing TensorFlow; we do that at the very top and keep the rest of your pipeline (preprocessing → model → argmax → submission) unchanged. To keep runtime acceptable, we only apply this protobuf fallback when needed, but in this environment it’s required to avoid the import-time exception. No score-tuning changes are made since your current score is already far above the (very low) target and the priority is correctness/stability.'
- What this solution (achieved 0.61398) has done: 'The timeout is dominated by JPEG decoding + resizing 2676 images to 512×512 on the CPU with a Python-protobuf stack, plus extra overhead from forcing the pure-Python protobuf implementation. I keep the exact same model and prediction semantics, but speed up the input pipeline by (1) using the default (faster) protobuf runtime, (2) enabling TF data pipeline optimizations (map/batch fusion, parallel batching), and (3) using the fused `tf.image.convert_image_dtype` + `resize(..., antialias=False)` path while keeping identical shapes and value ranges. I also ensure the dataset doesn’t introduce accidental bottlenecks by setting private threadpools/prefetch correctly and keeping determinism as requested. No changes are made to model architecture, training logic, loss, or evaluation.'
- What this solution (achieved 0.60314) has done: 'I keep the exact model and preprocessing logic, but remove the biggest source of timeout: per-image JPEG decoding from 2,676 separate files. Instead, the test pipeline read the provided TFRecords (same images) using a `tf.data` pipeline with parallel interleave, caching, and prefetch, which is equivalent in semantics and far faster. I also compile the `tf.data` parsing/mapping functions once and reuse a single `Options` object, and I compute `steps` to prevent any accidental extra iteration. All paths and the fallback training branch remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys


def _safe_import_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        msg = repr(e)
        needs_python_protobuf = (
            "GetPrototype" in msg or "MessageFactory" in msg or "google.protobuf" in msg
        )
        already_forced = (
            os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") == "python"
        )
        if needs_python_protobuf and not already_forced:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

tf = _safe_import_tensorflow()
from tensorflow.keras.models import load_model

MODEL_PATH = "/kaggle/input/leaf-model/tensorflow2/default/1/best_model.h5"

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_SUB = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
TEST_TFRECORD_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

IMAGE_SIZE = [512, 512]
BATCH_SIZE = 32
NUM_CLASSES = 5

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF choose
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def process_image(image_path):
    bits = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, antialias=False)
    return image


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def process_image_with_label(image_path, label):
    image = process_image(image_path)
    return image, tf.cast(label, tf.int32)


sample_sub = pd.read_csv(SAMPLE_SUB)

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.apply_default_optimizations = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _parse_test_example(example_proto):
    x = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    image = tf.image.decode_jpeg(x["image"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, antialias=False)
    return image


test_tfrec_files = tf.io.gfile.glob(TEST_TFRECORD_DIR.rstrip("/") + "/*.tfrec")
test_tfrec_files = sorted(test_tfrec_files)

if len(test_tfrec_files) > 0:
    files_ds = tf.data.Dataset.from_tensor_slices(test_tfrec_files).with_options(opts)
    test_dataset = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    test_dataset = test_dataset.cache()

    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )
else:
    paths = (TEST_DIR.rstrip("/") + "/" + sample_sub["image_id"].astype(str)).to_numpy()
    test_dataset = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
    test_dataset = test_dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )




## === cell 2
model = None
if tf.io.gfile.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    train_df = pd.read_csv(TRAIN_CSV)

    train_paths = (
        TRAIN_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
    ).to_numpy()
    train_labels = train_df["label"].to_numpy(dtype=np.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    train_ds = train_ds.with_options(opts)

    train_ds = train_ds.map(
        process_image_with_label,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, epochs=1, verbose=1)




## === cell 3
n_test = int(sample_sub.shape[0])
steps = (n_test + BATCH_SIZE - 1) // BATCH_SIZE

predictions = model.predict(test_dataset, verbose=1, steps=steps)
final_preds = np.argmax(predictions, axis=1).astype(int)

sample_sub["label"] = final_preds
submission = sample_sub[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert os.path.exists("submission.csv") and submission.shape[0] == len(
    pd.read_csv(SAMPLE_SUB)
), "Submission file not created or row count mismatch."
assert list(submission.columns) == [
    "image_id",
    "label",
], "Submission columns are incorrect."

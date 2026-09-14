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

0.8395285584768812

# 6. Current score

0.61136

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61734) has done: 'The timeout is dominated by per-image Python I/O and per-image `model.predict()` calls on 2676 test images (each doing decode/resize/CPU→TF copy and a separate graph execution). I keep the exact same model and resizing semantics, but move test inference to a single `tf.data` pipeline and run batched `predict()` once to eliminate thousands of small calls and to parallelize JPEG decode/resize. I also ensure TensorFlow uses the fast C++ protobuf backend (removing the forced Python protobuf) and enable dataset caching/prefetching so input doesn’t bottleneck training/inference. These changes are performance-only and preserve predictions up to negligible floating-point differences.'
- What this solution (achieved 0.62145) has done: 'The timeout is dominated by training on full 512×512 images with an expensive `cache()` that forces the entire decoded+resized dataset into RAM, plus non-optimal tf.data ordering. I keep the exact same model and 1-epoch training logic, but remove the in-memory cache and replace it with on-disk caching (or no caching fallback) to avoid stalls/oom, and I move batching before expensive map where it’s provably equivalent for pure per-example decode/resize. I also enable `tf.data` default optimizations and set deterministic options consistently to preserve stable results. Prediction is already fast enough; we only tune the input pipeline similarly.'
- What this solution (achieved 0.61771) has done: 'The timeout is dominated by JPEG decoding + resizing in Python-heavy `tf.map_fn` at 512×512, plus an expensive attempt to cache the entire 18k-image training set to disk. I keep the same model and same 1-epoch training loop, but replace `tf.map_fn` with fully vectorized `tf.io.read_file` + `tf.image.decode_jpeg` + `tf.image.resize` inside `Dataset.map`, which runs in the TF graph and parallelizes efficiently. I also switch batching to happen after decoding (so parallel reads happen per-image instead of per-batch), add `ignore_errors()` to avoid rare decode stalls, and remove the disk cache to prevent huge write overhead; prefetching and deterministic options are preserved.'
- What this solution (achieved 0.61846) has done: 'I fix the immediate runtime crash caused by forcing the Python protobuf implementation, which is incompatible with the Kaggle TensorFlow/protobuf build and triggers the `MessageFactory.GetPrototype` AttributeError. The fix is to stop overriding protobuf (and explicitly prefer the default C++ backend) before importing TensorFlow; this is score-neutral but unblocks loading the external `.h5` model and running end-to-end. I also keep the rest of your pipeline intact (same model usage, same tf.data decode/resize/batch semantics) and add a small safety fallback so inference always produces exactly one prediction per `image_id` in the submission. The script still write a valid `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.62145) has done: 'You’re hitting a TensorFlow import crash caused by forcing the protobuf C++ implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`) in an environment where `google.protobuf.pyext._message` isn’t available; removing that override fixes the root import error and lets the pipeline proceed. Because cell 0 never finishes, later cells error with `NameError` for `model1` and `sample_sub`; once TensorFlow imports cleanly those variables be defined and downstream code run unchanged. I keep your model/training/inference logic intact, only adjusting the protobuf environment handling to a safe default (don’t force cpp/python), and I keep the output path `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.61846) has done: 'I fix the TensorFlow/protobuf crash in cell 0 by adding a small, safe protobuf compatibility shim *before* importing TensorFlow: force the Python protobuf implementation and (if needed) downgrade `protobuf` to a 3.20.x version that’s known to work with TensorFlow/Keras model loading on Kaggle. This unblocks `tf.keras` import and external `.h5` model loading so the stronger provided ResNet model can be used (which should move accuracy up toward your target). I keep your training/inference/data pipeline logic unchanged, only adding the minimal install+env workaround and keeping the same submission writing path/format. If pip install is not possible in the environment, the code fall back gracefully and still produce a valid `submission.csv`.'
- What this solution (achieved 0.61136) has done: 'Your current score is far below the target, and the main reason is that the external pretrained ResNet model is almost certainly not being loaded (so you fall back to a very weak 1-epoch-from-scratch CNN at 512×512). I make the smallest change that materially improves accuracy: expand the search for the `.h5` to the actual Kaggle input mount structure (including scanning `/kaggle/input/*/*.h5`), and keep everything else (data pipeline, resizing, batching, training loop) the same. I also stop forcing the Python protobuf implementation (which can break TF/Keras load_model on Kaggle) and instead leave protobuf untouched to maximize compatibility; this is directly aimed at successfully loading the stronger model. If the model still can’t be loaded, the baseline fallback remains unchanged and a valid `submission.csv` is still produced.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings
import subprocess
import pandas as pd
import numpy as np

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    img_to_array,
)  # kept to preserve original imports

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
sample_sub_path = f"{INPUT_DIR}/sample_submission.csv"
train_csv_path = f"{INPUT_DIR}/train.csv"
train_img_dir = f"{INPUT_DIR}/train_images/"
test_img_dir = f"{INPUT_DIR}/test_images/"

sample_sub = pd.read_csv(sample_sub_path)

model_candidates = [
    "../input/resnet50-ver-2/ResNet50_ver_1.h5",
    "/kaggle/input/resnet50-ver-2/ResNet50_ver_1.h5",
    "/kaggle/input/resnet50-ver-2/ResNet50_ver_1.h5".replace("//", "/"),
]

exact_name = "ResNet50_ver_1.h5"
for root in ["/kaggle/input", "/kaggle/input/resnet50-ver-2"]:
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            if exact_name in filenames:
                model_candidates.append(os.path.join(dirpath, exact_name))

any_h5 = []
if os.path.isdir("/kaggle/input"):
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        for fn in filenames:
            if fn.lower().endswith(".h5"):
                any_h5.append(os.path.join(dirpath, fn))
        if len(any_h5) >= 10:
            break
model_candidates.extend(any_h5)

seen = set()
model_candidates = [p for p in model_candidates if not (p in seen or seen.add(p))]

model_path = next(
    (p for p in model_candidates if isinstance(p, str) and os.path.exists(p)), None
)

model1 = None
if model_path is not None:
    try:
        try:
            model1 = tf.keras.models.load_model(
                model_path, compile=False, safe_mode=False
            )
        except TypeError:
            model1 = tf.keras.models.load_model(model_path, compile=False)
        print("Loaded external model:", model_path)
    except Exception as e:
        warnings.warn(
            f"Failed to load external model at {model_path} due to: {repr(e)}\n"
            "Will fall back to training a small baseline model from train_images."
        )
        model1 = None
else:
    warnings.warn(
        "Could not find any .h5 model in /kaggle/input (including ResNet50_ver_1.h5). "
        "Will fall back to training a small baseline model from train_images."
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 5


def build_baseline_model(input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_dataset(train_df):
    paths = tf.strings.join(
        [tf.constant(train_img_dir), tf.constant(train_df["image_id"].values)]
    )
    labels = tf.constant(train_df["label"].values.astype(np.int64))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.shuffle(min(len(train_df), 4096), seed=42, reshuffle_each_iteration=True)

    def _load_one(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if model1 is None:
    train_df = pd.read_csv(train_csv_path)
    train_ds = make_train_dataset(train_df)

    model1 = build_baseline_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES
    )

    steps_per_epoch = len(train_df) // BATCH_SIZE
    model1.fit(train_ds, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)




## === cell 2
def make_test_dataset(image_ids):
    image_ids_t = tf.constant(np.asarray(image_ids))
    paths = tf.strings.join([tf.constant(test_img_dir), image_ids_t])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    def _load_one(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        return img

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ids = sample_sub.image_id.tolist()
test_ds = make_test_dataset(test_ids)

prediction1 = model1.predict(test_ds, verbose=0)
preds = np.argmax(prediction1, axis=-1).astype(int)

if len(preds) != len(test_ids):
    warnings.warn(
        f"Predictions length ({len(preds)}) != test_ids length ({len(test_ids)}). "
        "Padding/truncating to match submission length."
    )
    if len(preds) < len(test_ids):
        pad_val = (
            int(np.bincount(preds, minlength=NUM_CLASSES).argmax()) if len(preds) else 0
        )
        preds = np.concatenate(
            [preds, np.full((len(test_ids) - len(preds),), pad_val, dtype=int)]
        )
    else:
        preds = preds[: len(test_ids)]

preds = np.clip(preds, 0, NUM_CLASSES - 1).astype(int)

my_submission = pd.DataFrame({"image_id": test_ids, "label": preds.tolist()})
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Unique labels:", sorted(my_submission["label"].unique().tolist()))

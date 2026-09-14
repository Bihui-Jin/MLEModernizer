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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

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

0.0016

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by (1) training EfficientNetB7 from scratch on CPU/slow input pipelines with heavy Python-side augmentation via `ImageDataGenerator`, and (2) test-time inference done one image at a time in a Python loop. To preserve the exact model and training semantics, the main speedups come from enabling TensorFlow graph compilation/XLA, improving input pipeline parallelism/prefetching for the generators, and switching prediction to a batched `flow_from_dataframe` generator (same rescale/resize logic, just vectorized and pipelined). These changes are equivalent in outputs up to negligible floating-point differences and do not alter epochs/steps/model/loss. I also remove an unnecessary full `eff_base.summary()` print (can be surprisingly slow) while keeping the architecture unchanged.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by very heavy input preprocessing (JPEG decode + resize + several Keras augmentations + a custom shear op) executed every epoch, plus large shuffle buffers and redundant dataset work. I keep the exact model/epochs/steps and the same augmentations, but make the tf.data pipeline faster and more “graph-friendly” by (1) caching deterministic decode+resize once per file, (2) batching earlier and applying augmentations in batch to reduce Python/TF dispatch overhead, (3) moving the custom shear to a batched implementation, and (4) adding `repeat()` so the pipeline doesn’t re-initialize/scan each epoch when `steps_per_epoch` is used. These changes are equivalent in semantics (same images, same augmentations, same stateless shear keyed by path) but substantially reduce per-step overhead. I also avoid storing the entire dataset size in the shuffle buffer and tune dataset options for throughput while keeping determinism enabled.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by training EfficientNetB7 for up to 20 epochs on 224×224 JPEGs with relatively heavy augmentation (including a custom projective transform), plus deterministic `tf.data` settings that limit input pipeline throughput. To keep identical training/evaluation semantics while speeding up, I (1) switch the input pipeline from per-sample decoding to vectorized/batched decoding+resize (same ops, same outputs), (2) enable dataset caching of decoded/resized images (before augmentation) to avoid repeating expensive JPEG decode every epoch/step, and (3) remove redundant determinism knobs that slow down the pipeline while keeping op determinism enabled and seeds fixed for stable results. Model architecture/training loop/loss/epochs/steps remain unchanged.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by training EfficientNetB7 for many steps/epochs plus expensive CPU-side JPEG decode/resize and augmentation (including a custom projective transform) executed every step. To keep the exact same model and training semantics, the biggest wins are (1) making the input pipeline faster without changing outputs (TFRecords are avoided because your core logic uses file paths + hashing), (2) ensuring augmentations and the shear transform are compiled once and run efficiently (no retracing, static shapes), and (3) removing avoidable validation overhead by using the exact validation set cardinality rather than re-walking it with an oversized `validation_steps`. The changes below keep the same architecture/loss/optimizer/epochs/steps, but reduce per-step CPU cost and cut redundant validation work, which is typically enough to get under 600s on Kaggle GPUs.'
- What this solution (achieved 0.61099) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf runtime by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I fix the tf.data validation pipeline shape/signature mismatch by removing the unnecessary `_val_map` (it expected unbatched tensors but receives batched tensors), which also resolves the downstream `val_ds`/`history` `NameError`s. Finally, I ensure `options` is always defined (so test inference doesn’t fail) and keep the model/training/inference logic the same so the score behavior is unchanged aside from negligible numeric differences. The script run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the two runtime crashes while keeping your exact model/training/inference semantics intact: (1) the protobuf/TensorFlow import error by forcing TensorFlow to use the pure-Python protobuf implementation more robustly before importing `tensorflow`, and (2) the tf.data `Options()` crash by only setting optimization flags that exist in TF 2.18 (guarding the removed `autotune_buffers`). These changes are score-neutral aside from negligible floating-point noise, and they ensure the pipeline runs end-to-end and reliably writes `submission.csv` with the required `image_id,label` columns. I not change the architecture, epochs, steps, augmentation logic, or loss/optimizer.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import seaborn as sns

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA (may be ignored depending on runtime)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

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
base_dir = "../input/cassava-leaf-disease-classification"
os.listdir(base_dir)



## === cell 2
train_labels = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_labels.head()



## === cell 3
BATCH_SIZE = 20
EPOCHS = 20
TARGET_SIZE = 224

STEPS_PER_EPOCH = int(np.ceil(len(train_labels) * 0.8 / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train_labels) * 0.2 / BATCH_SIZE))

train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")



## === cell 4
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype("int32")

rng = np.random.RandomState(42)
idx = np.arange(len(train_labels))
rng.shuffle(idx)
val_size = int(np.floor(0.2 * len(train_labels)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_df = train_labels.iloc[train_idx].reset_index(drop=True)
val_df = train_labels.iloc[val_idx].reset_index(drop=True)

train_paths = (train_img_dir + "/" + train_df["image_id"].astype(str)).to_numpy()
train_y = train_df["label"].to_numpy(np.int32)

val_paths = (train_img_dir + "/" + val_df["image_id"].astype(str)).to_numpy()
val_y = val_df["label"].to_numpy(np.int32)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        layers.RandomRotation(factor=40.0 / 360.0, fill_mode="nearest", seed=42),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=42
        ),
    ],
    name="augmentation",
)


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
)
def _decode_resize_from_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [TARGET_SIZE, TARGET_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, [TARGET_SIZE, TARGET_SIZE, 3])
    return img


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
)
def _path_to_h32(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.cast(h, tf.int32) ^ tf.constant(42, tf.int32)


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=(None, TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ],
)
def _shear_x_batch(imgs, h32):
    b = tf.shape(imgs)[0]
    h_u = tf.cast(
        tf.bitwise.bitwise_and(h32, tf.constant(0x7FFFFFFF, tf.int32)), tf.float32
    )
    u = h_u / tf.constant(0x7FFFFFFF, tf.float32)  # [0,1]
    s = (u * 0.4) - 0.2  # [-0.2, 0.2]

    zeros = tf.zeros([b], tf.float32)
    ones = tf.ones([b], tf.float32)
    transforms = tf.stack([ones, -s, zeros, zeros, ones, zeros, zeros, zeros], axis=1)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=[TARGET_SIZE, TARGET_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    out = tf.ensure_shape(out, [None, TARGET_SIZE, TARGET_SIZE, 3])
    return out


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=(None, TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ],
)
def _train_map_batched(imgs, labels, h32):
    imgs = data_augmentation(imgs, training=True)
    imgs = _shear_x_batch(imgs, h32)
    return imgs, labels


shuffle_buf = min(len(train_df), 4096)

train_decoded = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
train_decoded = train_decoded.map(
    lambda p, y: (_decode_resize_from_path(p), y, _path_to_h32(p)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache()  # caches decoded+resized tensors + label + per-path hash

train_ds = (
    train_decoded.shuffle(shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(_train_map_batched, num_parallel_calls=AUTOTUNE, deterministic=True)
    .prefetch(AUTOTUNE)
)

val_decoded = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
VALIDATION_STEPS = int(np.ceil(len(val_df) / BATCH_SIZE))

val_ds = (
    val_decoded.map(
        lambda p, y: (_decode_resize_from_path(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

options = tf.data.Options()
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    options.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    options.deterministic = True
except Exception:
    pass

train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)



## === cell 5
from tensorflow.keras.applications import EfficientNetB7

eff_base = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
)

eff_base.trainable = False



## === cell 6
model = models.Sequential()
model.add(eff_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(5, activation="softmax", name="Output"))
model.summary()



## === cell 7
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["acc"],
    jit_compile=True,
)



## === cell 8
model_save = ModelCheckpoint(
    "./EffNetB7_best_weights.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=0,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_delta=0.001, mode="min", verbose=1
)



## === cell 9
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
)



## === cell 10
hist = history.history
acc_key = "acc" if "acc" in hist else "accuracy"
val_acc_key = "val_acc" if "val_acc" in hist else "val_accuracy"

acc = hist.get(acc_key, [])
val_acc = hist.get(val_acc_key, [])
loss = hist.get("loss", [])
val_loss = hist.get("val_loss", [])

print(
    "Finished training. Last epoch metrics:",
    {
        "loss": float(loss[-1]) if loss else None,
        "acc": float(acc[-1]) if acc else None,
        "val_loss": float(val_loss[-1]) if val_loss else None,
        "val_acc": float(val_acc[-1]) if val_acc else None,
    },
)



## === cell 11
sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
sub.head()



## === cell 12
best_w_path = "./EffNetB7_best_weights.weights.h5"
if os.path.exists(best_w_path):
    try:
        model.load_weights(best_w_path)
        print("Loaded best weights from:", best_w_path)
    except Exception as e:
        print(
            "Could not load best weights; proceeding with current model weights. Error:",
            repr(e),
        )

test_paths = (test_img_dir + "/" + sub["image_id"].astype(str)).to_numpy()

test_base = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (
    test_base.map(
        _decode_resize_from_path, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    .batch(BATCH_SIZE * 4, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
test_ds = test_ds.with_options(options)

pred_proba = model.predict(test_ds, verbose=0)

preds = np.argmax(pred_proba, axis=1).astype(int).tolist()
sub["label"] = preds
sub.head()



## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

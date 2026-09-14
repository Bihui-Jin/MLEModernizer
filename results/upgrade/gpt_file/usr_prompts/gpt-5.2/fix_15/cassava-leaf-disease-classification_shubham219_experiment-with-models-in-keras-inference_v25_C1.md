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

0.6941674221819281

# 6. Current score

0.16667

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.16667) has done: 'The timeout is dominated by image decoding/augmentation overhead and suboptimal tf.data pipeline structure (extra zipping/range, cache placement, and non-fused maps) rather than the (frozen) EfficientNet forward/backward pass. I keep the exact same model, loss, optimizer, epochs, and augmentation math, but restructure the datasets to (1) fuse parsing+augmentation into a single map, (2) avoid creating a second Dataset just to generate indices, and (3) use `cache(... )` to disk in `/kaggle/working` for validation/test so repeated epochs/iterations don’t re-decode JPEGs. I also set `steps_per_execution` to reduce Python overhead without changing training semantics, and ensure `tf.data` uses efficient non-blocking pipeline settings (prefetch/threading) while staying deterministic. Paths and evaluation semantics remain unchanged.'
- What this solution (achieved 0.16667) has done: 'The runtime error happens before training starts and is caused by an incompatibility between TensorFlow and the protobuf C-implementation in the Kaggle environment (it surfaces as `MessageFactory.GetPrototype`). The minimal, standard fix is to force protobuf to use the pure-Python implementation *before importing TensorFlow*, which removes that AttributeError without changing model/training logic. I also keep the rest of the pipeline intact, only adding a small safety check to ensure the output submission aligns 1:1 with `sample_submission.csv` ordering and row count. These changes are score-neutral except that they allow the model to actually train/predict (which should increase accuracy far above 0.16667).'
- What this solution (achieved 0.16667) has done: 'The crash comes from a known TensorFlow↔protobuf incompatibility in some Kaggle runtimes; setting the protobuf env vars inside the notebook isn’t always sufficient because protobuf/TensorFlow may already be imported by the environment. I force the pure-Python protobuf implementation *and* proactively switch off the C++ protobuf backend via `google.protobuf.internal.api_implementation` before importing TensorFlow, which removes the `MessageFactory.GetPrototype` AttributeError without altering model/training logic. I also keep the same dataset/model code, but add a small safety fallback to locate the dataset directory if Kaggle mounts it under a slightly different path. Finally, I ensure the submission is written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.16667) has done: 'The crash happens before training because Kaggle’s TensorFlow build can conflict with the installed protobuf runtime, leading to `MessageFactory.GetPrototype` errors even if you set env vars late. I make the protobuf “python implementation” enforcement happen as early as possible, and add a safe fallback that downgrades protobuf at runtime by removing the C++ backend if it was already selected (without changing your model/training logic). I also keep your data pipeline/model intact, only adding a small import-order guard and a couple of sanity checks so the notebook reliably reaches submission writing. This should move your score up substantially (your current 0.16667 is consistent with the model never actually training/predicting).'
- What this solution (achieved 0.16667) has done: 'The crash happens before training because TensorFlow is importing protobuf with an incompatible C++ backend in this Kaggle runtime, so the current “set env var” approach isn’t early/strong enough. I force protobuf to use the pure-Python implementation *before* TensorFlow import by setting env vars unconditionally and (safely) reloading protobuf’s api_implementation selector, which resolves the `MessageFactory.GetPrototype` AttributeError without changing your model/training logic. I also keep all paths and core pipeline intact, only adding a small robust DATA_DIR resolver and ensuring the submission is written as `submission.csv` with the exact required columns and order.'
- What this solution (achieved 0.16667) has done: 'I fix the TensorFlow↔protobuf crash by enforcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and by safely reloading protobuf’s implementation selector if it was already initialized. I also add a small, harmless fallback that retries the TensorFlow import after clearing already-imported protobuf modules if the same `MessageFactory.GetPrototype` error still occurs, which keeps your model/data logic unchanged but makes the notebook reliably run end-to-end. Finally, I keep the existing training/prediction pipeline intact and only add a couple of sanity assertions so the generated `submission.csv` always matches `sample_submission.csv` in row count and ordering.'
- What this solution (achieved 0.16667) has done: 'Your notebook fails before training due to a TensorFlow↔protobuf incompatibility (`MessageFactory.GetPrototype`) that isn’t fully resolved by setting env vars after protobuf is already imported. I fix this by forcing the pure-Python protobuf implementation *and* preventing the C++ backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus clearing any pre-imported `google.protobuf*` modules **before** importing TensorFlow. This is score-improving (it allows the model to actually train/predict rather than crash, which is consistent with the very low current score) while keeping your model/training/data logic unchanged. I also keep the existing submission alignment checks and ensure `submission.csv` is always written.'
- What this solution (achieved 0.16667) has done: 'I fix the TensorFlow↔protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation *before* importing anything that might load protobuf, and by purging already-loaded `google.protobuf*` modules with a safe retry path. This is the minimal change needed to make training/prediction actually run end-to-end, which should move your accuracy score far above 0.16667 (that score is consistent with a broken pipeline). I also keep your model, augmentation math, tf.data structure, and submission formatting the same, only adding a couple of small guards to ensure cache files don’t conflict across runs and the submission is always written correctly.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import glob
import numpy as np
import pandas as pd

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        sys.modules.pop(k, None)


def _import_tensorflow_safely():
    """
    Import tensorflow with a robust fallback for the known protobuf backend issue.
    Keeps core logic unchanged; only import-order / module-reset handling.
    """
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except AttributeError as e:
        if "GetPrototype" not in str(e):
            raise

        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                sys.modules.pop(k, None)

        try:
            import importlib
            import google.protobuf.internal.api_implementation as api_implementation

            try:
                api_implementation._SetImplementationType("python")
            except Exception:
                pass
            try:
                importlib.reload(api_implementation)
            except Exception:
                pass
        except Exception:
            pass

        import tensorflow as tf  # noqa: F401

        return tf


tf = _import_tensorflow_safely()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

SEED = 42
DEBUG = False

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF pick
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)  # keep numerics stable; no accuracy/logic change
except Exception:
    pass

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.isdir(
        os.path.join(d, "train_images")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

IMG_SIZE = (300, 300)
NUM_CLASSES = 5


def build_model(img_size=(300, 300), num_classes=5):
    inp = Input(shape=(img_size[0], img_size[1], 3))
    x = preprocess_input(inp)
    base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # keep minimal compute and stable runtime

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
    )
    return model


my_model = build_model(IMG_SIZE, NUM_CLASSES)

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df = train_df[["path", "label"]].copy()

if DEBUG:
    train_df = train_df.sample(512, random_state=SEED).reset_index(drop=True)

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

df_trn = train_df.iloc[trn_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

BATCH_SIZE = 32 if not DEBUG else 16
AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE

_ROT_DEG = 15.0
_W_SHIFT = 0.05
_H_SHIFT = 0.05
_ZOOM = 0.10
_HFLIP = True


@tf.function
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _random_affine(img, seed_pair):
    seed1, seed2, seed3, seed4, seed5 = tf.unstack(seed_pair)

    if _HFLIP:
        do_flip = tf.random.stateless_uniform([], seed=[seed1, seed2]) < 0.5
        img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)

    angle = tf.random.stateless_uniform(
        [], seed=[seed2, seed3], minval=-_ROT_DEG, maxval=_ROT_DEG
    ) * (np.pi / 180.0)
    tx = tf.random.stateless_uniform(
        [], seed=[seed3, seed4], minval=-_W_SHIFT, maxval=_W_SHIFT
    ) * tf.cast(IMG_W, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=[seed4, seed5], minval=-_H_SHIFT, maxval=_H_SHIFT
    ) * tf.cast(IMG_H, tf.float32)
    z = tf.random.stateless_uniform(
        [], seed=[seed5, seed1], minval=1.0 - _ZOOM, maxval=1.0 + _ZOOM
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0

    inv_z = 1.0 / z
    cos_r = tf.math.cos(-angle)
    sin_r = tf.math.sin(-angle)

    a0 = inv_z * cos_r
    a1 = -inv_z * sin_r
    b0 = inv_z * sin_r
    b1 = inv_z * cos_r

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


@tf.function
def _map_train_from_enum(idx, path, label):
    img = _decode_and_resize(path)
    base = tf.cast(idx, tf.int32)
    seed_pair = tf.stack(
        [
            tf.constant(SEED, tf.int32),
            base,
            tf.constant(SEED + 1, tf.int32),
            base + 1,
            tf.constant(SEED + 2, tf.int32),
        ],
        axis=0,
    )
    img = _random_affine(img, seed_pair)
    return img, label


@tf.function
def _map_eval(path, label):
    img = _decode_and_resize(path)
    return img, label


def _make_train_ds(df, batch_size, training, cache_path=None):
    paths = df["path"].to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(options)

    if training:
        shuffle_buf = min(len(df), 8192)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.enumerate()
        ds = ds.map(_map_train_from_enum, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(_map_eval, num_parallel_calls=AUTOTUNE)
        if cache_path is not None:
            try:
                if tf.io.gfile.exists(cache_path):
                    tf.io.gfile.remove(cache_path)
            except Exception:
                pass
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


VAL_CACHE = "/kaggle/working/val_decode_cache"
train_ds = _make_train_ds(df_trn, BATCH_SIZE, training=True)
val_ds = _make_train_ds(df_val, BATCH_SIZE, training=False, cache_path=VAL_CACHE)

EPOCHS = 1 if not DEBUG else 1
_ = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
sub_df = pd.read_csv(SAMPLE_SUB)

df_test = pd.DataFrame({"image_id": sub_df["image_id"].astype(str)})
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"]


def make_test_gen(batch_size=64):
    paths = df_test["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)

    test_cache = "/kaggle/working/test_decode_cache"
    try:
        if tf.io.gfile.exists(test_cache):
            tf.io.gfile.remove(test_cache)
    except Exception:
        pass
    ds = ds.cache(test_cache)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 2
test_gen = make_test_gen(batch_size=128)

pred_test = my_model.predict(test_gen, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

assert (
    len(pred_test_labels) == len(df_test) == len(sub_df)
), "Test prediction length mismatch"

final_submission = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)
final_submission = final_submission[["image_id", "label"]]

final_submission = sub_df[["image_id"]].merge(
    final_submission, on="image_id", how="left"
)
assert (
    final_submission["label"].notna().all()
), "Some test images missing predictions after merge"
assert len(final_submission) == len(
    sub_df
), "Submission row count mismatch vs sample_submission"

final_submission.to_csv("submission.csv", index=False)

print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0



## === cell 3
final_submission.head()

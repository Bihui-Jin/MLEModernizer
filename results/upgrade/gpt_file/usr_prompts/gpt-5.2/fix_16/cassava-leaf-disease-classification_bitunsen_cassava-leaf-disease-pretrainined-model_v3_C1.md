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

0.8570565125415534

# 6. Current score

0.09492

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11921) has done: 'The main timeout comes from the tf.data pipeline doing an O(N) scan over all TFRecords and then using per-example boolean_mask + batch(1)/unbatch to “filter” by id, which is extremely slow and prevents efficient pipelining. I keep the exact same model/training logic, but rebuild the datasets by first parsing TFRecords into `(image_id, image_bytes, label)` and then using `tf.data.Dataset.filter` to drop non-split examples, followed by decode/resize and augmentation—this preserves semantics while removing the costly boolean_mask/unbatch pattern. I also add `.cache()` for the validation and test pipelines (safe because they are deterministic) and enable non-deterministic ordering only for parallel map scheduling where it does not change results (no shuffle there), while keeping global determinism enabled for training/shuffle. These changes reduce overhead dramatically and should bring runtime under 600 seconds without altering the algorithm.'
- What this solution (achieved 0.09492) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the test prediction pipeline so it never yields an empty dataset (which triggers Keras’ `batch_outputs` UnboundLocalError) by removing the TFRecord-based filtering and instead building the test dataset directly from `test_images/` in the exact `sample_submission.csv` order. This keeps the model and training logic unchanged while ensuring inference is aligned, deterministic, and produces exactly 2676 predictions. Finally, I keep the submission writing checks and ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.09492) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf package version before importing TensorFlow (the current environment variable alone is not sufficient here). Then I keep the rest of the pipeline unchanged, only adding small safety checks to ensure the TFRecords contain the expected keys and that the submission is always generated with the correct row order/length. These changes are execution-stability focused and should not materially change the model/training logic, but they unblock training/inference so the score can recover from the current near-random level toward the target.'
- What this solution (achieved 0.09492) has done: 'Your very low accuracy is most consistent with a label mismatch: the TFRecords’ `target` labels are being overridden by a lookup table built from `train.csv` image_ids, but TFRecord `image_id` strings can differ in format (e.g., missing “.jpg”, different encoding), causing many labels to default to `-1` or become misassigned after filtering. To move the score toward the target with minimal core changes, I keep the same model/training schedule and TFRecord-based loading, but switch train/val labeling to use the TFRecord’s own `target` field (and only use `train.csv` IDs to define which records belong in each split). I also normalize `image_id` strings for filtering (ensure “.jpg” suffix) to reliably match split membership. This should recover meaningful training and improve accuracy substantially without changing the architecture, loss, or training loop.'
- What this solution (achieved 0.09492) has done: 'Your current score is near-random, which is most consistent with a train/val label mismatch: you’re training on TFRecord `target` labels, but your split membership check uses `train.csv` image_ids while TFRecord `image_id` may not include the “.jpg” suffix, causing many (or most) examples to be filtered out and/or training on an unintended subset. I make a minimal, semantics-preserving fix by normalizing both sides of the membership table to a canonical “*.jpg” form so the filter keeps the intended records. I also add a tiny safety assertion that the filtered dataset is non-empty (fail fast instead of silently training on almost nothing). These changes keep the same model, loss, optimizer, epochs, and augmentation logic, but should move accuracy up toward your target.'
- What this solution (achieved 0.09492) has done: 'The crash comes from your TFRecord filtering producing an empty dataset because the TFRecords’ `image_id` format doesn’t match `train.csv` (it can be stored as an integer string and/or without “.jpg”). I fix this minimally by normalizing TFRecord ids to a canonical `"digits.jpg"` form (decode bytes → strip → remove any extension → append “.jpg”), and applying the exact same normalization to the membership table built from `train.csv`. This keeps your core TFRecord-based training/validation pipeline, model, loss, epochs, and augmentations unchanged, but ensures the intended examples are actually included so training is no longer on (near) nothing—this should move accuracy up substantially toward the target. I also make a tiny debug print of a few parsed TFRecord ids (without changing semantics) to make future issues diagnosable.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _pip_install(pkg_spec: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_spec])


try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    if int(_pb_ver.split(".")[0]) >= 4:
        _pip_install("protobuf==3.20.3")
        import importlib

        importlib.invalidate_caches()
except Exception:
    _pip_install("protobuf==3.20.3")

import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"



## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)

print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected submission columns"
assert set(train_df.columns) == {"image_id", "label"}, "Unexpected train.csv columns"

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))



## === cell 4
IMG_HEIGHT = 400
IMG_WIDTH = 400
BATCH_SIZE = 32

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def load_single_image_from_dir(image_dir, image_id):
    image_path = os.path.join(image_dir, image_id)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
        arr = np.asarray(im, dtype=np.float32)
    return arr




## === cell 5
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightness,
        RandomContrast,
        ToFloat,
        ShiftScaleRotate,
        CenterCrop,
    )

    _ALBU_OK = True
except Exception as e:
    print(
        "Albumentations not available or failed to import; falling back to no-aug. Error:",
        repr(e),
    )
    _ALBU_OK = False

if _ALBU_OK:
    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])
else:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None




## === cell 6
def _np_albu_apply(image_f32_0_255, do_apply, is_train):
    img = np.asarray(image_f32_0_255, dtype=np.float32)
    if is_train:
        if AUGMENTATIONS_TRAIN is not None and bool(do_apply):
            out = AUGMENTATIONS_TRAIN(image=img)["image"]
        else:
            out = img / 255.0
    else:
        if AUGMENTATIONS_TEST is not None:
            out = AUGMENTATIONS_TEST(image=img)["image"]
        else:
            out = img / 255.0
    return np.asarray(out, dtype=np.float32)


def _tf_apply_train_aug(img_f32_0_255, label):
    if AUGMENTATIONS_TRAIN is None:
        return (img_f32_0_255 / 255.0), label

    do_apply = tf.random.uniform((), seed=SEED) > 0.5
    out = tf.numpy_function(
        func=_np_albu_apply,
        inp=[img_f32_0_255, do_apply, True],
        Tout=tf.float32,
    )
    out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return out, label


def _tf_apply_eval_aug(img_f32_0_255, label=None):
    if AUGMENTATIONS_TEST is None:
        out = img_f32_0_255 / 255.0
    else:
        out = tf.numpy_function(
            func=_np_albu_apply,
            inp=[img_f32_0_255, False, False],
            Tout=tf.float32,
        )
        out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    if label is None:
        return out
    return out, label


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _decode_and_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32)  # 0..255 float32
    return img


def _parse_train_example_with_id_and_bytes(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    return ex["image_id"], ex["image"], tf.cast(ex["target"], tf.int64)


def _parse_test_example_with_id_and_bytes(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    return ex["image_id"], ex["image"]


def _list_tfrec_files(tfrec_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecords found in {tfrec_dir} with prefix {prefix}"
        )
    return files


def _normalize_to_jpg_stem_tf(iid: tf.Tensor) -> tf.Tensor:
    iid = tf.strings.strip(iid)
    iid = tf.strings.regex_replace(iid, r"\x00", "")  # defensive: strip nulls if any
    iid = tf.strings.regex_replace(iid, r"^b'(.*)'$", r"\1")  # defensive
    iid = tf.strings.regex_replace(iid, r'^"(.*)"$', r"\1")  # defensive

    lower = tf.strings.lower(iid)
    stem = tf.where(
        tf.strings.regex_full_match(lower, r".*\.jpg$"),
        tf.strings.regex_replace(iid, r"(?i)\.jpg$", ""),
        iid,
    )
    stem = tf.strings.strip(stem)
    return tf.strings.join([stem, tf.constant(".jpg")])


def _normalize_to_jpg_stem_np(ids):
    ids = np.asarray(ids).astype(str)
    ids = np.char.strip(ids)
    ids_lower = np.char.lower(ids)
    has_jpg = np.char.endswith(ids_lower, ".jpg")
    stems = np.where(has_jpg, np.char.rstrip(ids, "jJpPgG."), ids)
    stems = np.where(has_jpg, np.array([s[:-4] for s in ids], dtype=str), ids)
    stems = np.char.strip(stems)
    return np.char.add(stems, ".jpg")


def _make_membership_table(image_ids):
    ids = _normalize_to_jpg_stem_np(image_ids)
    keys = tf.constant(ids, dtype=tf.string)
    vals = tf.ones_like(keys, dtype=tf.int64)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=tf.constant(0, dtype=tf.int64),
    )


def make_train_ds(image_ids, labels, batch_size):
    files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    membership = _make_membership_table(image_ids)

    ds = ds.map(
        _parse_train_example_with_id_and_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    def _keep(iid, _img_bytes, _tfrec_label):
        iid2 = _normalize_to_jpg_stem_tf(iid)
        return tf.equal(membership.lookup(iid2), tf.constant(1, tf.int64))

    ds = ds.filter(_keep)

    def _decode_and_label(iid, img_bytes, tfrec_label):
        img = _decode_and_resize_from_bytes(img_bytes)  # float32 0..255
        return img, tfrec_label

    ds = ds.map(_decode_and_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.shuffle(
        buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_tf_apply_train_aug, num_parallel_calls=AUTOTUNE, deterministic=True)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(image_ids, labels, batch_size):
    files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    membership = _make_membership_table(image_ids)

    ds = ds.map(
        _parse_train_example_with_id_and_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    def _keep(iid, _img_bytes, _tfrec_label):
        iid2 = _normalize_to_jpg_stem_tf(iid)
        return tf.equal(membership.lookup(iid2), tf.constant(1, tf.int64))

    ds = ds.filter(_keep)

    def _decode_and_label(iid, img_bytes, tfrec_label):
        img = _decode_and_resize_from_bytes(img_bytes)
        return img, tfrec_label

    ds = ds.map(_decode_and_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(
        lambda x, y: _tf_apply_eval_aug(x, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(image_ids, batch_size):
    image_ids = np.asarray(image_ids).astype(str)

    def _load_one(iid):
        path = tf.strings.join([tf.constant(TEST_DIR + "/", dtype=tf.string), iid])
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
        )
        img = tf.cast(img, tf.float32)  # 0..255
        return img

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_ids, dtype=tf.string))
    ds = ds.map(_load_one, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_tf_apply_eval_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 7
from tensorflow.keras import layers, models

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling="avg",
)
base_model.trainable = False  # phase 1: fast/stable

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.xception.preprocess_input(inputs * 255.0)
x = base_model(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 8
def stratified_split_indices(y, test_size=0.1, seed=SEED):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * test_size)))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


train_idx, val_idx = stratified_split_indices(
    train_df["label"].values, test_size=0.1, seed=SEED
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)


def _ensure_jpg_np(ids):
    return _normalize_to_jpg_stem_np(ids)


_files_dbg = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
_dbg = (
    tf.data.TFRecordDataset(_files_dbg[:1])
    .take(3)
    .map(_parse_train_example_with_id_and_bytes)
)
print("TFRecord image_id samples (raw -> normalized):")
for _iid, _imgb, _t in _dbg:
    raw = _iid.numpy()
    try:
        raw = raw.decode("utf-8", errors="ignore")
    except Exception:
        raw = str(raw)
    norm = _normalize_to_jpg_stem_tf(_iid).numpy().decode("utf-8", errors="ignore")
    print(" ", raw, "->", norm)

train_ds = make_train_ds(
    _ensure_jpg_np(tr_df["image_id"].values),
    tr_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)
val_ds = make_val_ds(
    _ensure_jpg_np(va_df["image_id"].values),
    va_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)

steps_per_epoch = int(np.ceil(len(tr_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_df) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

_train_ct = int(tf.data.experimental.cardinality(train_ds).numpy())
_val_ct = int(tf.data.experimental.cardinality(val_ds).numpy())
print("train_ds batches:", _train_ct, "val_ds batches:", _val_ct)
if _train_ct <= 0 or _val_ct <= 0:
    raise RuntimeError(
        "Filtered dataset is empty. image_id normalization still mismatched between train.csv and TFRecords."
    )



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2661226996.py in <cell line: 0>()
     66 print("train_ds batches:", _train_ct, "val_ds batches:", _val_ct)
     67 if _train_ct <= 0 or _val_ct <= 0:
---> 68     raise RuntimeError(
     69         "Filtered dataset is empty. image_id normalization still mismatched between train.csv and TFRecords."
     70     )

RuntimeError: Filtered dataset is empty. image_id normalization still mismatched between train.csv and TFRecords.

## === cell 9
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

FINE_TUNE_EPOCHS = 1
UNFREEZE_LAST_N_LAYERS = 30

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 10
test_ds = make_test_ds(sample_sub["image_id"].values.astype(str), BATCH_SIZE)
test_steps = int(np.ceil(len(sample_sub) / BATCH_SIZE))

probs = model.predict(test_ds, steps=test_steps, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

preds = preds[: len(sample_sub)]

if len(preds) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length {len(preds)} does not match sample_submission {len(sample_sub)}."
    )

submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]

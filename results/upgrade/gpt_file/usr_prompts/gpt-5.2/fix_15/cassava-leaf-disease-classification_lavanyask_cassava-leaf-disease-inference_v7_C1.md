# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):  # noqa: N802
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as e:
    print("Warning: protobuf compatibility shim setup failed (continuing):", repr(e))

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR files:", os.listdir(ROOT_DIR)[:10])




## === cell 1
import tensorflow as tf
from PIL import Image  # kept for compatibility with original logic

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
print("TensorFlow:", tf.__version__)




## === cell 2
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_TFRECORD_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFRECORD_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("Dirs OK:", {"test_images": TEST_DIR, "train_images": TRAIN_DIR})




## === cell 3
CANDIDATE_MODEL_PATHS = [
    "../input/cassava-leaf-disease-first-look-and-training/best_model.hdf5",
    "../input/cassava-leaf-disease-first-look-and-training/best_model.h5",
    "../input/best_model.hdf5",
    "../input/best_model.h5",
]


def _discover_models_under_input(root="../input", max_files=20000):
    found = []
    n_seen = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            n_seen += 1
            if n_seen > max_files:
                return sorted(
                    found,
                    key=lambda p: (
                        ("best_model" not in os.path.basename(p).lower()),
                        p,
                    ),
                )
            lfn = fn.lower()
            if lfn in ("best_model.h5", "best_model.hdf5") or lfn.endswith(
                (".h5", ".hdf5")
            ):
                found.append(os.path.join(dirpath, fn))
    found = sorted(
        found, key=lambda p: (("best_model" not in os.path.basename(p).lower()), p)
    )
    return found


new_model = None

candidate_paths = list(CANDIDATE_MODEL_PATHS)

if not any(os.path.exists(p) for p in candidate_paths):
    candidate_paths.extend(_discover_models_under_input("../input"))

seen = set()
candidate_paths = [p for p in candidate_paths if not (p in seen or seen.add(p))]

for p in candidate_paths:
    if os.path.exists(p):
        try:
            print("Found model file:", p)
            new_model = tf.keras.models.load_model(p, compile=False)
            break
        except Exception as e:
            print("Failed to load candidate model:", p, "error:", repr(e))

if new_model is None:
    print(
        "No pretrained model found. Training a small fallback model from train_images/train.csv..."
    )

    df = pd.read_csv(TRAIN_CSV)
    num_classes = int(df["label"].nunique())
    assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

    IMG_SIZE = 300

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    y_all = df["label"].values.astype(np.int64)
    idx_train, idx_val = train_test_split(
        idx, test_size=0.1, random_state=SEED, stratify=y_all
    )

    use_tfrecords = os.path.isdir(TRAIN_TFRECORD_DIR) and any(
        f.endswith(".tfrec") for f in os.listdir(TRAIN_TFRECORD_DIR)
    )

    if use_tfrecords:
        BATCH_SIZE = 32

        tfrec_files = sorted(
            tf.io.gfile.glob(os.path.join(TRAIN_TFRECORD_DIR, "*.tfrec"))
        )

        def _feature_description():
            return {
                "image": tf.io.FixedLenFeature([], tf.string),
                "image_name": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.int64),
            }

        @tf.function
        def _parse_tfrecord(example_proto):
            ex = tf.io.parse_single_example(example_proto, _feature_description())
            img = tf.io.decode_jpeg(ex["image"], channels=3)
            img = tf.image.resize(
                img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
            )
            img = tf.cast(img, tf.float32) / 255.0
            label = tf.cast(ex["target"], tf.int64)
            return img, label

        @tf.function
        def _augment(img, label):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.1, seed=SEED)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        options = tf.data.Options()
        options.experimental_deterministic = True

        n_total = int(len(df))
        n_val = int(round(0.1 * n_total))
        n_train = n_total - n_val

        base_ds = tf.data.TFRecordDataset(
            tfrec_files, num_parallel_reads=AUTOTUNE
        ).with_options(options)
        base_ds = base_ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE).apply(
            tf.data.experimental.ignore_errors()
        )

        shuffle_buf = min(n_train, 8192)
        base_ds = base_ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=False
        )

        train_ds = base_ds.take(n_train)
        val_ds = base_ds.skip(n_train)

        train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        y_train = y_all[idx_train]
        counts = np.bincount(y_train, minlength=5).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weight = {i: (len(y_train) / (5.0 * counts[i])) for i in range(5)}
        print("Train class counts:", counts.astype(int).tolist())
        print("Using class_weight:", class_weight)

        steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))
        validation_steps = int(np.ceil(n_val / BATCH_SIZE))

    else:
        train_paths = np.array(
            [
                os.path.join(TRAIN_DIR, fn)
                for fn in df.loc[idx_train, "image_id"].values
            ],
            dtype=str,
        )
        val_paths = np.array(
            [os.path.join(TRAIN_DIR, fn) for fn in df.loc[idx_val, "image_id"].values],
            dtype=str,
        )
        y_train = y_all[idx_train]
        y_val = y_all[idx_val]

        @tf.function
        def _decode_resize_norm(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(
                img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
            )
            img = tf.cast(img, tf.float32) / 255.0
            return img

        @tf.function
        def _augment(img, label):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.1, seed=SEED)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        @tf.function
        def _to_xy(img, label):
            return img, label

        BATCH_SIZE = 32

        options = tf.data.Options()
        options.experimental_deterministic = True

        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_paths, y_train)
        ).with_options(options)

        shuffle_buf = min(len(train_paths), 8192)
        train_ds = train_ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        train_ds = train_ds.map(
            lambda p, y: (_decode_resize_norm(p), y), num_parallel_calls=AUTOTUNE
        ).apply(tf.data.experimental.ignore_errors())

        train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, y_val)).with_options(
            options
        )
        val_ds = val_ds.map(
            lambda p, y: (_decode_resize_norm(p), y), num_parallel_calls=AUTOTUNE
        ).apply(tf.data.experimental.ignore_errors())
        val_ds = val_ds.map(_to_xy, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        counts = np.bincount(y_train, minlength=5).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weight = {i: (len(y_train) / (5.0 * counts[i])) for i in range(5)}
        print("Train class counts:", counts.astype(int).tolist())
        print("Using class_weight:", class_weight)

        steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
        validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

    new_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    new_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=2,
        class_weight=class_weight,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )




## === cell 4
new_model.summary()




## === cell 5
if "IMG_SIZE" not in globals():
    IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)
print("Inference IMG_SIZE:", IMG_SIZE)




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub["image_id"].tolist()

missing = [fn for fn in test_images if not os.path.exists(os.path.join(TEST_DIR, fn))]
print("Missing test files:", len(missing))
assert (
    len(missing) == 0
), "Some test images referenced in sample_submission are missing on disk."




## === cell 7
INFER_BATCH_SIZE = 64

options = tf.data.Options()
options.experimental_deterministic = True

use_test_tfrecords = os.path.isdir(TEST_TFRECORD_DIR) and any(
    f.endswith(".tfrec") for f in os.listdir(TEST_TFRECORD_DIR)
)

if use_test_tfrecords:
    tfrec_test_files = sorted(
        tf.io.gfile.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec"))
    )

    def _test_feature_description():
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }

    @tf.function
    def _parse_test_tfrecord(example_proto):
        ex = tf.io.parse_single_example(example_proto, _test_feature_description())
        img = tf.io.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32) / 255.0
        return ex["image_name"], img

    test_ds = tf.data.TFRecordDataset(
        tfrec_test_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    test_ds = test_ds.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE).apply(
        tf.data.experimental.ignore_errors()
    )

    names_ds = test_ds.map(lambda n, x: n, num_parallel_calls=AUTOTUNE)
    imgs_ds = test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE)

    names_ds = names_ds.batch(INFER_BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    imgs_ds = imgs_ds.batch(INFER_BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    all_names = np.concatenate([b.numpy() for b in names_ds], axis=0).astype("S")
    all_probs = new_model.predict(imgs_ds, verbose=0)

    name_to_index = {n.decode("utf-8"): i for i, n in enumerate(all_names)}
    order_idx = np.fromiter(
        (name_to_index[n] for n in test_images), dtype=np.int64, count=len(test_images)
    )
    probs = all_probs[order_idx]
    preds = probs.argmax(axis=1).astype(int).tolist()
else:
    test_paths = np.array([os.path.join(TEST_DIR, fn) for fn in test_images], dtype=str)

    @tf.function
    def _decode_resize_norm_infer(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
    test_ds = test_ds.map(_decode_resize_norm_infer, num_parallel_calls=AUTOTUNE).apply(
        tf.data.experimental.ignore_errors()
    )
    test_ds = test_ds.batch(INFER_BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    probs = new_model.predict(test_ds, verbose=0)
    preds = probs.argmax(axis=1).astype(int).tolist()

print("Num preds:", len(preds), "Num test_images:", len(test_images))




## === cell 8
preds[:20], pd.Series(preds).value_counts().sort_index()




## === cell 9
assert len(test_images) == len(preds), "Prediction count mismatch with test images."

sub = pd.DataFrame({"image_id": test_images, "label": preds})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

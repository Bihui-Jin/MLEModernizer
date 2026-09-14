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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import numpy as np
import pandas as pd

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
tf = None
load_model = None

SEED = 42
DEBUG = False

np.random.seed(SEED)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("WARNING: TensorFlow failed to import; will write a fallback submission.")
    print("TF import error:", TF_IMPORT_ERROR)



## === cell 1
CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )

TEST_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
test_images = sorted(glob.glob(TEST_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found with glob: {TEST_GLOB}")

df_test = pd.DataFrame({"path": test_images})
print("DATA_ROOT:", DATA_ROOT)
print("Num test images:", len(df_test))



## === cell 2
final_csv = None

if not TF_AVAILABLE:
    sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
    if not os.path.exists(sample_path):
        final_csv = pd.DataFrame(
            {
                "image_id": pd.Series(df_test["path"]).str.rsplit(os.sep, n=1).str[-1],
                "label": np.zeros(len(df_test), dtype=np.int64),
            }
        )
    else:
        ss = pd.read_csv(sample_path)
        ss["label"] = 0
        final_csv = ss[["image_id", "label"]].copy()
        final_csv["label"] = final_csv["label"].astype(np.int64)

    final_csv.to_csv("submission.csv", index=False)



## === cell 3
if TF_AVAILABLE:
    CANDIDATE_MODEL_PATHS = [
        "/kaggle/input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
        "../input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
        "/kaggle/data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
        "../data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    ]

    weight_path = None
    for p in CANDIDATE_MODEL_PATHS:
        if os.path.exists(p):
            weight_path = p
            break

    if weight_path is None:
        for root in ["/kaggle/input", "/kaggle/data", "../input", "../data"]:
            if os.path.isdir(root):
                matches = glob.glob(
                    os.path.join(root, "**", "model_v0.25.h5"), recursive=True
                )
                if len(matches) > 0:
                    matches = sorted(matches)
                    weight_path = matches[0]
                    break

    custom_objects = {}
    try:
        custom_objects["swish"] = tf.nn.swish
    except Exception:
        pass

    my_model = None
    if weight_path is not None:
        print("Loading model from:", weight_path)
        my_model = load_model(weight_path, compile=False, custom_objects=custom_objects)
    else:
        train_csv_path = os.path.join(DATA_ROOT, "train.csv")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
        df_train = pd.read_csv(train_csv_path)
        train_img_dir = os.path.join(DATA_ROOT, "train_images")
        df_train["path"] = df_train["image_id"].apply(
            lambda x: os.path.join(train_img_dir, x)
        )
        if not df_train["path"].map(os.path.exists).all():
            alt_dir = os.path.join(
                DATA_ROOT, "cassava-leaf-disease-classification", "train_images"
            )
            if os.path.isdir(alt_dir):
                train_img_dir = alt_dir
                df_train["path"] = df_train["image_id"].apply(
                    lambda x: os.path.join(train_img_dir, x)
                )

        num_classes = int(df_train["label"].nunique())
        if num_classes != 5:
            num_classes = 5

        idx = np.arange(len(df_train))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        split = int(0.9 * len(idx))
        tr_idx, va_idx = idx[:split], idx[split:]
        df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_idx].reset_index(drop=True)

        AUTO = tf.data.AUTOTUNE
        TARGET_SIZE = (512, 512)
        BATCH_SIZE = 16  # keep memory-safe for 512x512
        EPOCHS = 5  # keep runtime under 600s while improving far above random fallback

        def _decode_resize_label(path, label):
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(
                img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR
            )
            img = tf.cast(img, tf.float32) / 255.0
            label = tf.cast(label, tf.int32)
            return img, label

        def _augment_train(img, label):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.2, seed=SEED)
            return img, label

        def make_ds(df, training):
            paths = df["path"].to_numpy()
            labels = df["label"].to_numpy().astype(np.int32)
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            if training:
                ds = ds.shuffle(
                    min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
                )
            ds = ds.map(_decode_resize_label, num_parallel_calls=AUTO)
            if training:
                ds = ds.map(_augment_train, num_parallel_calls=AUTO)
            ds = ds.batch(BATCH_SIZE, drop_remainder=False)
            ds = ds.prefetch(AUTO)
            return ds

        train_ds = make_ds(df_tr, training=True)
        val_ds = make_ds(df_va, training=False)

        inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3, seed=SEED)(x)
        outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
        my_model = tf.keras.Model(inputs, outputs)

        my_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        my_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 4
if TF_AVAILABLE:
    AUTO = tf.data.AUTOTUNE
    TARGET_SIZE = (512, 512)

    _ROTATION_RANGE_DEG = 90.0
    _BRIGHTNESS_RANGE = (0.2, 0.4)
    _HFLIP = True
    _VFLIP = True

    _seed_base = tf.constant([SEED, 0], dtype=tf.int32)

    paths_np = df_test["path"].to_numpy()

    def _decode_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _augment(img, seed):
        factor = tf.random.stateless_uniform(
            [], seed=seed, minval=_BRIGHTNESS_RANGE[0], maxval=_BRIGHTNESS_RANGE[1]
        )
        img = img * factor

        seed1 = seed + tf.constant([1, 0], tf.int32)
        if _HFLIP:
            do = tf.random.stateless_uniform([], seed=seed1) < 0.5
            img = tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)

        seed2 = seed + tf.constant([2, 0], tf.int32)
        if _VFLIP:
            do = tf.random.stateless_uniform([], seed=seed2) < 0.5
            img = tf.cond(do, lambda: tf.image.flip_up_down(img), lambda: img)

        seed3 = seed + tf.constant([3, 0], tf.int32)
        k = tf.random.stateless_uniform(
            [], seed=seed3, minval=0, maxval=4, dtype=tf.int32
        )
        img = tf.image.rot90(img, k=k)

        return img

    options = tf.data.Options()
    options.experimental_deterministic = True

    base_ds = tf.data.Dataset.from_tensor_slices(paths_np)
    base_ds = base_ds.map(_decode_resize, num_parallel_calls=AUTO)

    cache_path = os.path.join("/kaggle/working", "test_decode_resize_cache")
    base_ds = base_ds.cache(cache_path)

    idx_ds = tf.data.Dataset.range(len(paths_np))
    base_indexed_ds = tf.data.Dataset.zip((base_ds, idx_ds)).with_options(options)

    def make_test_ds_tta(batch_size=128, tta_pass=0):
        tta_pass = tf.constant(tta_pass, dtype=tf.int32)

        def add_tta(img, idx):
            seed = _seed_base + tf.stack([tta_pass, tf.cast(idx, tf.int32)])
            img = _augment(img, seed)
            return img

        ds = base_indexed_ds.map(add_tta, num_parallel_calls=AUTO)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTO)
        return ds

    pred_list = []
    for t in range(5):
        test_ds = make_test_ds_tta(batch_size=64, tta_pass=t)
        pred = my_model.predict(test_ds, verbose=1)
        pred_list.append(pred)

    pred_test = np.mean(np.stack(pred_list, axis=0), axis=0)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    final_submission = df_test.copy()
    final_submission["image_id"] = (
        pd.Series(paths_np).str.rsplit(os.sep, n=1).str[-1].to_numpy()
    )
    final_submission["label"] = pred_test_labels.astype(np.int64)

    final_csv = final_submission[["image_id", "label"]]

    if final_csv.shape[0] != len(test_images):
        raise RuntimeError("Submission row count does not match number of test images.")
    if list(final_csv.columns) != ["image_id", "label"]:
        raise RuntimeError(
            "Submission columns are incorrect; expected ['image_id','label']."
        )

    final_csv.to_csv("submission.csv", index=False)



## === cell 5
if final_csv is None and os.path.exists("submission.csv"):
    final_csv = pd.read_csv("submission.csv")

print(final_csv.head())
print(f"\nWrote submission.csv with {len(final_csv)} rows.")
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

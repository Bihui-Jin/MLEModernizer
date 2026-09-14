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

3.13

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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 2))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(1337)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass



## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = f"{BASE}/test_images"
train_image_dir = f"{BASE}/train_images"
train_csv_path = f"{BASE}/train.csv"
sample_path = f"{BASE}/sample_submission.csv"

assert os.path.exists(test_image_dir), f"Missing test image dir: {test_image_dir}"
assert os.path.exists(sample_path), f"Missing sample submission: {sample_path}"
assert os.path.exists(train_image_dir), f"Missing train image dir: {train_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"

sample_csv = pd.read_csv(sample_path)
train_csv = pd.read_csv(train_csv_path)
print("sample_submission shape:", sample_csv.shape)
print("train_csv shape:", train_csv.shape)
sample_csv.head()



## === cell 2
source_dir = "/kaggle/input/cp-model"
dest_dir = "/kaggle/working/cp-model"
if os.path.exists(source_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    os.environ["TFHUB_CACHE_DIR"] = dest_dir
    print("TFHUB_CACHE_DIR set to:", dest_dir)
else:
    print(
        "Optional TFHub cache dataset not found at",
        source_dir,
        "- continuing without TFHub.",
    )



## === cell 3
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"
model_path_7 = "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5"

classifier = None  # placeholder to keep variable names consistent
model_path_8 = classifier

paths = [
    model_path_1,
    model_path_2,
    model_path_3,
    model_path_4,
    model_path_5,
    model_path_6,
    model_path_7,
]
exists = {p: os.path.exists(p) for p in paths}
print("Model files found:", sum(exists.values()), "of", len(paths))
for p, ok in exists.items():
    if ok:
        print("  OK:", p)
    else:
        print("  MISSING:", p)



## === cell 4
models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (550, 550)),
    (model_path_3, (299, 299)),  # common for InceptionV3
    (model_path_4, (550, 550)),
    (model_path_5, (550, 550)),
    (model_path_6, (550, 550)),
    (model_path_7, (512, 512)),
]

models = []
for path, input_size in models_info:
    if not os.path.exists(path):
        continue
    try:
        m = load_model(path, compile=False)
        models.append((m, input_size))
        print("Loaded model:", os.path.basename(path), "input_size:", input_size)
    except Exception as e:
        print("Failed to load model:", path, "error:", repr(e))

if classifier is not None:
    models.append((classifier, (224, 224)))

print("Total loaded models:", len(models))

NEED_FALLBACK = len(models) == 0
print("Need fallback model:", NEED_FALLBACK)



## === cell 5
fallback_model = None
fallback_input_size = (224, 224)

if NEED_FALLBACK:
    seed = 1337
    tf.keras.utils.set_random_seed(seed)

    img_size = fallback_input_size
    batch_size = 32
    num_classes = 5

    train_csv["filepath"] = train_csv["image_id"].apply(
        lambda x: os.path.join(train_image_dir, x)
    )
    train_csv = train_csv[train_csv["filepath"].apply(os.path.exists)].reset_index(
        drop=True
    )

    val_frac = 0.10
    parts = []
    val_parts = []
    for lbl, grp in train_csv.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(grp) * val_frac))
        val_parts.append(grp.iloc[:n_val])
        parts.append(grp.iloc[n_val:])
    tr_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    va_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    print("Fallback train/val sizes:", len(tr_df), len(va_df))
    print("Train label dist:", tr_df["label"].value_counts().sort_index().to_dict())
    print("Val label dist:", va_df["label"].value_counts().sort_index().to_dict())

    def decode_image(path, label=None):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        if label is None:
            return img
        return img, tf.one_hot(tf.cast(label, tf.int32), num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tr_df["filepath"].values, tr_df["label"].values)
    )
    train_ds = train_ds.shuffle(
        min(len(tr_df), 8192), seed=seed, reshuffle_each_iteration=True
    )
    train_ds = (
        train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (va_df["filepath"].values, va_df["label"].values)
    )
    val_ds = (
        val_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 5
    history = fallback_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, verbose=2
    )

    models = [(fallback_model, fallback_input_size)]
    print("Fallback model trained and attached. Total models:", len(models))



## === cell 6
image_predictions = []

image_ids = sample_csv["image_id"].tolist()

try:
    test_files_set = set(os.listdir(test_image_dir))
except Exception:
    test_files_set = None

if test_files_set is None:
    present_mask = np.fromiter(
        (os.path.exists(os.path.join(test_image_dir, img_id)) for img_id in image_ids),
        dtype=bool,
        count=len(image_ids),
    )
else:
    present_mask = np.fromiter(
        (img_id in test_files_set for img_id in image_ids),
        dtype=bool,
        count=len(image_ids),
    )

present_ids = [img_id for img_id, ok in zip(image_ids, present_mask) if ok]
present_paths = [os.path.join(test_image_dir, img_id) for img_id in present_ids]

missing_count = int((~present_mask).sum())
if missing_count:
    first_missing = next(
        (img_id for img_id, ok in zip(image_ids, present_mask) if not ok), None
    )
    print(
        "Warning: missing",
        missing_count,
        "test images referenced in sample_submission (first):",
        first_missing,
    )

if len(present_paths) == 0 or len(models) == 0:
    for image_id in image_ids:
        image_predictions.append({"image_id": image_id, "label": 0})
else:
    AUTOTUNE = tf.data.AUTOTUNE
    num_classes = 5
    bs_pred = (
        32  # keep stable predict batch size; semantics unchanged (order preserved)
    )

    @tf.function(reduce_retracing=True)
    def _decode_to_uint8(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.cast(img, tf.uint8)

    base_ds_uint8 = tf.data.Dataset.from_tensor_slices(present_paths)
    base_ds_uint8 = base_ds_uint8.map(
        _decode_to_uint8, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    base_ds_uint8 = base_ds_uint8.cache()
    base_ds_uint8 = base_ds_uint8.prefetch(AUTOTUNE)

    unique_sizes = sorted({tuple(sz) for _, sz in models})
    ds_by_size = {}

    for h, w in unique_sizes:

        @tf.function(reduce_retracing=True)
        def _prep_uint8_to_float(x_uint8, hh=h, ww=w):
            x = tf.cast(x_uint8, tf.float32) / 255.0
            x = tf.image.resize(x, (hh, ww), method="bilinear")
            return x

        ds_for_size = (
            base_ds_uint8.map(
                _prep_uint8_to_float, num_parallel_calls=AUTOTUNE, deterministic=True
            )
            .batch(bs_pred, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        ds_by_size[(h, w)] = ds_for_size

    all_model_probs = []
    for model, input_size in models:
        h, w = input_size
        ds_for_model = ds_by_size[(h, w)]
        probs = model.predict(ds_for_model, verbose=0)
        all_model_probs.append(probs.astype(np.float32, copy=False))

    probs_stack = np.stack(all_model_probs, axis=0)  # (M, N, C)
    M, N, C = probs_stack.shape

    pred_cls = np.argmax(probs_stack, axis=2).astype(np.int64)  # (M, N)
    pred_conf = np.take_along_axis(probs_stack, pred_cls[..., None], axis=2)[
        ..., 0
    ]  # (M, N)

    flat_idx = (np.arange(N, dtype=np.int64)[None, :] * C + pred_cls).reshape(-1)
    counts_flat = np.bincount(flat_idx, minlength=N * C).astype(np.int16)
    conf_flat = np.bincount(
        flat_idx, weights=pred_conf.reshape(-1), minlength=N * C
    ).astype(np.float32)

    counts = counts_flat.reshape(N, C)
    conf_sums = conf_flat.reshape(N, C)

    max_counts = counts.max(axis=1, keepdims=True)
    tied = counts == max_counts
    num_tied = tied.sum(axis=1)

    first_tied = tied.argmax(axis=1).astype(np.int64)

    denom = np.maximum(counts.astype(np.float32), 1.0)
    mean_conf = conf_sums / denom
    mean_conf_masked = np.where(tied, mean_conf, -np.inf)
    best_mean = mean_conf_masked.argmax(axis=1).astype(np.int64)

    final_preds_present = np.where(num_tied > 1, best_mean, first_tied).astype(np.int64)

    j = 0
    for image_id, ok in zip(image_ids, present_mask):
        if not ok:
            image_predictions.append({"image_id": image_id, "label": 0})
        else:
            image_predictions.append(
                {"image_id": image_id, "label": int(final_preds_present[j])}
            )
            j += 1

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]

assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(submission_df.columns) == ["image_id", "label"], "Wrong submission columns"
assert (
    submission_df["label"].between(0, 4).all()
), "Labels must be in [0,4] for this competition."

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission_df.shape)



## === cell 7
submission_df.head()



## === cell 8
print("Label value counts:")
print(submission_df["label"].value_counts().sort_index())
print(
    "Min label:",
    submission_df["label"].min(),
    "Max label:",
    submission_df["label"].max(),
)
submission_df.tail()

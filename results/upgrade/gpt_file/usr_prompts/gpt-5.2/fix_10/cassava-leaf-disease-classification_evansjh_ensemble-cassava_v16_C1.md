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

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]
os.environ.setdefault("PYTHONHASHSEED", "0")

import sys
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

np.random.seed(0)
tf.random.set_seed(0)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(BASE_INPUT, "test_images")
train_image_dir = os.path.join(BASE_INPUT, "train_images")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
sample = os.path.join(BASE_INPUT, "sample_submission.csv")
out_path = "/kaggle/working/submission.csv"

if not os.path.exists(sample):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        sample = alt
    else:
        raise FileNotFoundError(f"sample_submission.csv not found at {sample} or {alt}")

if not os.path.exists(test_image_dir):
    alt = "/kaggle/input/test_images"
    if os.path.exists(alt):
        test_image_dir = alt
    else:
        raise FileNotFoundError(
            f"test_images directory not found at {test_image_dir} or {alt}"
        )

if not os.path.exists(train_image_dir):
    alt = "/kaggle/input/train_images"
    if os.path.exists(alt):
        train_image_dir = alt

if not os.path.exists(train_csv_path):
    alt = "/kaggle/input/train.csv"
    if os.path.exists(alt):
        train_csv_path = alt

print("Using:")
print(" sample:", sample)
print(" test_image_dir:", test_image_dir)
print(" train_image_dir:", train_image_dir, "exists:", os.path.exists(train_image_dir))
print(" train_csv:", train_csv_path, "exists:", os.path.exists(train_csv_path))
print(" tf version:", tf.__version__)
print(" python:", sys.version.split()[0])

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
USE_TFHUB = False
classifier = None




## === cell 2
sample_csv = pd.read_csv(sample)
assert list(sample_csv.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv format"
print("Sample rows:", len(sample_csv))
sample_csv.head()




## === cell 3
def find_h5_models(root="/kaggle/input", max_models=8):
    search_dirs = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/models",
        "/kaggle/input/cassava-leaf-disease-classification/weights",
        "/kaggle/input/models",
        "/kaggle/input/weights",
        root,
    ]

    patterns = []
    for d in search_dirs:
        if d and os.path.isdir(d):
            patterns.append(os.path.join(d, "*.h5"))
            patterns.append(os.path.join(d, "*.hdf5"))
            patterns.append(os.path.join(d, "**", "*.h5"))
            patterns.append(os.path.join(d, "**", "*.hdf5"))

    h5_paths = []
    seen = set()
    for pat in patterns:
        try:
            for p in tf.io.gfile.glob(pat):
                if p not in seen:
                    h5_paths.append(p)
                    seen.add(p)
        except Exception:
            pass
        if len(h5_paths) >= max_models:
            break

    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        if "bestmodel" in name or "best_model" in name:
            s += 3
        if "inception" in name or "goog" in name:
            s += 1
        return (-s, len(p))

    h5_paths = sorted(h5_paths, key=score)
    return h5_paths[:max_models], h5_paths


selected_h5, all_h5 = find_h5_models()
print(f"Found {len(all_h5)} .h5/.hdf5 files under searched roots")
print("Selected for loading:")
for p in selected_h5:
    print(" -", p)




## === cell 4
def get_model_input_size(model):
    ish = model.input_shape
    if (
        isinstance(ish, (list, tuple))
        and len(ish) > 0
        and isinstance(ish[0], (list, tuple))
    ):
        ish = ish[0]
    if ish is None or len(ish) < 4:
        return (224, 224)
    h, w = ish[1], ish[2]
    if h is None or w is None:
        return (224, 224)
    return (int(h), int(w))


models = []
for path in selected_h5:
    try:
        m = load_model(path, compile=False)
        input_size = get_model_input_size(m)
        models.append((m, input_size))
        print(f"Loaded: {os.path.basename(path)}  input_size={input_size}")
    except Exception as e:
        print(f"Skipping model (failed to load) {path}: {type(e).__name__}: {e}")




## === cell 5
NUM_CLASSES = 5


def build_fallback_model(input_size=(224, 224), num_classes=5):
    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _decode_and_resize(path, label, input_size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, input_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img, label


def make_train_ds(
    df, image_dir, input_size=(224, 224), batch_size=32, shuffle=True, cache_path=None
):
    paths = (image_dir.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=0, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, y: _decode_and_resize(p, y, input_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    if cache_path is not None:
        ds = ds.cache(cache_path)

    def aug(img, y):
        img = tf.image.random_flip_left_right(img, seed=0)
        img = tf.image.random_flip_up_down(img, seed=0)
        return img, y

    ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if len(models) == 0:
    if not (os.path.exists(train_csv_path) and os.path.exists(train_image_dir)):
        raise RuntimeError(
            "No .h5 models found under /kaggle/input AND training data not available; cannot proceed."
        )

    train_df = pd.read_csv(train_csv_path)
    assert set(train_df.columns) >= {"image_id", "label"}
    idx = np.arange(len(train_df))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    fallback_input_size = (224, 224)
    fallback_model = build_fallback_model(fallback_input_size, NUM_CLASSES)

    batch_size = 32

    cache_dir = "/kaggle/working/tfds_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(
        cache_dir, f"train_{fallback_input_size[0]}x{fallback_input_size[1]}.cache"
    )
    val_cache = os.path.join(
        cache_dir, f"val_{fallback_input_size[0]}x{fallback_input_size[1]}.cache"
    )

    train_ds = make_train_ds(
        tr_df,
        train_image_dir,
        fallback_input_size,
        batch_size=batch_size,
        shuffle=True,
        cache_path=train_cache,
    )
    val_ds = make_train_ds(
        va_df,
        train_image_dir,
        fallback_input_size,
        batch_size=batch_size,
        shuffle=False,
        cache_path=val_cache,
    )

    epochs = 3
    print(
        f"Training fallback CNN for {epochs} epochs on {len(tr_df)} images; val {len(va_df)} images..."
    )
    fallback_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    models = [(fallback_model, fallback_input_size)]
    print("Fallback model trained and will be used for prediction.")




## === cell 6
def _make_test_ds_for_input_size(image_paths, input_size, batch_size):
    image_paths = tf.convert_to_tensor(image_paths)  # cheaper + avoids Python overhead

    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, input_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.cache()  # IMPORTANT: assign; caches the mapped tensors for reuse
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


image_ids = sample_csv["image_id"].tolist()
image_paths = np.array(
    [os.path.join(test_image_dir, iid) for iid in image_ids], dtype=object
)

try:
    test_files = set(os.listdir(test_image_dir))
    exists_mask = np.fromiter(
        (iid in test_files for iid in image_ids), dtype=bool, count=len(image_ids)
    )
except Exception:
    exists_mask = np.fromiter(
        (os.path.exists(p) for p in image_paths), dtype=bool, count=len(image_paths)
    )

missing_images = int((~exists_mask).sum())
valid_indices = np.flatnonzero(exists_mask)
valid_paths = image_paths[valid_indices]

if len(valid_paths) == 0:
    image_predictions = [{"image_id": iid, "label": 0} for iid in image_ids]
    failed_preds = 0
else:
    size_to_models = {}
    for mi, (m, input_size) in enumerate(models):
        size_to_models.setdefault(tuple(input_size), []).append((mi, m))

    n_models = len(models)
    n_valid = len(valid_paths)
    model_pred_class = np.empty((n_models, n_valid), dtype=np.int16)
    model_pred_conf = np.empty((n_models, n_valid), dtype=np.float32)

    BATCH_SIZE = 128  # unchanged

    ds_cache = {}
    for input_size_t, model_list in size_to_models.items():
        if input_size_t not in ds_cache:
            ds_cache[input_size_t] = _make_test_ds_for_input_size(
                valid_paths, input_size_t, batch_size=BATCH_SIZE
            )
        ds = ds_cache[input_size_t]

        for mi, model in model_list:
            y = model.predict(ds, verbose=0)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if y.shape[1] != NUM_CLASSES:
                if y.shape[1] < NUM_CLASSES:
                    pad = np.zeros(
                        (y.shape[0], NUM_CLASSES - y.shape[1]), dtype=y.dtype
                    )
                    y = np.concatenate([y, pad], axis=1)
                else:
                    y = y[:, :NUM_CLASSES]

            pred = np.argmax(y, axis=1).astype(np.int16)
            conf = y[np.arange(y.shape[0]), pred].astype(np.float32)

            model_pred_class[mi, :] = pred
            model_pred_conf[mi, :] = conf

    image_predictions = [{"image_id": iid, "label": 0} for iid in image_ids]
    failed_preds = 0

    preds_int = model_pred_class.astype(np.int32)  # (M, N)
    confs_f = model_pred_conf.astype(np.float32)  # (M, N)

    np.clip(preds_int, 0, NUM_CLASSES - 1, out=preds_int)

    counts = np.zeros((n_valid, NUM_CLASSES), dtype=np.int16)
    for c in range(NUM_CLASSES):
        counts[:, c] = (preds_int == c).sum(axis=0).astype(np.int16)

    max_counts = counts.max(axis=1)
    candidates = counts == max_counts[:, None]
    winners = counts.argmax(axis=1).astype(np.int32)

    tie_mask = candidates.sum(axis=1) > 1
    tie_idx = np.flatnonzero(tie_mask)

    if tie_idx.size:
        T = tie_idx.size
        sum_conf = np.zeros((T, NUM_CLASSES), dtype=np.float32)
        cnt_conf = np.zeros((T, NUM_CLASSES), dtype=np.int16)

        pm = preds_int[:, tie_idx]  # (M, T)
        cm = confs_f[:, tie_idx]  # (M, T)

        rows = np.broadcast_to(np.arange(T), pm.shape).ravel()
        cols = pm.ravel()
        vals = cm.ravel()

        np.add.at(sum_conf, (rows, cols), vals)
        np.add.at(cnt_conf, (rows, cols), 1)

        avg_conf = sum_conf / np.maximum(cnt_conf, 1)
        cand_t = candidates[tie_idx].astype(bool)
        masked_avg = np.where(cand_t, avg_conf, -np.inf)
        winners[tie_idx] = masked_avg.argmax(axis=1).astype(np.int32)

    for j, idx in enumerate(valid_indices):
        image_predictions[idx] = {"image_id": image_ids[idx], "label": int(winners[j])}

print("Missing images:", missing_images)
print("Failed predictions:", failed_preds)
print("Pred rows:", len(image_predictions))




## === cell 7
pred_df = pd.DataFrame(image_predictions)

if "image_id" not in pred_df.columns:
    pred_df["image_id"] = sample_csv["image_id"].values
if "label" not in pred_df.columns:
    pred_df["label"] = 0

pred_df["label"] = (
    pd.to_numeric(pred_df["label"], errors="coerce").fillna(0).astype(int)
)
pred_df = sample_csv[["image_id"]].merge(
    pred_df[["image_id", "label"]], on="image_id", how="left"
)
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

assert list(pred_df.columns) == ["image_id", "label"]
assert len(pred_df) == len(sample_csv), "Submission row count mismatch"

pred_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
pred_df.head()




## === cell 8
print(pred_df.shape)
print(pred_df.dtypes)
print(pred_df["label"].value_counts().sort_index())
print("Unique image_ids:", pred_df["image_id"].nunique())
print("Submission path exists:", os.path.exists(out_path))
print("Submission suffix ok:", out_path.lower().endswith(".csv"))

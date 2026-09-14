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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PYTHONHASHSEED", "0")

import sys
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

np.random.seed(0)
tf.random.set_seed(0)

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
    search_roots = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input",
    ]

    h5_paths = []
    seen = set()
    for sr in search_roots:
        if not os.path.exists(sr):
            continue
        for dirpath, dirnames, filenames in os.walk(sr):
            for fn in filenames:
                if fn.lower().endswith((".h5", ".hdf5")):
                    p = os.path.join(dirpath, fn)
                    if p not in seen:
                        h5_paths.append(p)
                        seen.add(p)
        if h5_paths:
            break  # found models in the likely location; no need to scan everything

    if not h5_paths and root not in search_roots and os.path.exists(root):
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".h5", ".hdf5")):
                    h5_paths.append(os.path.join(dirpath, fn))

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


def make_train_ds(df, image_dir, input_size=(224, 224), batch_size=32, shuffle=True):
    paths = (image_dir.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=0, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, y: _decode_and_resize(p, y, input_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def aug(img, y):
        img = tf.image.random_flip_left_right(img, seed=0)
        img = tf.image.random_flip_up_down(img, seed=0)
        return img, y

    ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
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
    train_ds = make_train_ds(
        tr_df, train_image_dir, fallback_input_size, batch_size=batch_size, shuffle=True
    )
    val_ds = make_train_ds(
        va_df,
        train_image_dir,
        fallback_input_size,
        batch_size=batch_size,
        shuffle=False,
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
    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, input_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


image_ids = sample_csv["image_id"].tolist()
image_paths = [os.path.join(test_image_dir, iid) for iid in image_ids]
exists_mask = np.fromiter(
    (os.path.exists(p) for p in image_paths), dtype=bool, count=len(image_paths)
)
missing_images = int((~exists_mask).sum())
valid_indices = np.flatnonzero(exists_mask)
valid_paths = [image_paths[i] for i in valid_indices]

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

    BATCH_SIZE = 64

    for input_size_t, model_list in size_to_models.items():
        ds = _make_test_ds_for_input_size(
            valid_paths, input_size_t, batch_size=BATCH_SIZE
        )

        for mi, model in model_list:
            y = model.predict(ds, verbose=0)
            y = np.asarray(y)
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

    for j, idx in enumerate(valid_indices):
        preds = model_pred_class[:, j].astype(int).tolist()
        confs = model_pred_conf[:, j].astype(float).tolist()

        for k in range(len(preds)):
            pc = preds[k]
            if pc < 0 or pc >= NUM_CLASSES:
                preds[k] = int(np.clip(pc, 0, NUM_CLASSES - 1))

        class_votes = Counter(preds)
        most_common = class_votes.most_common()
        if len(most_common) == 0:
            failed_preds += 1
            image_predictions[idx] = {"image_id": image_ids[idx], "label": 0}
            continue

        final_predicted_class = most_common[0][0]

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            top_count = most_common[0][1]
            tied_classes = [cls for cls, cnt in most_common if cnt == top_count]

            confidence_scores = {}
            for pc, cf in zip(preds, confs):
                confidence_scores.setdefault(pc, []).append(cf)

            final_predicted_class = max(
                tied_classes,
                key=lambda cls: (
                    sum(confidence_scores.get(cls, [0.0]))
                    / max(1, len(confidence_scores.get(cls, [])))
                ),
            )

        image_predictions[idx] = {
            "image_id": image_ids[idx],
            "label": int(final_predicted_class),
        }

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

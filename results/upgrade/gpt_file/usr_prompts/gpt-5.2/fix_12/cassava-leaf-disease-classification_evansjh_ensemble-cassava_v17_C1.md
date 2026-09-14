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
os.environ["PYTHONHASHSEED"] = "0"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

tf.random.set_seed(0)
np.random.seed(0)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 4
    _threads = max(1, min(8, _cpu))
    tf.config.threading.set_intra_op_parallelism_threads(_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_DIR, "test_images")
train_image_dir = os.path.join(DATA_DIR, "train_images")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_csv_path = os.path.join(DATA_DIR, "train.csv")

assert os.path.isdir(test_image_dir), f"Missing test images directory: {test_image_dir}"
assert os.path.isfile(sample_path), f"Missing sample submission: {sample_path}"
assert os.path.isdir(
    train_image_dir
), f"Missing train images directory: {train_image_dir}"
assert os.path.isfile(train_csv_path), f"Missing train.csv: {train_csv_path}"

sample_csv = pd.read_csv(sample_path)
if not {"image_id", "label"}.issubset(sample_csv.columns):
    raise ValueError(
        f"sample_submission.csv must contain image_id,label. Got: {list(sample_csv.columns)}"
    )

train_df = pd.read_csv(train_csv_path)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv must contain image_id,label. Got: {list(train_df.columns)}"
    )

print("Train rows:", len(train_df), "Test rows:", len(sample_csv))



## === cell 2
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

models_info_all = [
    (model_path_1, (550, 550)),
    (model_path_6, (512, 512)),
]

existing_models_info = [(p, s) for (p, s) in models_info_all if os.path.isfile(p)]

if len(existing_models_info) == 0:
    print(
        "No external .h5 models found under expected paths. Will train a fallback model from train_images."
    )
else:
    print("Models to load:")
    for p, s in existing_models_info:
        print(f" - {p} @ {s}")



## === cell 3
models = []
if len(existing_models_info) > 0:
    for path, input_size in existing_models_info:
        try:
            m = load_model(path, compile=False)
            models.append((m, input_size))
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {path}: {e}")

print(f"Loaded {len(models)} external model(s).")

fallback_model = None
fallback_input_size = (224, 224)
NUM_CLASSES = 5

if len(models) == 0:
    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = True

    train_df_shuf = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    val_frac = 0.1
    val_n = int(len(train_df_shuf) * val_frac)
    val_df = train_df_shuf.iloc[:val_n].copy()
    trn_df = train_df_shuf.iloc[val_n:].copy()

    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, fallback_input_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return img, label

    def make_ds(df, training):
        paths = (train_image_dir + "/" + df["image_id"].astype(str)).to_numpy()
        labels = df["label"].to_numpy(np.int32, copy=False)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.with_options(options)
        if training:
            ds = ds.shuffle(4096, seed=0, reshuffle_each_iteration=True)
        ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(32, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(trn_df, training=True)
    val_ds = make_ds(val_df, training=False)

    inputs = tf.keras.Input(shape=(fallback_input_size[0], fallback_input_size[1], 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2, seed=0)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fallback_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)



## === cell 4
test_files = sample_csv["image_id"].astype(str).tolist()
test_paths = [os.path.join(test_image_dir, fn) for fn in test_files]

final_labels = np.zeros(len(test_files), dtype=np.int64)

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.autotune = True

BATCH_SIZE = 64

try:
    _present_names = set(os.listdir(test_image_dir))
except Exception:
    _present_names = None

if _present_names is not None:
    exists_mask = np.fromiter(
        (fn in _present_names for fn in test_files), dtype=bool, count=len(test_files)
    )
else:
    exists_mask = np.fromiter(
        (os.path.isfile(p) for p in test_paths), dtype=bool, count=len(test_paths)
    )

present_idx = np.flatnonzero(exists_mask)
missing_idx = np.flatnonzero(~exists_mask)
if missing_idx.size:
    print(
        f"Warning: {missing_idx.size} test images listed in sample_submission not found on disk. "
        f"Will predict label=0 for those."
    )

present_paths = np.asarray([test_paths[i] for i in present_idx], dtype=np.str_)

if len(models) > 0:
    @tf.function(reduce_retracing=True)
    def _decode_norm(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # == /255.0
        return img

    decoded_ds = (
        tf.data.Dataset.from_tensor_slices(present_paths)
        .with_options(options)
        .map(_decode_norm, num_parallel_calls=AUTOTUNE, deterministic=True)
        .cache()
        .prefetch(AUTOTUNE)
    )

    unique_sizes = {}
    for _, input_size in models:
        unique_sizes[tuple(map(int, input_size))] = None

    @tf.function(reduce_retracing=True)
    def _resize_to(img, h: int, w: int):
        return tf.image.resize(img, (h, w), method="bilinear")

    resized_ds_map = {}
    for h, w in unique_sizes.keys():

        def _map_fn(img, hh=h, ww=w):
            return _resize_to(img, hh, ww)

        ds_hw = (
            decoded_ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        resized_ds_map[(h, w)] = ds_hw

    all_model_preds = []
    all_model_confs = []

    for model, input_size in models:
        h, w = int(input_size[0]), int(input_size[1])
        ds = resized_ds_map[(h, w)]

        preds = model.predict(ds, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        preds = np.asarray(preds)

        pred_cls = np.argmax(preds, axis=1).astype(np.int64)
        conf = preds[np.arange(preds.shape[0]), pred_cls].astype(np.float64)

        all_model_preds.append(pred_cls)
        all_model_confs.append(conf)

    pred_mat = np.stack(all_model_preds, axis=1)  # [N_present, M]
    conf_mat = np.stack(all_model_confs, axis=1)  # [N_present, M]

    n_present, m_models = pred_mat.shape
    flat_index = (
        np.arange(n_present, dtype=np.int64)[:, None] * NUM_CLASSES + pred_mat
    ).reshape(-1)
    counts_flat = np.bincount(flat_index, minlength=n_present * NUM_CLASSES)
    counts_all = counts_flat.reshape(n_present, NUM_CLASSES).astype(
        np.int16, copy=False
    )

    top = counts_all.max(axis=1)
    tied_mask = counts_all == top[:, None]
    n_tied = tied_mask.sum(axis=1)

    out_present = np.argmax(counts_all, axis=1).astype(np.int64)
    tie_rows = np.flatnonzero(n_tied > 1)

    if tie_rows.size:
        pr = pred_mat[tie_rows]  # [T, M]
        cf = conf_mat[tie_rows]  # [T, M]
        tied = tied_mask[tie_rows]  # [T, C]

        t = pr.shape[0]
        flat_index_t = (
            np.arange(t, dtype=np.int64)[:, None] * NUM_CLASSES + pr
        ).reshape(-1)
        sum_conf_flat = np.bincount(
            flat_index_t, weights=cf.reshape(-1), minlength=t * NUM_CLASSES
        )
        sum_conf = sum_conf_flat.reshape(t, NUM_CLASSES)

        vote_cnt = counts_all[tie_rows].astype(np.float64)
        mean_conf = sum_conf / np.maximum(vote_cnt, 1.0)
        mean_conf[~tied] = -np.inf  # only consider tied classes

        out_present[tie_rows] = np.argmax(mean_conf, axis=1).astype(np.int64)

    final_labels[present_idx] = out_present
else:

    @tf.function(reduce_retracing=True)
    def _decode_resize_norm(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, fallback_input_size, method="bilinear")
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    ds = (
        tf.data.Dataset.from_tensor_slices(present_paths)
        .with_options(options)
        .map(_decode_resize_norm, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    preds = fallback_model.predict(ds, verbose=0)
    preds = np.asarray(preds)
    final_labels[present_idx] = np.argmax(preds, axis=1).astype(np.int64)

submission_df = pd.DataFrame(
    {"image_id": test_files, "label": final_labels.astype(int)}
)



## === cell 5
submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path}")
print(submission_df.head())

print("Submission shape:", submission_df.shape)
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))
assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission must have columns: image_id,label"
assert out_path.endswith(".csv") and os.path.isfile(
    out_path
), "submission.csv was not created"

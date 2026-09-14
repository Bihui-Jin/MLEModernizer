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

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 1
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

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

assert os.path.exists(sample_path), f"Missing sample submission at {sample_path}"
assert os.path.exists(test_image_dir), f"Missing test images dir at {test_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"

sample_csv = pd.read_csv(sample_path)
train_df = pd.read_csv(train_csv_path)

print("sample_submission:", sample_csv.shape, sample_csv.columns.tolist())
print("train_df:", train_df.shape, train_df.columns.tolist())
print(
    "test images:",
    len([f for f in os.listdir(test_image_dir) if f.lower().endswith(".jpg")]),
)




## === cell 2
def try_load_model(path):
    if not os.path.exists(path):
        print("Missing model path:", path)
        return None
    try:
        m = tf.keras.models.load_model(path, compile=False)
        return m
    except Exception as e:
        print(f"Warning: failed to load model at {path}: {type(e).__name__}: {e}")
        return None


def infer_input_size(model):
    shp = model.inputs[0].shape  # (None, H, W, C)
    h, w = int(shp[1]), int(shp[2])
    return (h, w)


models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_4, (550, 550)),
    (model_path_5, (512, 512)),
    (model_path_6, (512, 512)),
]

models = []
for path, declared_size in models_info:
    m = try_load_model(path)
    if m is not None:
        size = (
            infer_input_size(m)
            if (hasattr(m, "inputs") and m.inputs and m.inputs[0].shape.rank == 4)
            else declared_size
        )
        models.append((m, size))
        print("Loaded:", os.path.basename(path), "input:", size)

print("Loaded model count:", len(models))




## === cell 3
def build_fallback_model(img_size=(512, 512), num_classes=5):
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_val_datasets(
    df, img_dir, img_size=(512, 512), batch_size=32, val_split=0.1
):
    df = df.copy()
    df["filepath"] = df["image_id"].apply(lambda x: os.path.join(img_dir, x))
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    val_idx = []
    for label, group in df.groupby("label"):
        idx = group.index.values
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:n_val])
    val_idx = set(val_idx)

    val_df = df[df.index.isin(val_idx)].reset_index(drop=True)
    trn_df = df[~df.index.isin(val_idx)].reset_index(drop=True)

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32)
        return img, tf.cast(label, tf.int32)

    trn_ds = tf.data.Dataset.from_tensor_slices(
        (trn_df["filepath"].values, trn_df["label"].values)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_df["filepath"].values, val_df["label"].values)
    )

    trn_ds = (
        trn_ds.shuffle(min(8192, len(trn_df)), seed=SEED, reshuffle_each_iteration=True)
        .map(_load, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds.map(_load, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return trn_ds, val_ds, trn_df, val_df


fallback_model = None
fallback_img_size = (512, 512)

if len(models) == 0:
    batch_size = 32
    epochs = 3  # keep runtime within 600s; do not add early stopping/sampling.

    trn_ds, val_ds, trn_df, val_df = make_train_val_datasets(
        train_df,
        train_image_dir,
        img_size=fallback_img_size,
        batch_size=batch_size,
        val_split=0.1,
    )
    print("Training fallback model. Train/Val:", len(trn_df), len(val_df))

    fallback_model = build_fallback_model(img_size=fallback_img_size, num_classes=5)
    history = fallback_model.fit(
        trn_ds, validation_data=val_ds, epochs=epochs, verbose=1
    )

    models = [(fallback_model, fallback_img_size)]
    print("Fallback model trained and added for inference.")



## === cell 4
test_ids = sample_csv["image_id"].tolist()

test_paths = np.array(
    [os.path.join(test_image_dir, image_id) for image_id in test_ids], dtype=object
)
missing = [p for p in test_paths.tolist() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing, e.g. {missing[:3]}"
    )

if len(models) == 0:
    raise RuntimeError(
        "No models available for inference. Check model paths and fallback training."
    )

BATCH_SIZE = 64

models_by_size = {}
for m, sz in models:
    models_by_size.setdefault(tuple(sz), []).append(m)
sizes_sorted = sorted(models_by_size.keys())

num_test = len(test_paths)
num_classes = 5
model_count = sum(len(v) for v in models_by_size.values())

DATA_OPTS = tf.data.Options()
DATA_OPTS.deterministic = True


def build_decoded_uint8_dataset(image_paths):
    image_paths = tf.convert_to_tensor(image_paths)

    def _read_decode(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8 [H,W,3]
        return img

    ds = tf.data.Dataset.from_tensor_slices(image_paths).with_options(DATA_OPTS)
    ds = ds.map(_read_decode, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


decoded_uint8_ds = build_decoded_uint8_dataset(test_paths)


def build_resized_float_dataset_from_decoded(decoded_ds, img_size, batch_size):
    h, w = int(img_size[0]), int(img_size[1])

    def _resize_norm(img_uint8):
        x = tf.image.resize(img_uint8, (h, w), method="bilinear")
        x = tf.cast(x, tf.float32) / 255.0
        return x

    ds_x = decoded_ds.with_options(DATA_OPTS).map(
        _resize_norm, num_parallel_calls=AUTOTUNE
    )
    ds_x = ds_x.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_x = ds_x.enumerate()  # (batch_number, batch_x)
    return ds_x.prefetch(AUTOTUNE)


predict_fns = {}
for sz in sizes_sorted:
    model_list = models_by_size[sz]

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _predict_sum(batch_x, _model_list=model_list):
        s = tf.zeros((tf.shape(batch_x)[0], num_classes), dtype=tf.float32)
        for mm in _model_list:
            s = s + tf.cast(mm(batch_x, training=False), tf.float32)
        return s

    predict_fns[sz] = _predict_sum

sum_probs_tf = tf.Variable(
    tf.zeros((num_test, num_classes), dtype=tf.float32), trainable=False
)

for sz in sizes_sorted:
    resized_ds = build_resized_float_dataset_from_decoded(
        decoded_uint8_ds, img_size=sz, batch_size=BATCH_SIZE
    )
    predict_sum = predict_fns[sz]

    for batch_num, batch_x in resized_ds:
        batch_sum_tf = predict_sum(batch_x)  # [bs, C]
        bs = tf.shape(batch_sum_tf)[0]
        start = tf.cast(batch_num, tf.int32) * tf.cast(BATCH_SIZE, tf.int32)
        idx = tf.range(start, start + bs, dtype=tf.int32)
        sum_probs_tf.scatter_add(tf.IndexedSlices(batch_sum_tf, idx))

avg_probs = (sum_probs_tf / float(model_count)).numpy()
pred_labels = np.argmax(avg_probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", submission_df.shape)
print(submission_df.head())



## === cell 5
sub = pd.read_csv("/kaggle/working/submission.csv")
assert sub.shape[0] == sample_csv.shape[0], "Row count mismatch vs sample_submission"
assert sub.columns.tolist() == ["image_id", "label"], "Column mismatch"
assert (
    sub["image_id"].tolist() == sample_csv["image_id"].tolist()
), "image_id ordering mismatch"
assert sub["label"].between(0, 4).all(), "Labels out of range [0,4]"
print("Submission OK.")

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
import os, glob, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)
print("Keras version " + keras.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.data.experimental.enable_autotune(True)
except Exception:
    pass




## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))




## === cell 2
FOLD_MODEL_PATHS = [
    "../input/tpus-with-5-fold/resnet50_0.h5",
    "../input/tpus-with-5-fold/resnet50_1.h5",
    "../input/tpus-with-5-fold/resnet50_2.h5",
    "../input/tpus-with-5-fold/resnet50_3.h5",
    "../input/tpus-with-5-fold/resnet50_4.h5",
]


def try_load_models(paths):
    models = []
    for p in paths:
        if os.path.exists(p):
            try:
                models.append(keras.models.load_model(p, compile=False))
                print(f"Loaded model: {p}")
            except Exception as e:
                print(f"Failed to load {p}: {e}")
        else:
            print(f"Model not found (will skip): {p}")
    return models


loaded_models = try_load_models(FOLD_MODEL_PATHS)




## === cell 3
def build_resnet50_classifier(image_size=512, num_classes=5):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.applications.resnet50.preprocess_input(inputs)
    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False  # fast + stable within time limit
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@tf.function
def _safe_decode_jpeg(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, try_recover_truncated=True)
    img.set_shape([None, None, 3])
    return img


def make_train_val_datasets(
    train_df, img_dir, image_size=512, batch_size=64, seed=42, val_frac=0.1
):
    labels = train_df["label"].values
    rng = np.random.RandomState(seed)
    idx = np.arange(len(train_df))
    val_idx = []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_val = max(1, int(round(val_frac * len(c_idx))))
        val_idx.append(c_idx[:n_val])
    val_idx = np.concatenate(val_idx)
    val_mask = np.zeros(len(train_df), dtype=bool)
    val_mask[val_idx] = True
    trn_df = train_df.loc[~val_mask].reset_index(drop=True)
    val_df = train_df.loc[val_mask].reset_index(drop=True)

    @tf.function
    def load_and_resize(image_id, label):
        img_path = tf.strings.join([img_dir, "/", image_id])
        img_bytes = tf.io.read_file(img_path)
        img = _safe_decode_jpeg(img_bytes)
        img = tf.image.resize(img, [image_size, image_size], method="bilinear")
        img = tf.cast(img, tf.float32)  # preprocessing is in model via preprocess_input
        img.set_shape([image_size, image_size, 3])
        return img, label

    def df_to_ds(df, training):
        ds = tf.data.Dataset.from_tensor_slices(
            (df["image_id"].values, df["label"].values.astype(np.int32))
        )
        if training:
            ds = ds.shuffle(
                min(len(df), 8192), seed=seed, reshuffle_each_iteration=True
            )

        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        ds = ds.with_options(options)

        ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    return df_to_ds(trn_df, True), df_to_ds(val_df, False)




## === cell 4
model_list = []

if len(loaded_models) == 0:
    train_df = pd.read_csv(TRAIN_CSV)
    assert set(train_df.columns) >= {"image_id", "label"}
    train_df["label"] = train_df["label"].astype(int)

    train_ds, val_ds = make_train_val_datasets(
        train_df,
        TRAIN_IMG_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        val_frac=0.1,
    )

    fallback_model = build_resnet50_classifier(
        image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )

    EPOCHS = 3
    history = fallback_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1
    )
    model_list = [fallback_model]
else:
    model_list = loaded_models

print("Number of models used for prediction:", len(model_list))




## === cell 5
test_dir = TEST_IMG_DIR

sample_df = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_df["image_id"].astype(str).tolist()


def _build_multi_output_ensemble(models):
    if len(models) == 1:
        return models[0]
    inp = keras.Input(shape=models[0].inputs[0].shape[1:], name="ensemble_input")
    outs = []
    for i, m in enumerate(models):
        outs.append(m(inp))
    return keras.Model(inputs=inp, outputs=outs, name="ensemble_multi_output")


def get_preds_model_list(image_ids, img_dir, model_obj_list, batch_size=BATCH_SIZE):
    n = len(image_ids)
    if n == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    if len(model_obj_list) == 0:
        return pd.DataFrame(
            {"image_id": image_ids, "label": np.zeros(n, dtype=np.int64)}
        )

    file_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(image_ids, dtype=tf.string)
    )

    @tf.function
    def load_resize_from_id(image_id):
        path = tf.strings.join([img_dir, "/", image_id])
        img_bytes = tf.io.read_file(path)
        img = _safe_decode_jpeg(img_bytes)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32)  # model does preprocess_input internally
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True

    ds = (
        file_ds.with_options(options)
        .map(load_resize_from_id, num_parallel_calls=tf.data.AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    steps = (n + batch_size - 1) // batch_size

    ensemble = _build_multi_output_ensemble(model_obj_list)

    pred = ensemble.predict(ds, verbose=0, steps=steps)
    if isinstance(pred, list):
        avg_probs = np.mean(
            np.stack([p.astype(np.float32, copy=False) for p in pred], axis=0), axis=0
        )
    else:
        avg_probs = pred.astype(np.float32, copy=False)

    preds = avg_probs.argmax(axis=1).astype(np.int64)
    return pd.DataFrame({"image_id": image_ids, "label": preds})


predict_df = get_preds_model_list(test_image_ids, test_dir, model_list)




## === cell 6
out_df = sample_df[["image_id"]].merge(predict_df, on="image_id", how="left")
out_df["label"] = out_df["label"].fillna(0).astype(int)

assert list(out_df.columns) == ["image_id", "label"]
assert len(out_df) == len(sample_df)

out_path = "submission.csv"
out_df.to_csv(out_path, index=False)
print("Wrote submission.csv with shape:", out_df.shape)
print(out_df.head())




## === cell 7
try:
    display(out_df.head(10))
except Exception:
    print(out_df.head(10))

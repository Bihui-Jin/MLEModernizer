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

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as _e:
    print("Warning: could not set TF threading:", repr(_e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as _e:
    print("Warning: could not enable XLA JIT:", repr(_e))

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 1
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant((0,))
    for i in range(len(y_true)):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(count, tf.TensorShape([None]))]
        )
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]):
            count = tf.math.add(count, 1)
    return float(count) / float(len(y_true))


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true = tf.math.add(
            y_true,
            tf.math.add(
                tf.math.multiply(label_smoothing / 2.0, tf.math.add(1.0, -1 * y_true)),
                tf.math.multiply(-1 * label_smoothing / 2.0, y_true),
            ),
        )
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        lamb = tf.math.divide(
            tf.math.multiply(K.sum(y_temp), K.sum(y_temp)),
            K.sum(tf.math.multiply(y_temp, y_temp)),
        )
        loss = tf.constant((0.0,))
        for i in range(len(y_true[0])):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(loss, tf.TensorShape([None]))]
            )
            temp = tf.constant((0.0,))
            loss = tf.math.add(
                loss,
                tf.math.add(
                    temp,
                    (
                        -1.0
                        * (1 / float(len(y_true)))
                        * tf.math.multiply(
                            y_true[:, i], K.log(y_temp[:, i] + f0 / lamb)
                        )
                    ),
                ),
            )
        return tf.math.reduce_sum(loss)

    return loss_gamb




## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_v1 = sample_sub[["image_id"]].copy()
test_v2 = sample_sub[["image_id"]].copy()

assert os.path.isdir(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

missing = [
    fn
    for fn in test_v1["image_id"].head(20).tolist()
    if not os.path.exists(os.path.join(TEST_DIR, fn))
]
if missing:
    print(
        "Warning: some sample_submission image_ids not found (first 20 checked):",
        missing,
    )

print("Test rows:", len(sample_sub))




## === cell 3
MODEL1_DIR = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
MODEL2_DIR = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"


def _savedmodel_exists(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    )


def _unwrap_savedmodel_output(y):
    if isinstance(y, dict):
        if "outputs" in y:
            return y["outputs"]
        return y[sorted(y.keys())[0]]
    return y


def build_inference_model_from_savedmodel(
    savedmodel_dir, input_size, call_endpoint="serving_default"
):
    """
    Wrap a TF SavedModel as a tf.keras.Model for prediction.

    BUGFIX/ROBUSTNESS:
    - Prefer tf.keras.layers.TFSMLayer when available (typical in TF 2.11+).
    - Fall back to keras.layers.TFSMLayer (Keras 3) if present.
    - Last resort: tf.saved_model.load + signature.
    """
    inp = tf.keras.Input(shape=(input_size, input_size, 3), dtype=tf.float32)

    layer = None
    last_err = None

    try:
        layer = tf.keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
    except Exception as e:
        last_err = e
        layer = None

    if layer is None:
        try:
            from keras.layers import TFSMLayer as K3TFSMLayer

            layer = K3TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
        except Exception as e:
            last_err = e
            layer = None

    if layer is not None:
        out = layer(inp)
        out = _unwrap_savedmodel_output(out)
        return tf.keras.Model(inp, out)

    print(f"Warning: TFSMLayer load failed for {savedmodel_dir}: {repr(last_err)}")
    reloaded = tf.saved_model.load(savedmodel_dir)
    if hasattr(reloaded, "signatures") and call_endpoint in reloaded.signatures:
        fn = reloaded.signatures[call_endpoint]

        def _call(x):
            y = fn(x)
            return _unwrap_savedmodel_output(y)

        out = tf.keras.layers.Lambda(_call)(inp)
        return tf.keras.Model(inp, out)

    raise OSError(
        f"SavedModel at {savedmodel_dir} does not expose signature '{call_endpoint}'."
    )


use_external_models = _savedmodel_exists(MODEL1_DIR) and _savedmodel_exists(MODEL2_DIR)
print("External SavedModels available:", use_external_models)

model_v1, model_v2 = None, None
if use_external_models:
    model_v1 = build_inference_model_from_savedmodel(
        MODEL1_DIR, input_size=448, call_endpoint="serving_default"
    )
    model_v2 = build_inference_model_from_savedmodel(
        MODEL2_DIR, input_size=512, call_endpoint="serving_default"
    )
    print("Loaded external models.")
    print("model_v1 output shape:", model_v1.output_shape)
    print("model_v2 output shape:", model_v2.output_shape)
else:
    print(
        "External models not found. Will train a small fallback model from train_images to produce a valid submission."
    )




## === cell 4
def make_paths_ds(image_ids, directory, shuffle=False):
    image_ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    dir_prefix = tf.constant(directory + "/", dtype=tf.string)
    paths = tf.strings.join([dir_prefix, image_ids])

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=tf.shape(paths)[0], seed=42, reshuffle_each_iteration=False
        )

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    ds = ds.with_options(options)
    return ds


def make_decoded_ds(paths_ds):
    @tf.function
    def _decode(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8
        return img

    return paths_ds.map(_decode, num_parallel_calls=AUTOTUNE)


def make_batched_two_size_ds(image_ids, directory, batch_size):
    paths_ds = make_paths_ds(image_ids, directory, shuffle=False)
    decoded_ds = make_decoded_ds(paths_ds)

    decoded_ds = decoded_ds.cache()

    @tf.function
    def _to_512_and_448(img):
        img512 = tf.image.resize(img, (512, 512), method=tf.image.ResizeMethod.BILINEAR)
        img512 = tf.cast(img512, tf.float32)
        img448 = tf.image.resize_with_crop_or_pad(img512, 448, 448)
        return img512, img448

    ds = decoded_ds.map(_to_512_and_448, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


pred_v1, pred_v2 = None, None

if use_external_models:
    @tf.function(reduce_retracing=True)
    def _infer_both(batch512, batch448):
        y1 = model_v1(batch448, training=False)
        y2 = model_v2(batch512, training=False)
        y1 = _unwrap_savedmodel_output(y1)
        y2 = _unwrap_savedmodel_output(y2)
        return y1, y2

    BS_SHARED = 64  # same semantics; larger batch reduces per-step overhead, keeps full data and exact logic
    n_test = len(sample_sub)

    two_size_ds = make_batched_two_size_ds(
        sample_sub["image_id"].values, TEST_DIR, BS_SHARED
    )

    pred_v1 = np.empty((n_test, 5), dtype=np.float32)
    pred_v2_full = np.empty((n_test, 6), dtype=np.float32)  # may be 6 for gambler model

    offset = 0
    for batch512, batch448 in two_size_ds:
        y1, y2 = _infer_both(batch512, batch448)
        y1 = tf.convert_to_tensor(y1)
        y2 = tf.convert_to_tensor(y2)

        bsz = int(y1.shape[0])
        pred_v1[offset : offset + bsz, :] = y1.numpy()

        y2_np = y2.numpy()
        if y2_np.shape[1] == 6:
            pred_v2_full[offset : offset + bsz, :] = y2_np
        else:
            pred_v2_full[offset : offset + bsz, :5] = y2_np
        offset += bsz

    pred_v1 = pred_v1[:n_test]
    if pred_v2_full.shape[1] == 6:
        pred_v2 = pred_v2_full[
            :n_test, 1:
        ]  # drop abstain column, identical to original logic
    else:
        pred_v2 = pred_v2_full[:n_test, :5]

    print("pred_v1 shape:", pred_v1.shape)
    print("pred_v2 shape:", pred_v2.shape)
else:
    assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
    assert os.path.isdir(TRAIN_DIR), f"Missing train_images at {TRAIN_DIR}"

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["label"] = train_df["label"].astype(str)

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    IMG_SIZE = 224
    BATCH = 32

    @tf.function
    def _parse_train(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    @tf.function
    def _augment(img, label):
        seed = tf.constant([42, 123], dtype=tf.int32)
        h = tf.strings.to_hash_bucket_fast(tf.as_string(tf.reduce_sum(img)), 2**31 - 1)
        s = tf.stack([seed[0], tf.cast(h, tf.int32) ^ seed[1]])

        img = tf.image.stateless_random_flip_left_right(img, seed=s)
        img = tf.image.stateless_random_brightness(img, max_delta=0.1, seed=s + 1)
        img = tf.image.stateless_random_contrast(img, lower=0.9, upper=1.1, seed=s + 2)
        return img, label

    num_classes = 5

    tr_paths = (TRAIN_DIR + "/" + tr_df["image_id"].astype(str)).values
    va_paths = (TRAIN_DIR + "/" + va_df["image_id"].astype(str)).values

    tr_labels = tf.keras.utils.to_categorical(
        tr_df["label"].astype(int).values, num_classes=num_classes
    )
    va_labels = tf.keras.utils.to_categorical(
        va_df["label"].astype(int).values, num_classes=num_classes
    )

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
    va_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    tr_ds = tr_ds.with_options(opts).shuffle(
        len(tr_paths), seed=42, reshuffle_each_iteration=True
    )
    tr_ds = tr_ds.map(_parse_train, num_parallel_calls=AUTOTUNE)
    tr_ds = tr_ds.map(_augment, num_parallel_calls=AUTOTUNE)
    tr_ds = tr_ds.batch(BATCH).prefetch(AUTOTUNE)

    va_ds = va_ds.with_options(opts)
    va_ds = va_ds.map(_parse_train, num_parallel_calls=AUTOTUNE)
    va_ds = va_ds.batch(BATCH).prefetch(AUTOTUNE)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fallback_model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=2,
        verbose=2,
    )

    test_paths = (TEST_DIR + "/" + sample_sub["image_id"].astype(str)).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)

    @tf.function
    def _parse_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds = (
        test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE)
        .batch(BATCH)
        .prefetch(AUTOTUNE)
    )

    pred = fallback_model.predict(test_ds, verbose=1)
    pred_v1 = pred
    pred_v2 = pred
    print("Fallback predictions shape:", pred.shape)




## === cell 5
pred_new = 0.5 * pred_v1 + 0.5 * pred_v2
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

assert len(predicted_class_indices_new) == len(sample_sub), (
    len(predicted_class_indices_new),
    len(sample_sub),
)

submission = sample_sub.copy()
submission["label"] = predicted_class_indices_new

submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
print("Label value counts (head):")
print(submission["label"].value_counts().head())

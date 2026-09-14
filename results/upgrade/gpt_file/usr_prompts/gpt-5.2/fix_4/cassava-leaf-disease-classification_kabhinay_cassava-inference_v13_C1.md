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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE




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


def build_inference_model_from_savedmodel(
    savedmodel_dir, input_size, call_endpoint="serving_default"
):
    """
    Wrap a TF SavedModel as a tf.keras.Model for prediction.
    Tries keras.layers.TFSMLayer (Keras 3), and falls back to tf.saved_model.load if needed.
    """
    inp = tf.keras.Input(shape=(input_size, input_size, 3), dtype=tf.float32)

    try:
        from keras.layers import TFSMLayer  # keras (standalone) in recent Kaggle images

        layer = TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
        out = layer(inp)
    except Exception as e:
        print(f"Warning: TFSMLayer load failed for {savedmodel_dir}: {repr(e)}")
        reloaded = tf.saved_model.load(savedmodel_dir)
        if hasattr(reloaded, "signatures") and call_endpoint in reloaded.signatures:
            fn = reloaded.signatures[call_endpoint]

            def _call(x):
                y = fn(x)
                if isinstance(y, dict):
                    if "outputs" in y:
                        y = y["outputs"]
                    else:
                        y = y[sorted(y.keys())[0]]
                return y

            out = tf.keras.layers.Lambda(_call)(inp)
        else:
            raise OSError(
                f"SavedModel at {savedmodel_dir} does not expose signature '{call_endpoint}'."
            )

    if isinstance(out, dict):
        if "outputs" in out:
            out = out["outputs"]
        else:
            out = out[sorted(out.keys())[0]]
    return tf.keras.Model(inp, out)


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
def make_image_ds(image_ids, directory, target_size, batch_size, shuffle=False):
    image_ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    paths = tf.strings.join([tf.constant(directory + "/", dtype=tf.string), image_ids])

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(
            img, tf.float32
        )  # keep same scale as original generator (no /255)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=tf.shape(paths)[0], seed=42, reshuffle_each_iteration=False
        )
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _predict_step(model, x):
    return model(x, training=False)


def predict_ds(model, ds):
    outs = []
    for xb in ds:
        yb = _predict_step(model, xb)
        outs.append(tf.convert_to_tensor(yb))
    return tf.concat(outs, axis=0).numpy()


pred_v1, pred_v2 = None, None

if use_external_models:
    BS_V1 = 16
    BS_V2 = 8

    ds_v1 = make_image_ds(
        test_v1["image_id"].values,
        TEST_DIR,
        target_size=(448, 448),
        batch_size=BS_V1,
        shuffle=False,
    )
    ds_v2 = make_image_ds(
        test_v2["image_id"].values,
        TEST_DIR,
        target_size=(512, 512),
        batch_size=BS_V2,
        shuffle=False,
    )

    pred_v1 = predict_ds(model_v1, ds_v1)
    temp_v2 = predict_ds(model_v2, ds_v2)

    pred_v2 = (
        temp_v2[:, 1:]
        if (len(temp_v2.shape) == 2 and temp_v2.shape[1] == 6)
        else temp_v2
    )

    print("pred_v1 shape:", pred_v1.shape)
    print("pred_v2 shape:", pred_v2.shape)
else:
    assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
    assert os.path.isdir(TRAIN_DIR), f"Missing train_images at {TRAIN_DIR}"

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["label"] = train_df["label"].astype(
        str
    )  # for flow_from_dataframe categorical mode

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    IMG_SIZE = 224
    BATCH = 32

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    ).flow_from_dataframe(
        tr_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
        va_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
    )

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
        train_gen,
        validation_data=valid_gen,
        epochs=2,
        verbose=2,
    )

    test_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
        sample_sub[["image_id"]],
        directory=TEST_DIR,
        x_col="image_id",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode=None,
        shuffle=False,
    )

    pred = fallback_model.predict(test_gen, verbose=1, steps=len(test_gen))
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

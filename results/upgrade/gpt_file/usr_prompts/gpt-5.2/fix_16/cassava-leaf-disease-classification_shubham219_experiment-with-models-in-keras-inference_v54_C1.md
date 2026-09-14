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
import sys
import glob
import numpy as np
import pandas as pd

if (
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ
    and os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"
):
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
DEBUG = False

np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def _pick_data_root(candidates):
    for p in candidates:
        if os.path.exists(os.path.join(p, "sample_submission.csv")) and os.path.exists(
            os.path.join(p, "test_images")
        ):
            return p
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DATA_ROOT = _pick_data_root(DATA_ROOT_CANDIDATES)

TEST_IMG_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

WEIGHT_PATH = (
    "/kaggle/input/experiment-with-models-using-keras-with-updates/model_v0.25.h5"
)

print("Python:", sys.version)
print("DATA_ROOT:", DATA_ROOT, "exists:", os.path.exists(DATA_ROOT))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train csv exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train image dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test images glob example:", TEST_IMG_GLOB)
print("Weight path exists:", os.path.exists(WEIGHT_PATH))




## === cell 1
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    _cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu_cnt)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TensorFlow:", tf.__version__)

custom_objects = {}

try:
    from tensorflow.keras.layers import DepthwiseConv2D

    custom_objects["DepthwiseConv2D"] = DepthwiseConv2D
except Exception:
    pass

try:
    custom_objects["swish"] = tf.nn.swish
except Exception:
    pass


class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


custom_objects["FixedDropout"] = FixedDropout


def _find_weight_file(preferred_path: str):
    """Try to locate the .h5 if the originally referenced dataset isn't attached."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    base_dir = "/kaggle/input"
    target_name = "model_v0.25.h5"
    if os.path.isdir(base_dir):
        try:
            for d1 in os.listdir(base_dir):
                p1 = os.path.join(base_dir, d1)
                if not os.path.isdir(p1):
                    continue
                cand = os.path.join(p1, target_name)
                if os.path.exists(cand):
                    return cand
                try:
                    for d2 in os.listdir(p1):
                        p2 = os.path.join(p1, d2)
                        if not os.path.isdir(p2):
                            continue
                        cand2 = os.path.join(p2, target_name)
                        if os.path.exists(cand2):
                            return cand2
                except Exception:
                    continue
        except Exception:
            pass

    hits = glob.glob(os.path.join(base_dir, "*", "*.h5"))
    for h in hits:
        if os.path.basename(h) == target_name:
            return h
    return None


MODEL_TARGET_SIZE = (512, 512)
MODEL_INPUT_SHAPE = (512, 512, 3)

resolved_weight_path = _find_weight_file(WEIGHT_PATH)
print("Resolved weight path:", resolved_weight_path)

USE_EFFNET_FALLBACK = False
IDX_TO_LABEL = None

if resolved_weight_path is not None and os.path.exists(resolved_weight_path):
    my_model = load_model(
        resolved_weight_path, custom_objects=custom_objects, compile=False
    )
    print("Model loaded. Output shape:", my_model.output_shape)

    try:
        inferred = my_model.input_shape
        if isinstance(inferred, list):
            inferred = inferred[0]
        if inferred is not None and len(inferred) == 4:
            h, w, c = inferred[1], inferred[2], inferred[3]
            if (h is not None) and (w is not None) and (c == 3):
                MODEL_TARGET_SIZE = (int(h), int(w))
                MODEL_INPUT_SHAPE = (int(h), int(w), 3)
    except Exception as e:
        print("WARNING: could not infer model input shape; using default 512x512.", e)
else:
    USE_EFFNET_FALLBACK = True
    print(
        "WARNING: pretrained .h5 not found; using EfficientNetB0 fallback + head training."
    )

    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    MODEL_TARGET_SIZE = (224, 224)
    MODEL_INPUT_SHAPE = (224, 224, 3)

    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=MODEL_INPUT_SHAPE
    )
    base.trainable = False

    inputs = layers.Input(shape=MODEL_INPUT_SHAPE)
    x = effnet_preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=SEED)(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    my_model = Model(inputs, outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

print(
    "Final MODEL_TARGET_SIZE:",
    MODEL_TARGET_SIZE,
    "MODEL_INPUT_SHAPE:",
    MODEL_INPUT_SHAPE,
    "USE_EFFNET_FALLBACK:",
    USE_EFFNET_FALLBACK,
)




## === cell 2
def _train_head_if_needed():
    if not USE_EFFNET_FALLBACK:
        return

    if (not os.path.exists(TRAIN_CSV_PATH)) or (not os.path.isdir(TRAIN_IMG_DIR)):
        raise FileNotFoundError(
            f"Fallback training requested but train data not found. "
            f"TRAIN_CSV_PATH exists={os.path.exists(TRAIN_CSV_PATH)}, "
            f"TRAIN_IMG_DIR exists={os.path.isdir(TRAIN_IMG_DIR)}"
        )

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_train["path"] = df_train["image_id"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, x)
    )
    if DEBUG:
        df_train = df_train.sample(
            n=min(1000, len(df_train)), random_state=SEED
        ).reset_index(drop=True)

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0,  # images are uint8 -> float32; keep scale and let model preprocess handle it
        validation_split=0.1,
    )

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="path",
        y_col="label",
        target_size=MODEL_TARGET_SIZE,
        color_mode="rgb",
        class_mode="raw",
        batch_size=32,
        shuffle=True,
        subset="training",
        seed=SEED,
    )
    val_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="path",
        y_col="label",
        target_size=MODEL_TARGET_SIZE,
        color_mode="rgb",
        class_mode="raw",
        batch_size=32,
        shuffle=False,
        subset="validation",
        seed=SEED,
    )

    my_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1,
    )


_train_head_if_needed()




## === cell 3
test_images = sorted(glob.glob(TEST_IMG_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found via glob: {TEST_IMG_GLOB}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = [os.path.basename(p) for p in df_test["path"].values]

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    df_test = sample_sub[["image_id"]].merge(
        df_test[["image_id", "path"]], on="image_id", how="left"
    )
    if df_test["path"].isna().any():
        missing = df_test[df_test["path"].isna()]["image_id"].head(5).tolist()
        raise FileNotFoundError(
            f"Some sample_submission image_ids not found in test_images. Example missing: {missing}"
        )


def _decode_resize_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, MODEL_TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def make_test_gen(batch_size=64):
    options = tf.data.Options()
    options.deterministic = True

    paths = df_test["path"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(
        _decode_resize_tf, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


print("Test images:", len(df_test))
print("Example test row:", df_test.head(1).to_dict(orient="records")[0])




## === cell 4
pred_list = []
for _ in range(1):
    test_gen = make_test_gen(batch_size=256)

    pred_test = my_model.predict(
        test_gen,
        verbose=1,
    )
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)

pred_test_labels = pred_test_idx

if pred_test_labels.shape[0] != len(df_test):
    raise ValueError(
        f"Predictions count {pred_test_labels.shape[0]} != test rows {len(df_test)}"
    )

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

final_csv["image_id"] = final_csv["image_id"].astype(str)
final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 5
print(final_csv["label"].value_counts().sort_index())
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv size (bytes):", os.path.getsize("submission.csv"))

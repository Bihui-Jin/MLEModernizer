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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
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
    patterns = [
        "/kaggle/input/**/*.h5",
        "/kaggle/input/**/*.keras",
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        for h in hits:
            if os.path.basename(h) == "model_v0.25.h5":
                return h
        if len(hits) > 0:
            return hits[0]
    return None


resolved_weight_path = _find_weight_file(WEIGHT_PATH)
print("Resolved weight path:", resolved_weight_path)


def _build_fallback_model_imagenet(input_shape=(224, 224, 3), num_classes=5):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False  # keep backbone frozen for minimal-change, fast training

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


USE_EFFNET_FALLBACK = False
MODEL_TARGET_SIZE = (512, 512)  # overridden if model expects a different size
MODEL_INPUT_SHAPE = (512, 512, 3)

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
    print(
        "WARNING: Pretrained cassava weight file not found. "
        "To improve score toward target vs near-random predictions, we will "
        "train a small softmax head on top of frozen ImageNet EfficientNetB0 "
        "using train.csv + train_images, then run the same predict→argmax pipeline."
    )
    USE_EFFNET_FALLBACK = True
    MODEL_TARGET_SIZE = (224, 224)
    MODEL_INPUT_SHAPE = (224, 224, 3)
    my_model = _build_fallback_model_imagenet(
        input_shape=MODEL_INPUT_SHAPE, num_classes=5
    )
    print("Fallback model built. Output shape:", my_model.output_shape)

print(
    "Final MODEL_TARGET_SIZE:",
    MODEL_TARGET_SIZE,
    "MODEL_INPUT_SHAPE:",
    MODEL_INPUT_SHAPE,
)




## === cell 2
def _train_head_if_needed():
    global IDX_TO_LABEL

    if not USE_EFFNET_FALLBACK:
        return

    if not (os.path.exists(TRAIN_CSV_PATH) and os.path.exists(TRAIN_IMG_DIR)):
        raise FileNotFoundError(
            f"Fallback training requested but missing TRAIN_CSV_PATH={TRAIN_CSV_PATH} "
            f"or TRAIN_IMG_DIR={TRAIN_IMG_DIR}"
        )

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    if "image_id" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train.csv must contain columns: image_id, label")

    train_df = train_df.copy()
    img_ids = train_df["image_id"].astype(str).values
    train_df["path"] = [os.path.join(TRAIN_IMG_DIR, x) for x in img_ids]
    train_df["label"] = train_df["label"].astype(str)

    if train_df["path"].isna().any():
        raise ValueError("Some training paths are NaN after join.")
    if not os.path.exists(train_df["path"].iloc[0]):
        raise FileNotFoundError(
            f"Train image path does not exist: {train_df['path'].iloc[0]}"
        )

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    val_frac = 0.10
    n_val = int(len(idx) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    tr_idg = ImageDataGenerator(preprocessing_function=preprocess_fn)
    val_idg = ImageDataGenerator(preprocessing_function=preprocess_fn)

    batch_size = 32

    tr_gen = tr_idg.flow_from_dataframe(
        dataframe=df_tr,
        x_col="path",
        y_col="label",
        target_size=MODEL_TARGET_SIZE,
        batch_size=batch_size,
        seed=SEED,
        shuffle=True,
        class_mode="sparse",
    )
    val_gen = val_idg.flow_from_dataframe(
        dataframe=df_val,
        x_col="path",
        y_col="label",
        target_size=MODEL_TARGET_SIZE,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode="sparse",
    )

    inv = {v: int(k) for k, v in tr_gen.class_indices.items()}
    IDX_TO_LABEL = np.array([inv[i] for i in range(len(inv))], dtype=int)
    print("Fallback class_indices:", tr_gen.class_indices)
    print("IDX_TO_LABEL:", IDX_TO_LABEL.tolist())

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 3 if not DEBUG else 1

    steps_per_epoch = max(1, len(df_tr) // batch_size)
    validation_steps = max(1, len(df_val) // batch_size)

    history = my_model.fit(
        tr_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        verbose=1,
    )
    last = history.history
    print(
        "Fallback training done. Final train acc:",
        float(last.get("accuracy", [np.nan])[-1]),
        "Final val acc:",
        float(last.get("val_accuracy", [np.nan])[-1]),
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


def make_test_gen(batch_size=64):
    if USE_EFFNET_FALLBACK:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    else:
        preprocess_fn = None

    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=MODEL_TARGET_SIZE,
    )
    return test_gen


print("Test images:", len(df_test))
print("Example test row:", df_test.head(1).to_dict(orient="records")[0])



## === cell 4
pred_list = []
for _ in range(1):
    test_gen = make_test_gen(batch_size=128)

    pred_test = my_model.predict(
        test_gen,
        verbose=1,
    )
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)

if USE_EFFNET_FALLBACK:
    if IDX_TO_LABEL is None:
        raise RuntimeError(
            "USE_EFFNET_FALLBACK=True but IDX_TO_LABEL is None; training mapping was not created."
        )
    pred_test_labels = IDX_TO_LABEL[pred_test_idx]
else:
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

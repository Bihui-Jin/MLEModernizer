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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUB_PATH = "/kaggle/working/submission.csv"

MODEL_PATH = "/kaggle/input/cropnet_from_kaggle/tensorflow2/default/1/kaggle/working/cropnet_model_tf"

IMG_SIZE = (224, 224)
NUM_CLASSES = 5

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.AUTOTUNE  # existence check
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
def load_and_preprocess_image_bytes(img_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img  # (H, W, 3)


def load_and_preprocess_image(img_path: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(img_path)
    img = load_and_preprocess_image_bytes(img_bytes)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


def _extract_logits_or_probs(pred):
    """
    pred: output of infer(...) which may be a dict of tensors or a tensor.
    Returns: a 2D numpy array shape (batch, num_classes).
    """
    if isinstance(pred, dict):
        keys = list(pred.keys())
        preferred = None
        for k in keys:
            lk = str(k).lower()
            if "logit" in lk or "prob" in lk or "pred" in lk or "output" in lk:
                preferred = k
                break
        if preferred is None:
            preferred = sorted(keys, key=lambda x: str(x))[0]
        t = pred[preferred]
    else:
        t = pred

    t = tf.convert_to_tensor(t)
    arr = t.numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def find_savedmodel_dir(start_path: str) -> str | None:
    """
    Returns a directory containing saved_model.pb (or saved_model.pbtxt), searching:
    - start_path itself
    - its parents (a few levels up)
    - (bounded) within the dataset root ONLY as a last resort (avoid a broad /kaggle/input walk)
    """

    def is_sm_dir(d):
        return (
            os.path.isdir(d)
            and (
                os.path.exists(os.path.join(d, "saved_model.pb"))
                or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
            )
            and os.path.isdir(os.path.join(d, "variables"))
        )

    p = start_path
    for _ in range(8):
        if is_sm_dir(p):
            return p
        parent = os.path.dirname(p.rstrip("/"))
        if parent == p:
            break
        p = parent

    roots = []
    if start_path.startswith("/kaggle/input/"):
        parts = start_path.split("/")
        if len(parts) >= 4:
            roots.append("/".join(parts[:4]))  # /kaggle/input/<dataset>
    else:
        return None

    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                if is_sm_dir(dirpath):
                    candidates.append(dirpath)
                    break
            rel_depth = dirpath[len(root) :].count(os.sep)
            if rel_depth > 10:
                dirnames[:] = []
        if candidates:
            break

    if not candidates:
        return None

    def score_path(d):
        ld = d.lower()
        s = 0
        if "cropnet" in ld:
            s += 5
        if "cassava" in ld:
            s += 3
        if "model" in ld:
            s += 1
        return s

    candidates.sort(key=score_path, reverse=True)
    return candidates[0]




## === cell 2
infer = None
loaded = None

resolved_model_dir = find_savedmodel_dir(MODEL_PATH)
if resolved_model_dir is not None:
    try:
        print("Resolved SavedModel directory:", resolved_model_dir)
        loaded = tf.saved_model.load(resolved_model_dir)

        if (
            hasattr(loaded, "signatures")
            and isinstance(loaded.signatures, dict)
            and len(loaded.signatures) > 0
        ):
            if "serving_default" in loaded.signatures:
                infer = loaded.signatures["serving_default"]
            else:
                infer = next(iter(loaded.signatures.values()))
        else:
            infer = loaded
    except Exception as e:
        print("Warning: Failed to load SavedModel from:", resolved_model_dir)
        print("Reason:", repr(e))
        infer = None
else:
    print(
        "Warning: Could not find a SavedModel to load under MODEL_PATH (bounded search)."
    )

OUTPUT_KEYS = None
if infer is not None:
    try:
        dummy = tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
        out = infer(dummy)
        if isinstance(out, dict):
            OUTPUT_KEYS = list(out.keys())
    except Exception:
        OUTPUT_KEYS = None

print("Using inference callable:", type(infer))
print("Detected output keys:", OUTPUT_KEYS)

if infer is not None:
    try:
        infer = tf.function(infer, reduce_retracing=True, jit_compile=False)
    except Exception:
        pass




## === cell 3
def build_fallback_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False  # keep lightweight and fast
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = inputs
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2, seed=42)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_ds(
    df: pd.DataFrame, batch_size: int = 32, shuffle: bool = True
) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).values
    labels = df["label"].astype(np.int64).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = load_and_preprocess_image_bytes(img_bytes)
        return img, label

    options = tf.data.Options()
    options.experimental_deterministic = True

    if shuffle:
        ds = ds.shuffle(min(len(df), 8192), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    return ds


fallback_model = None
if infer is None:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df["image_path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
    train_df = train_df[train_df["image_path"].map(os.path.exists)].reset_index(
        drop=True
    )

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds = make_train_ds(tr_df, batch_size=32, shuffle=True)
    val_ds = make_train_ds(va_df, batch_size=32, shuffle=False)

    fallback_model = build_fallback_model()
    fallback_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    def infer(x):
        return fallback_model(x, training=False)

    infer = tf.function(infer, reduce_retracing=True, jit_compile=False)

print("Fallback model used:", fallback_model is not None)




## === cell 4
def predict_batch(image_paths: list[str], batch_size: int = 32) -> np.ndarray:
    paths = tf.constant(image_paths)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = load_and_preprocess_image_bytes(img_bytes)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _predict_labels(xb):
        out = infer(xb)
        if isinstance(out, dict):
            keys = tf.nest.flatten(list(out.keys()))
            return out  # placeholder (handled outside tf.function)

        return tf.argmax(out, axis=1, output_type=tf.int64)

    selected_key = None
    if infer is not None:
        dummy = tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
        warm = infer(dummy)
        if isinstance(warm, dict):
            keys = list(warm.keys())
            preferred = None
            for k in keys:
                lk = str(k).lower()
                if "logit" in lk or "prob" in lk or "pred" in lk or "output" in lk:
                    preferred = k
                    break
            if preferred is None:
                preferred = sorted(keys, key=lambda x: str(x))[0]
            selected_key = preferred

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _predict_labels_fast(xb):
        out = infer(xb)
        if selected_key is not None:
            out = out[selected_key]
        out = tf.convert_to_tensor(out)
        if out.shape.rank == 1:
            out = out[None, :]
        return tf.argmax(out, axis=1, output_type=tf.int64)

    preds = []
    for xb in ds:
        preds.append(_predict_labels_fast(xb))
    if not preds:
        return np.array([], dtype=np.int64)
    return tf.concat(preds, axis=0).numpy()




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv missing required column: image_id")

image_ids = sample_sub["image_id"].astype(str).tolist()

image_paths = [TEST_IMG_DIR + "/" + image_id for image_id in image_ids]

try:
    test_files = set(os.listdir(TEST_IMG_DIR))
except FileNotFoundError:
    test_files = set()

missing_mask = np.fromiter(
    (img_id not in test_files for img_id in image_ids), dtype=bool, count=len(image_ids)
)
missing = int(missing_mask.sum())
if missing:
    print(
        f"Warning: {missing} images listed in sample_submission.csv were not found on disk."
    )

pred_labels = np.zeros(len(image_paths), dtype=np.int64)
if (~missing_mask).any():
    existing_idx = np.where(~missing_mask)[0]
    existing_paths = [image_paths[i] for i in existing_idx]

    bs = 64 if tf.config.list_physical_devices("GPU") else 32

    pred_existing = predict_batch(existing_paths, batch_size=bs)
    pred_labels[existing_idx] = pred_existing

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels.tolist()})
submission_df.to_csv(SUB_PATH, index=False)

print("Submission file created:", SUB_PATH)
print(submission_df.head())
print("Rows:", len(submission_df), "Columns:", list(submission_df.columns))
print("Label value counts:\n", submission_df["label"].value_counts().sort_index())

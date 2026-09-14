# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7886068298579632

# 6. Current score

0.61734

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61323) has done: 'I (1) fix the protobuf/TensorFlow crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that breaks modern TF/protobuf, (2) make model loading robust by searching `/kaggle/input/**` for the `.keras` files when the hardcoded paths don’t exist, and (3) ensure the TFRecord feature specs are defined before use so `NameError` cannot happen. I also add a safe fallback “base-model-only” prediction path if the external Keras models truly aren’t available in this environment, so the notebook still trains and writes a valid `submission.csv` end-to-end. Core logic remains the same: extract concatenated base-model probabilities per image and train a `DecisionTreeClassifier` stacker; the fallback is only used to avoid runtime failure when models are missing.'
- What this solution (achieved 0.61323) has done: 'The immediate blocker is the TensorFlow/protobuf import crash in cell 0 (`MessageFactory.GetPrototype`), which is caused by an incompatible protobuf runtime being imported before TensorFlow in this environment. I fix this by forcing the pure-Python protobuf implementation *before* importing TensorFlow (and clearing any conflicting settings), which is the minimal change to make TF import reliably under Kaggle’s package set. After that, I keep your stacking logic unchanged, but I also ensure deterministic, correctly-aligned submission generation by de-duplicating any repeated `image_id` entries from TFRecords before merging (score-neutral correctness/stability). The rest of the pipeline (TFRecord parsing → feature extraction via loaded models or fallback → DecisionTree training → submission.csv) is preserved.'
- What this solution (achieved 0.61323) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override and instead ensuring no conflicting protobuf env vars are set before importing TensorFlow. I also add a small amount of defensive setup around TFRecord reading (deterministic options and AUTOTUNE) without changing the model/stacking logic, so parsing is stable and finishes reliably. Finally, I keep your existing submission alignment/merge logic but make sure datatypes and shapes are consistent so `submission.csv` is always written correctly. These changes are execution-critical and score-positive because they restore use of the intended pretrained base models (instead of the weak fallback path) whenever the `.keras` files exist.'
- What this solution (achieved 0.61323) has done: 'The immediate blocker is the TensorFlow/protobuf crash; I fix it by forcing a protobuf version/implementation combination that is compatible with the TensorFlow build in this Kaggle environment (pure-Python protobuf), and doing it *before* importing TensorFlow. To move your score up toward the target, I keep your stacking approach unchanged but ensure the base `.keras` models actually load by using the same protobuf fix (so you don’t fall back to weak handcrafted features). I also make TFRecord parsing faster/safer without changing semantics by batching/prefetching and avoiding repeated `.numpy()` calls where possible, which helps complete within the time limit while producing the same features. Finally, I keep the exact submission schema and alignment with `sample_submission.csv` intact so a valid `submission.csv` is always written.'
- What this solution (achieved 0.61323) has done: 'We fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment override, which is the root cause of `MessageFactory.GetPrototype` failing in modern TF/protobuf combos. Then we make TFRecord parsing run correctly and faster by parsing examples *before* batching (so we don’t call `parse_single_example` on serialized bytes in Python loops), while keeping the exact same feature extraction (base-model probability concatenation + DecisionTree stacker). This should also increase accuracy toward your target because it restores the intended pretrained-model inference path (instead of erroring/falling back) and avoids silently malformed parsing that can degrade features. Finally, we keep submission alignment with `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.61734) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the known compatibility workaround in this environment. To improve accuracy toward your 0.7886 target (from 0.61323) without changing the core stacking approach, we also switch the stacker from a single `DecisionTreeClassifier` to a `RandomForestClassifier` (still a tree-based model in the same sklearn family, trained on the same concatenated base-model probabilities) which typically boosts generalization on this feature space. Everything else (TFRecord parsing, base-model probability concatenation, submission alignment to `sample_submission.csv`) is kept the same, and we still write a valid `submission.csv`. If the external `.keras` models are unavailable, the script still run end-to-end with the existing fallback features.'
- What this solution (achieved 0.61734) has done: 'We fix the TensorFlow/protobuf import crash by *not* forcing the pure-Python protobuf implementation (that setting is what triggers the `MessageFactory.GetPrototype` failure in modern TF/protobuf combos). Then we add a tiny, score-positive calibration fix: train the RandomForest on out-of-fold (OOF) base-model probabilities via `cross_val_predict` (same model family + same features, but prevents leakage from training on the same data you later evaluate against implicitly), and fit the final model on full data afterward for test inference. Finally, we make TFRecord image-name decoding robust (bytes → str) to avoid silent ID mismatches that reduce training rows and hurt accuracy, while keeping paths and submission format unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_TFRECS_DIR = f"{DATA_ROOT}/train_tfrecords"
TEST_TFRECS_DIR = f"{DATA_ROOT}/test_tfrecords"

feature_description_train = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
feature_description_test = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _find_model_path(preferred_path: str, filename_contains: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    candidates = glob.glob("/kaggle/input/**/*.keras", recursive=True)
    matches = [
        p
        for p in candidates
        if filename_contains.lower() in os.path.basename(p).lower()
    ]
    return matches[0] if matches else None


model1_path = _find_model_path(
    "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras", "densenet"
)
model2_path = _find_model_path(
    "/kaggle/input/abc/keras/default/1/newModel7.keras", "newmodel7"
)
model3_path = _find_model_path(
    "/kaggle/input/mobilenet/keras/default/1/MobileNet (4).keras", "mobilenet"
)

print("Resolved model paths:")
print(" model1:", model1_path)
print(" model2:", model2_path)
print(" model3:", model3_path)

models_loaded = True
try:
    if model1_path is None or model2_path is None or model3_path is None:
        raise FileNotFoundError(
            "One or more .keras model files not found under /kaggle/input."
        )
    model1 = load_model(model1_path, compile=False)
    model2 = load_model(model2_path, compile=False)
    model3 = load_model(model3_path, compile=False)
except Exception as e:
    models_loaded = False
    print(
        "WARNING: Could not load one or more base models; switching to fallback features."
    )
    print("Load error:", repr(e))
    model1 = model2 = model3 = None

NUM_CLASSES = 5


def preprocess_for_keras_models(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    img = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img.numpy()


def predict_concat_probs(image_np: np.ndarray) -> np.ndarray:
    if models_loaded:
        x = preprocess_for_keras_models(image_np)
        p1 = model1.predict(x, verbose=0)[0]
        p2 = model2.predict(x, verbose=0)[0]
        p3 = model3.predict(x, verbose=0)[0]
        return np.concatenate([p1, p2, p3], axis=0).astype(np.float32)

    img = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    if img.shape.rank == 2:
        img = tf.stack([img, img, img], axis=-1)
    if img.shape.rank == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    means = tf.reduce_mean(img, axis=[0, 1]).numpy()  # (3,)
    stds = tf.math.reduce_std(img, axis=[0, 1]).numpy()  # (3,)
    feats = np.concatenate([means, stds, means, stds, means], axis=0)  # 15 dims
    return feats.astype(np.float32)


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, feature_description_train)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    name = ex["image_name"]
    target = ex["target"]
    return img, name, target


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, feature_description_test)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    name = ex["image_name"]
    return img, name


def _to_py_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if hasattr(x, "dtype") and x.dtype.kind in ("S", "O"):
        try:
            b = x.tobytes()
            return b.decode("utf-8")
        except Exception:
            return str(x)
    return str(x)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_labels_by_id = dict(
    zip(train_df["image_id"].astype(str).values, train_df["label"].values)
)

train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFRECS_DIR, f)
        for f in os.listdir(TRAIN_TFRECS_DIR)
        if f.endswith(".tfrec")
    ]
)
if len(train_tfrecs) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {TRAIN_TFRECS_DIR}")

options = tf.data.Options()
options.experimental_deterministic = True

X_train = []
y_train = []

BATCH = 32

for tfrec_path in train_tfrecs:
    ds = tf.data.TFRecordDataset(tfrec_path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for img_batch, name_batch, target_batch in ds:
        name_batch = name_batch.numpy()
        target_batch = target_batch.numpy().astype(np.int64)

        img_batch_np = img_batch.numpy()  # uint8
        for img_np, image_name_raw, _y in zip(img_batch_np, name_batch, target_batch):
            image_name = _to_py_str(image_name_raw)
            if image_name not in train_labels_by_id:
                continue
            feats = predict_concat_probs(img_np)
            X_train.append(feats)
            y_train.append(int(train_labels_by_id[image_name]))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

if X_train.shape[0] == 0:
    raise RuntimeError("No training records were parsed; cannot train tree model.")
if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)

if X_train.shape[0] != len(train_df):
    print(
        f"Warning: parsed {X_train.shape[0]} training examples, train.csv has {len(train_df)} rows."
    )
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("Models loaded:", models_loaded)



## === cell 3
stacker = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=SEED,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_proba = cross_val_predict(
    stacker,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)

stacker.fit(X_train, y_train)

oof_pred = np.argmax(oof_proba, axis=1)
oof_acc = (oof_pred == y_train).mean()
print(f"OOF accuracy (stacker on base features): {oof_acc:.5f}")



## === cell 4
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFRECS_DIR, f)
        for f in os.listdir(TEST_TFRECS_DIR)
        if f.endswith(".tfrec")
    ]
)
if len(test_tfrecs) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {TEST_TFRECS_DIR}")

image_ids = []
X_test = []

BATCH = 32
for tfrec_path in test_tfrecs:
    ds = tf.data.TFRecordDataset(tfrec_path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for img_batch, name_batch in ds:
        name_batch = name_batch.numpy()
        img_batch_np = img_batch.numpy()
        for img_np, image_name_raw in zip(img_batch_np, name_batch):
            image_name = _to_py_str(image_name_raw)
            image_ids.append(image_name)
            X_test.append(predict_concat_probs(img_np))

X_test = np.asarray(X_test, dtype=np.float32)
if X_test.shape[0] == 0:
    raise RuntimeError("No test records were parsed; cannot create submission.")
if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

prediction = stacker.predict(X_test).astype(int)
print("X_test shape:", X_test.shape, "pred shape:", prediction.shape)



## === cell 5
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})
pred_df["image_id"] = pred_df["image_id"].astype(str)
pred_df["label"] = pred_df["label"].astype(int)
pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    fallback = int(pd.Series(y_train).value_counts().idxmax())
    submission["label"] = submission["label"].fillna(fallback).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission = submission[["image_id", "label"]]
assert len(submission) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
print("Models loaded:", models_loaded)

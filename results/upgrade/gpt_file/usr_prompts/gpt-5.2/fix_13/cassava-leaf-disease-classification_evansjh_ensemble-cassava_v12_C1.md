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

0.8907524932003626

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I first remove `tensorflow_hub` (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment and isn’t used by your inference pipeline). Next, I make model loading robust by checking which of the provided model paths actually exist and only loading those; if none exist, the script fall back to a standard Keras application model so a valid `submission.csv` is still produced end-to-end. I also fix the `models` undefined cascade by ensuring `models` is always defined (even in fallback) and ensure predictions are generated for exactly the `image_id`s in `sample_submission.csv` (stable ordering and correct set). These changes preserve your core ensemble voting logic and only adjust I/O and robustness so it runs and yields a proper submission file.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by repeated JPEG decode/resize work and by expensive “fallback training” when model files are missing. I keep the same ensemble logic and voting, but make inference faster by (1) batching and caching decoded images once, (2) resizing once per unique input size, and (3) using a single multi-output Keras wrapper per size so each size is only iterated once through the dataset even if multiple models share that size. I also hard-fail if no external models are found (instead of training), because training is the primary 10-minute+ risk on CPU and is not required for the intended submission flow. These changes are equivalent in semantics (same models, same preprocessing, same argmax/mean-confidence tie-break), but remove redundant passes and Python overhead.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("TF version:", tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)  # keep numerics stable; avoid XLA variability
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
model_path_7 = "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5"

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(test_image_dir), f"Missing test image dir: {test_image_dir}"
assert os.path.exists(sample), f"Missing sample submission: {sample}"



## === cell 2
sample_csv = pd.read_csv(sample)
if not {"image_id", "label"}.issubset(sample_csv.columns):
    raise ValueError(
        f"Unexpected sample submission columns: {sample_csv.columns.tolist()}"
    )

print("Sample rows:", len(sample_csv))
print(sample_csv.head())



## === cell 3
models_info = [
    (model_path_6, (512, 512)),
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_7, (512, 512)),
]

existing_models_info = [(p, sz) for (p, sz) in models_info if os.path.exists(p)]
missing = [(p, sz) for (p, sz) in models_info if not os.path.exists(p)]

print(f"Found {len(existing_models_info)} model files; missing {len(missing)}.")
if missing:
    print("Missing model paths (will be skipped):")
    for p, sz in missing:
        print(" -", p)

models = []
for path, input_size in existing_models_info:
    try:
        m = load_model(path, compile=False)
        models.append((m, input_size))
        print(f"Loaded: {path} with input {input_size}")
    except Exception as e:
        print(f"Failed to load {path}: {type(e).__name__}: {e}")




## === cell 4
def build_baseline_cnn(input_shape=(224, 224, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
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


def make_train_dataset(df, image_dir, img_size=(224, 224), batch_size=32, shuffle=True):
    paths = [os.path.join(image_dir, x) for x in df["image_id"].astype(str).tolist()]
    labels = df["label"].astype(np.int32).to_numpy()

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32)
        return img, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(df), 4096), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if len(models) == 0:
    raise RuntimeError(
        "No external models were loaded from the provided /kaggle/input paths. "
        "Fallback training is disabled to avoid exceeding the 600s timeout."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3932447880.py in <cell line: 0>()
     49 # This preserves the intended inference-only core logic and avoids a guaranteed >600s CPU timeout.
     50 if len(models) == 0:
---> 51     raise RuntimeError(
     52         "No external models were loaded from the provided /kaggle/input paths. "
     53         "Fallback training is disabled to avoid exceeding the 600s timeout."

RuntimeError: No external models were loaded from the provided /kaggle/input paths. Fallback training is disabled to avoid exceeding the 600s timeout.

## === cell 5
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

test_ids = sample_csv["image_id"].astype(str).tolist()
test_paths = [os.path.join(test_image_dir, img_id) for img_id in test_ids]

present_idx = np.arange(len(test_paths), dtype=np.int32)
present_paths = test_paths
n_present = len(present_paths)
n_total = len(test_paths)




## === cell 6
def make_decoded_test_dataset(paths, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    path_ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)

    def _decode(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = path_ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE).with_options(options)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def resize_from_decoded_dataset(decoded_batched_ds, img_size):
    @tf.function(reduce_retracing=True)
    def _resize(batch_imgs):
        return tf.image.resize(batch_imgs, img_size, method="bilinear")

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    return decoded_batched_ds.map(
        _resize, num_parallel_calls=tf.data.AUTOTUNE
    ).with_options(options)


from collections import defaultdict

models_by_size = defaultdict(list)
for m, sz in models:
    models_by_size[tuple(sz)].append(m)

all_model_pred_classes = []  # list of (n_present,) int32
all_model_pred_conf = []  # list of (n_present,) float32

BATCH_SIZE = 64
decoded_ds = make_decoded_test_dataset(present_paths, batch_size=BATCH_SIZE)

for input_size, ms in models_by_size.items():
    ds = resize_from_decoded_dataset(decoded_ds, img_size=input_size)

    for m in ms:
        try:
            m.make_predict_function()
        except Exception:
            pass

    if len(ms) == 1:
        preds_list = [ms[0].predict(ds, verbose=0)]
    else:
        inp = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
        outs = [m(inp, training=False) for m in ms]
        multi = tf.keras.Model(inp, outs)
        try:
            multi.make_predict_function()
        except Exception:
            pass
        preds_list = multi.predict(ds, verbose=0)
        if not isinstance(preds_list, (list, tuple)):
            preds_list = [preds_list]

    for preds in preds_list:
        pred_cls = np.argmax(preds, axis=1).astype(np.int32)
        pred_conf = preds[np.arange(preds.shape[0]), pred_cls].astype(np.float32)
        all_model_pred_classes.append(pred_cls)
        all_model_pred_conf.append(pred_conf)

pred_classes = (
    np.stack(all_model_pred_classes, axis=0)
    if all_model_pred_classes
    else np.empty((0, n_present), dtype=np.int32)
)
pred_confs = (
    np.stack(all_model_pred_conf, axis=0)
    if all_model_pred_conf
    else np.empty((0, n_present), dtype=np.float32)
)

final_present_labels = np.empty((n_present,), dtype=np.int32)
n_models = pred_classes.shape[0]

if n_models == 0:
    final_present_labels.fill(4)
else:
    counts = np.zeros((5, n_present), dtype=np.int16)
    for c in range(5):
        counts[c] = (pred_classes == c).sum(axis=0, dtype=np.int16)

    max_count = counts.max(axis=0)
    tied = counts == max_count[None, :]
    n_tied = tied.sum(axis=0)

    first_winner = np.argmax(tied, axis=0).astype(np.int32)

    mean_conf = np.full((5, n_present), -np.inf, dtype=np.float32)
    for c in range(5):
        mask = pred_classes == c  # (n_models, n_present)
        sum_conf = (pred_confs * mask).sum(axis=0, dtype=np.float32)
        denom = mask.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mc = sum_conf / denom
        mean_conf[c] = np.where(denom > 0, mc, -np.inf)

    best_by_mean = np.argmax(np.where(tied, mean_conf, -np.inf), axis=0).astype(
        np.int32
    )

    final_present_labels = np.where(n_tied == 1, first_winner, best_by_mean).astype(
        np.int32
    )

final_labels = final_present_labels

submission_df = pd.DataFrame({"image_id": test_ids, "label": final_labels.astype(int)})

print(f"Predicted {n_present}/{n_total} test images with {n_models} model(s).")



## === cell 7
submission_df = submission_df[["image_id", "label"]].copy()
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(4).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission_df.head())
print(
    "Rows:",
    len(submission_df),
    "Unique image_ids:",
    submission_df["image_id"].nunique(),
)



## === cell 8
submission_df

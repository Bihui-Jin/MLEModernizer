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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

print("Python:", __import__("sys").version)
print("TensorFlow:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
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

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(test_image_dir), f"Missing test image dir: {test_image_dir}"
assert os.path.exists(sample), f"Missing sample submission: {sample}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.exists(train_image_dir), f"Missing train image dir: {train_image_dir}"




## === cell 2
sample_csv = pd.read_csv(sample)
if not {"image_id", "label"}.issubset(sample_csv.columns):
    raise ValueError(
        f"sample_submission.csv must contain image_id,label. Found: {sample_csv.columns.tolist()}"
    )
print("Sample rows:", len(sample_csv))
sample_csv.head()




## === cell 3
def try_load_models(models_info):
    loaded = []
    for path, input_size in models_info:
        if os.path.exists(path):
            try:
                m = load_model(path, compile=False)
                loaded.append((m, input_size, path))
            except Exception as e:
                print(f"Could not load model at {path}: {type(e).__name__}: {e}")
        else:
            print(f"Model path not found, skipping: {path}")
    return loaded


models_info = [
    (model_path_1, (550, 550)),
    (model_path_4, (550, 550)),
]

loaded_models = try_load_models(models_info)
print("Loaded models:", len(loaded_models))




## === cell 4
from tensorflow.keras import layers, models, optimizers


def build_fallback_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_dataset_from_df(
    df, image_dir, image_size=(224, 224), batch_size=32, shuffle=True
):
    filepaths = [os.path.join(image_dir, fn) for fn in df["image_id"].tolist()]
    labels = df["label"].astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load(path, y):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        return img, y

    if shuffle:
        ds = ds.shuffle(min(len(df), 4096), seed=42, reshuffle_each_iteration=True)

    opts = tf.data.Options()
    try:
        opts.deterministic = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = ds.with_options(opts)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


fallback_models = []
if len(loaded_models) == 0:
    train_df = pd.read_csv(train_csv_path)
    if not {"image_id", "label"}.issubset(train_df.columns):
        raise ValueError(
            f"train.csv must contain image_id,label. Found: {train_df.columns.tolist()}"
        )

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(0.1 * len(train_df))
    val_df = train_df.iloc[:val_size].copy()
    tr_df = train_df.iloc[val_size:].copy()

    img_size = (224, 224)
    batch_size = 32
    tr_ds = make_dataset_from_df(
        tr_df, train_image_dir, image_size=img_size, batch_size=batch_size, shuffle=True
    )
    val_ds = make_dataset_from_df(
        val_df,
        train_image_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    fallback_model = build_fallback_model(
        input_shape=(img_size[0], img_size[1], 3), num_classes=5
    )

    fallback_model.fit(tr_ds, validation_data=val_ds, epochs=3, verbose=2)

    fallback_models = [(fallback_model, img_size, "fallback_trained_cnn")]
    print("Using fallback trained model.")
else:
    print("Using provided pretrained model(s).")




## === cell 5
models = [(m, input_size) for (m, input_size, _path) in loaded_models]
if len(models) == 0:
    models = [(m, input_size) for (m, input_size, _name) in fallback_models]

if len(models) == 0:
    raise RuntimeError("No models available (neither pretrained nor fallback).")

print(
    "Inference will use", len(models), "model(s). Input sizes:", [s for _, s in models]
)




## === cell 6
def make_test_base_dataset(filepaths):
    ds = tf.data.Dataset.from_tensor_slices(filepaths)

    def _read_decode(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    opts = tf.data.Options()
    try:
        opts.deterministic = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = ds.with_options(opts)
    ds = ds.map(_read_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset_from_base(base_ds, image_size, batch_size=32):
    def _resize(img):
        img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
        return img

    ds = base_ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def majority_vote_2models(a, b):
    return a.astype(np.int64, copy=False)


image_ids = sample_csv["image_id"].tolist()
filepaths = [os.path.join(test_image_dir, iid) for iid in image_ids]

exists_mask = np.fromiter(
    (os.path.exists(p) for p in filepaths), dtype=bool, count=len(filepaths)
)

missing = int((~exists_mask).sum())
print("Missing images:", missing)

pred_labels = np.zeros(len(filepaths), dtype=np.int64)
valid_idx = np.flatnonzero(exists_mask)
valid_paths = [filepaths[i] for i in valid_idx]

if len(valid_paths) == 0:
    pred_labels = pred_labels.tolist()
else:
    batch_size = 32

    base_ds = make_test_base_dataset(valid_paths)

    per_model_preds = []
    for model, input_size in models:
        ds_for_model = make_test_dataset_from_base(
            base_ds, image_size=tuple(map(int, input_size)), batch_size=batch_size
        )
        probs = model.predict(ds_for_model, verbose=0)
        per_model_preds.append(np.argmax(probs, axis=1).astype(np.int64))

    if len(per_model_preds) == 1:
        voted = per_model_preds[0]
    elif len(per_model_preds) == 2:
        voted = majority_vote_2models(per_model_preds[0], per_model_preds[1])
    else:
        stacked = np.stack(per_model_preds, axis=0)  # [M, N]
        counts = np.zeros((stacked.shape[1], 5), dtype=np.int16)
        ar = np.arange(stacked.shape[1])
        for m in range(stacked.shape[0]):
            counts[ar, stacked[m]] += 1
        voted = counts.argmax(axis=1).astype(np.int64)

    pred_labels[valid_idx] = voted
    pred_labels = pred_labels.tolist()




## === cell 7
submission_df = pd.DataFrame(
    {
        "image_id": sample_csv["image_id"].values,
        "label": np.array(pred_labels, dtype=np.int64),
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved at: {submission_path}")
submission_df.head()




## === cell 8
assert os.path.exists(submission_path), "submission.csv was not created."
check = pd.read_csv(submission_path)
assert check.shape[0] == sample_csv.shape[0], "Submission row count mismatch."
assert list(check.columns) == [
    "image_id",
    "label",
], f"Wrong columns: {check.columns.tolist()}"
assert check["label"].between(0, 4).all(), "Labels must be integers 0..4."
check.tail()

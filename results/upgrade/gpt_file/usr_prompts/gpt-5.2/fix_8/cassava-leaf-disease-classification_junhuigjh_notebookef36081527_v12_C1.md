# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.804472650347537

# 6. Current score

0.7216

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16181) has done: 'The pipeline fails before producing a submission because (1) importing TensorFlow triggers a known protobuf incompatibility in some Kaggle images, and (2) the referenced external DenseNet model file path doesn’t exist, so `load_model()` raises “File not found,” which then cascades into `NameError` later. I (a) set a safe protobuf implementation env var before importing TensorFlow to prevent the `MessageFactory.GetPrototype` crash, (b) make model loading robust by falling back to a lightweight built-in Keras application model when the external file is missing (so the code still runs end-to-end), and (c) ensure we always produce a correctly formatted `submission.csv` aligned to `sample_submission.csv`. This keeps the core approach intact (single-image preprocessing + `model.predict` + argmax classification) while guaranteeing a valid CSV output.'
- What this solution (achieved 0.0639) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by importing `google.protobuf` early so the env var takes effect in this Kaggle image. Then I keep your same inference pipeline but ensure the DenseNet fallback uses the correct preprocessing (`tf.keras.applications.densenet.preprocess_input`) so predictions aren’t essentially random, which should move accuracy substantially toward your target. Finally, I make TFRecord reading deterministic and robust and ensure the submission is aligned to `sample_submission.csv` and always written to `submission.csv`.'
- What this solution (achieved 0.11958) has done: 'The timeout is dominated by per-image Python overhead: eager execution, decoding each TFRecord example in a Python loop, converting tensors to NumPy, doing PIL resize per image, and calling `model.predict` one image at a time. The core model and preprocessing are preserved, but the input pipeline is rewritten to use `tf.data` batching with parallel decode/parse and vectorized resize + DenseNet preprocess in TensorFlow, then a single batched `model.predict` over the full dataset. This removes thousands of Python→TF roundtrips and eliminates PIL work inside the hot loop while keeping identical semantics (224×224 bilinear resize + `tf.keras.applications.densenet.preprocess_input`). Seeds/paths remain unchanged; determinism is maintained while enabling graph execution for the input pipeline.'
- What this solution (achieved 0.37593) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation early and pinning the protobuf runtime to the compatible major version before TensorFlow is imported. This is an execution blocker; once TensorFlow loads, the rest of your vectorized `tf.data` batched inference pipeline can run unchanged and produce a valid `submission.csv`. I also make the model-loading fallback robust: if the external DenseNet `.keras` file is missing, we still build a DenseNet121-based classifier, but ensure the top Dense layer is initialized deterministically to avoid completely random predictions. Finally, I keep submission alignment to `sample_submission.csv` intact to guarantee correct ordering and no missing labels.'
- What this solution (achieved 0.7216) has done: 'Your current score is far below the target (0.37593 vs 0.80447), and the main reason is that when the external trained model file is missing, the fallback DenseNet121 classifier head is randomly initialized, making predictions largely uninformative. I keep your exact inference pipeline (TFRecord→`tf.data`→DenseNet preprocess→`model.predict`→`argmax`→CSV) but replace the random-head fallback with a tiny training step on `train.csv` + `train_images` (same architecture, same loss semantics) so the produced probabilities become meaningfully class-discriminative. To stay within Kaggle runtime, I train only the top Dense layer with the DenseNet base frozen for 1 epoch using a batched `tf.data` pipeline. The submission writing/merging logic stays the same and still guarantees a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(str(_pb_ver).split(".")[0])
    if _pb_major >= 5:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TFREC_TEST_DIR = os.path.join(DATA_ROOT, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

MODEL_PATH = "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras"

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224), resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = tf.keras.applications.densenet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@tf.function
def second_model_preprocess_tf(image_uint8: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize(
        image_uint8, [224, 224], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image


def parse_example(raw_record: tf.Tensor):
    ex = tf.io.parse_single_example(raw_record, feature_description)
    image = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8
    name = ex["image_name"]  # bytes
    return image, name


def build_test_dataset_from_tfrecs(tfrec_paths, batch_size: int):
    ds = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda img, name: (second_model_preprocess_tf(img), name),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_test_dataset_from_files(image_paths, image_ids, batch_size: int):
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, image_ids))

    def _load_and_preprocess(path, image_id):
        bytes_ = tf.io.read_file(path)
        img = tf.io.decode_jpeg(bytes_, channels=3)  # uint8
        img = second_model_preprocess_tf(img)
        return img, image_id

    ds = path_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_dataset_from_files(
    image_paths, labels, batch_size: int, shuffle: bool = True
):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(image_paths), 4096),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _load_and_preprocess(path, label):
        bytes_ = tf.io.read_file(path)
        img = tf.io.decode_jpeg(bytes_, channels=3)  # uint8
        img = second_model_preprocess_tf(img)
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 1
BATCH_SIZE = 64

if os.path.exists(MODEL_PATH):
    model2 = load_model(MODEL_PATH)
else:
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    outputs = tf.keras.layers.Dense(
        5,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="cls_head",
    )(x)
    model2 = tf.keras.Model(inputs, outputs)

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    if os.path.exists(TRAIN_CSV_PATH) and os.path.isdir(TRAIN_IMG_DIR):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        train_image_ids = train_df["image_id"].astype(str).tolist()
        train_labels = train_df["label"].astype(np.int32).values
        train_image_paths = [os.path.join(TRAIN_IMG_DIR, x) for x in train_image_ids]

        train_ds = build_train_dataset_from_files(
            train_image_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True
        )

        model2.fit(train_ds, epochs=1, verbose=0)

tfrecs = []
if os.path.isdir(TFREC_TEST_DIR):
    tfrecs = sorted(
        [
            os.path.join(TFREC_TEST_DIR, f)
            for f in os.listdir(TFREC_TEST_DIR)
            if f.endswith(".tfrec")
        ]
    )

image_ids = []
prediction = []

if len(tfrecs) > 0:
    ds = build_test_dataset_from_tfrecs(tfrecs, batch_size=BATCH_SIZE)

    for _, name_batch in ds:
        image_ids.extend([n.decode("utf-8") for n in name_batch.numpy().tolist()])

    probs = model2.predict(
        ds.map(lambda img, name: img, num_parallel_calls=tf.data.AUTOTUNE), verbose=0
    )
    prediction = probs.argmax(axis=1).astype(int).tolist()
else:
    test_img_dir = os.path.join(DATA_ROOT, "test_images")
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    image_ids = sample_df["image_id"].tolist()
    image_paths = [os.path.join(test_img_dir, x) for x in image_ids]

    ds = build_test_dataset_from_files(image_paths, image_ids, batch_size=BATCH_SIZE)
    probs = model2.predict(
        ds.map(lambda img, name: img, num_parallel_calls=tf.data.AUTOTUNE), verbose=0
    )
    prediction = probs.argmax(axis=1).astype(int).tolist()




## === cell 2
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_df[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    if submission["label"].notna().any():
        fill_val = int(submission["label"].dropna().mode().iloc[0])
    else:
        fill_val = 0
    submission["label"] = submission["label"].fillna(fill_val)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Missing labels after merge:", int(submission["label"].isna().sum()))

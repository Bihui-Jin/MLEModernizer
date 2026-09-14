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

# 5. Target score

0.8750377757630704

# 6. Current score

0.11248

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.32773) has done: 'I fix the pipeline so it runs end-to-end and writes a valid `submission.csv`. The main blockers are (1) loading a Keras `.h5` that fails due to protobuf/TensorFlow/Keras incompatibilities in the environment, and (2) imports that assume unavailable Albumentations symbols, causing the notebook to stop before `pd`, `ImageDataGenerator`, etc. are defined. To keep core logic intact while ensuring a submission is produced, I (a) switch to `tf.keras`-native loading with a safe fallback model if the external weight file can’t be loaded, (b) remove the broken Albumentations imports and keep augmentation minimal/optional, and (c) ensure inference resizes and normalizes images to the expected input shape and that submission rows align with `sample_submission.csv`.'
- What this solution (achieved 0.11697) has done: 'I fix the TensorFlow/protobuf crash causing `MessageFactory.GetPrototype` by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which resolves this common Kaggle environment incompatibility. I also make the fallback model construction correct for transfer learning by replacing the current “add Dense on top of 1000-way logits” bug with a proper `include_top=False` backbone + pooling + Dense(5), which should substantially improve accuracy versus the current broken head. Finally, I keep the rest of the pipeline intact but ensure preprocessing matches InceptionResNetV2 and that submission ordering matches `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.05568) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf runtime *and* downgrading the protobuf API usage inside the process before importing TensorFlow. Then I make training actually happen (your current script builds generators but never calls `model.fit`), keeping the same model core (either loaded `.h5` or the same InceptionResNetV2 fallback head) so predictions aren’t random. Finally, I ensure preprocessing is consistent between training and inference by using the same InceptionResNetV2 `preprocess_input` in the `ImageDataGenerator`, and keep the submission ordering aligned to `sample_submission.csv` while writing a valid `submission.csv`.'
- What this solution (achieved 0.17975) has done: 'The main timeout drivers here are (1) forcing the pure-Python protobuf implementation (very slow TFRecord/proto parsing inside TF/Keras), and (2) PIL-per-image loading + preprocessing for the whole test set in Python loops. I switch protobuf back to the default compiled implementation (equivalent semantics, much faster) and replace the per-image PIL prediction loop with a `tf.data` pipeline that decodes/resizes/preprocesses images in parallel and feeds `model.predict` in batches. I also enable TF dataset prefetching/parallel mapping and tune Keras generator worker settings to reduce input bottlenecks during the single training epoch, without changing the model, loss, metrics, or training procedure.'
- What this solution (achieved 0.23468) has done: 'The main timeout comes from building two training pipelines and accidentally overriding the TFRecord pipeline with the slower per-JPEG pipeline, plus iterating the full TFRecord set for both train and validation. I fix the dataset selection so TFRecords are actually used when present, and make validation read only the same split as `val_df` (via a deterministic filter) while keeping the same decode/resize/preprocess and one-hot labels. I also remove unnecessary Python-side path dataset creation when TFRecords exist, and add small tf.data optimizations that are semantically equivalent (better interleave cycle length, fixed options, and avoiding redundant shuffles). Model, loss, optimizer, epochs, and evaluation semantics remain unchanged.'
- What this solution (achieved 0.11248) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the known compatibility workaround in this environment. Then I fix the TFRecord parsing error by removing the non-existent `image_id` feature from the schema and by splitting TFRecords into train/val using a deterministic per-record hash (so we don’t need `image_id` at all). Finally, I keep the same model/training core (same backbone, loss, optimizer, epochs) but ensure label depth is always 5 (competition classes) so the head/one-hot encoding can’t silently mismatch.'

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

MODEL_PATH = "../input/mdpa56/initialweightInceptionResnet4.h5"

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def _build_fallback_model(num_classes: int = 5):
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=(299, 299, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model


model = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        load_error = e
        model = None

if model is None:
    print("WARNING: Could not load provided .h5 model. Using fallback model instead.")
    if load_error is not None:
        print("Load error:", repr(load_error))
    model = _build_fallback_model(num_classes=5)

try:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    H = input_shape[1] or 299
    W = input_shape[2] or 299
except Exception:
    H, W = 299, 299

print("Model input size:", (H, W))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import numpy as np
import pandas as pd
from PIL import Image  # kept (may be unused now but preserved)

from sklearn.model_selection import train_test_split
import json
import glob

print("Imports OK")



## === cell 3
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
print(train.head())



## === cell 4
train["path"] = train["image_id"].apply(
    lambda x: os.path.join(INPUT_DIR, "train_images", str(x))
)
train_df, val_df = train_test_split(
    train, test_size=0.05, random_state=100, stratify=train["label"].values
)
print("Train/val sizes:", len(train_df), len(val_df))



## === cell 5
tf.keras.utils.set_random_seed(123)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

batch_size = 12

train_df = train_df.copy()
val_df = val_df.copy()
train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)

num_classes = 5

train_paths = train_df["path"].to_numpy()
val_paths = val_df["path"].to_numpy()
train_labels_idx = train_df["label"].to_numpy(dtype=np.int32)
val_labels_idx = val_df["label"].to_numpy(dtype=np.int32)


@tf.function
def _decode_resize_preprocess(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (H, W), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img


AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(INPUT_DIR, "test_tfrecords")
train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

print("Found train tfrecs:", len(train_tfrecs), "test tfrecs:", len(test_tfrecs))

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

VAL_FRAC = 0.05
VAL_MOD = 1000
VAL_THRESHOLD = int(VAL_FRAC * VAL_MOD)


def _make_trainval_ds_from_tfrecords(tfrecs, training: bool):
    def _parse(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_description)
        img = _decode_resize_preprocess(ex["image"])
        y = tf.one_hot(
            tf.cast(ex["label"], tf.int32), depth=num_classes, dtype=tf.float32
        )
        return img, y, example_proto

    ds_files = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(tfrecs, dtype=tf.string)
    )
    if training:
        ds_files = ds_files.shuffle(
            len(tfrecs), seed=123, reshuffle_each_iteration=True
        )

    cycle_len = min(max(len(tfrecs), 1), 8)
    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=cycle_len,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    if training:
        ds = ds.shuffle(4096, seed=123, reshuffle_each_iteration=True)

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE, deterministic=True)

    def _is_val(img, y, raw):
        h = tf.strings.to_hash_bucket_fast(raw, VAL_MOD)
        return h < VAL_THRESHOLD

    if training:
        ds = ds.filter(lambda img, y, raw: tf.logical_not(_is_val(img, y, raw)))
    else:
        ds = ds.filter(lambda img, y, raw: _is_val(img, y, raw))

    ds = ds.map(
        lambda img, y, raw: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_ds_from_paths(paths_np, labels_idx_np, training: bool):
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
    labels_idx = tf.convert_to_tensor(labels_idx_np, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_idx))

    if training:
        shuffle_buf = int(min(len(paths_np), 4096))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=123, reshuffle_each_iteration=True
        )

    @tf.function
    def _load_example(path, y_idx):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        y = tf.one_hot(y_idx, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if len(train_tfrecs) > 0:
    train_ds = _make_trainval_ds_from_tfrecords(train_tfrecs, training=True)
    val_ds = _make_trainval_ds_from_tfrecords(train_tfrecs, training=False)
else:
    train_ds = _make_ds_from_paths(train_paths, train_labels_idx, training=True)
    val_ds = _make_ds_from_paths(val_paths, val_labels_idx, training=False)

print("tf.data train/val pipelines ready:", num_classes, "classes")



## === cell 6
os.makedirs("./checkpoints", exist_ok=True)
print("checkpoints dir ready")



## === cell 7
if getattr(model, "optimizer", None) is None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    try:
        model.compile(
            optimizer=model.optimizer,
            loss=(
                model.loss
                if getattr(model, "loss", None) is not None
                else "categorical_crossentropy"
            ),
            metrics=["accuracy"],
        )
    except Exception:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

ckpt_path = "./checkpoints/best.weights.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

EPOCHS = 1

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    try:
        model.load_weights(ckpt_path)
        print("Loaded best weights from:", ckpt_path)
    except Exception as e:
        print("WARNING: could not load checkpoint weights:", repr(e))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/970208863.py in <cell line: 0>()
     36 EPOCHS = 1
     37 
---> 38 history = model.fit(
     39     train_ds,
     40     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Filter::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_121105]

## === cell 8
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

TEST_DIR = os.path.join(INPUT_DIR, "test_images")
print("Num test images in sample:", len(test_image_ids))
print("Example:", test_image_ids[:3])




## === cell 9
def _build_test_dataset(image_ids, batch_size: int):
    paths_np = np.fromiter(
        (os.path.join(TEST_DIR, x) for x in image_ids),
        dtype=object,
        count=len(image_ids),
    )
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)

    @tf.function
    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


bs = 32
test_ds = _build_test_dataset(test_image_ids, batch_size=bs)

probs = model.predict(test_ds, verbose=0)
predictions = probs.argmax(axis=1).astype(int).tolist()

print("Predictions generated:", len(predictions))

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

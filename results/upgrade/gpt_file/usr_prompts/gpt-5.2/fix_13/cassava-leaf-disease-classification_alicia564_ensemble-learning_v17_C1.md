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

0.8939256572982774

# 6. Current score

0.12967

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12967) has done: 'The timeout is dominated by slow Python-side image loading/augmentation: you build both `ImageDataGenerator` generators and `tf.data` pipelines (duplicated work), and you force the pure-Python protobuf implementation which makes TensorFlow I/O slower. I keep the exact same model/training/prediction semantics, but remove the unused `tf.data` pipeline, switch protobuf back to the fast C++ default, and make the generators use multi-worker prefetching to overlap CPU decoding/augmentation with GPU training. For test-time inference, I replace the Python loop using `load_img` with a `tf.data` input pipeline (same preprocessing) and predict in one streamed pass, plus a vectorized label mapping (no `np.vectorize`). These changes are performance-only and keep the same architecture, loss, and training loop behavior.'
- What this solution (achieved 0.12967) has done: 'The timeout is dominated by slow input pipelines: (1) `flow_from_dataframe` uses a Python/CPU image loader/augmenter that becomes the bottleneck for EfficientNet training, and (2) you forced protobuf to the pure-Python implementation which slows TensorFlow graph parsing/execution overhead. To keep the same model, loss, and training semantics while speeding up, I switch training/validation to `tf.data` reading from the provided TFRecords (same images/labels) and implement the same augmentations with TensorFlow ops, enabling parallel map + prefetch for full pipeline throughput. I also remove the protobuf “python” override (keeping determinism seeds) so TF can use the faster default implementation. Prediction already uses `tf.data`; I only add minor graph-friendly tweaks (AUTOTUNE, non-Python loops) while keeping identical output formatting and label mapping.'
- What this solution (achieved 0.12967) has done: 'The timeout is most likely coming from the expensive tf.data split logic that enumerates and filters the full TFRecord stream twice (train/valid), plus extra pipeline overhead from repeated `with_options()` and non-fused maps. I keep the exact same model, loss, augmentation, and epoch settings, but replace the train/valid split with a single-pass TFRecord stream that routes examples into train or valid via a hash-table lookup on `image_id` (built once from `train.csv`). I also fuse parsing + one-hot into one map, avoid redundant dataset enumerations, and add `.cache()` on the decoded+resized (pre-augmentation) training stream (still applying augmentation each epoch) to cut repeated JPEG decode/resize cost without changing semantics. Finally, I keep determinism/seeds and keep I/O paths unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import math

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])
train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecord_files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
if not train_tfrecord_files:
    raise FileNotFoundError(f"No TFRecord files found in {TRAIN_TFREC_DIR}")

label_int_to_disease = {int(k): str(v) for k, v in label_to_disease.to_dict().items()}
disease_names = [label_int_to_disease[i] for i in sorted(label_int_to_disease.keys())]
NUM_CLASSES = len(disease_names)
print("NUM_CLASSES:", NUM_CLASSES)

class_indices = {d: i for i, d in enumerate(disease_names)}
print("Class indices (disease->idx):", class_indices)

disease_to_label_str = (
    train_csv[["disease", "label"]]
    .drop_duplicates()
    .set_index("disease")["label"]
    .to_dict()
)
idx_to_disease = {v: k for k, v in class_indices.items()}
idx_to_label_int = {
    idx: int(disease_to_label_str[disease]) for idx, disease in idx_to_disease.items()
}

idx_to_label_arr = np.empty((NUM_CLASSES,), dtype=np.int64)
for i in range(NUM_CLASSES):
    idx_to_label_arr[i] = idx_to_label_int[i]
print("Example idx_to_label_int mapping:", dict(list(idx_to_label_int.items())[:5]))

train_targets_np = train["label"].astype(int).to_numpy()
valid_targets_np = valid["label"].astype(int).to_numpy()

train_target_counts = np.bincount(train_targets_np, minlength=NUM_CLASSES).astype(
    np.int64
)
valid_target_counts = np.bincount(valid_targets_np, minlength=NUM_CLASSES).astype(
    np.int64
)
print(
    "Train target counts:", train_target_counts, "sum:", int(train_target_counts.sum())
)
print(
    "Valid target counts:", valid_target_counts, "sum:", int(valid_target_counts.sum())
)

FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}

IMG_SIZE = 224
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32


def _parse_example_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = preprocess_input(img)
    target = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_id"]
    return img, target, image_id


data_augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=45 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=SEED,
        ),
    ],
    name="data_augmenter",
)


def _random_shear(img):
    shear = tf.random.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32, seed=SEED)
    transform = tf.stack([1.0, shear, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
    transform = tf.expand_dims(transform, 0)
    img4 = tf.expand_dims(img, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


def _augment_full(img, target):
    img = data_augmenter(tf.expand_dims(img, 0), training=True)
    img = tf.squeeze(img, 0)
    img = _random_shear(img)
    return img, target


def make_datasets():
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    train_ids = tf.constant(train["image_id"].astype(str).to_numpy())
    train_ids_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=train_ids,
            values=tf.ones([tf.shape(train_ids)[0]], dtype=tf.int32),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    raw = tf.data.TFRecordDataset(
        train_tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(options)

    parsed = raw.map(_parse_example_with_id, num_parallel_calls=tf.data.AUTOTUNE)

    def _is_train(img, target, image_id):
        return tf.equal(train_ids_table.lookup(image_id), 1)

    def _drop_id(img, target, image_id):
        return img, target

    train_ds = parsed.filter(_is_train).map(
        _drop_id, num_parallel_calls=tf.data.AUTOTUNE
    )
    valid_ds = parsed.filter(
        lambda img, target, image_id: tf.logical_not(_is_train(img, target, image_id))
    ).map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache()

    def _to_onehot(img, target):
        return img, tf.one_hot(target, depth=NUM_CLASSES, dtype=tf.float32)

    train_ds = train_ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_augment_full, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(_to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(_to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache()
    valid_ds = valid_ds.batch(BATCH_SIZE_VALID, drop_remainder=False)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


train_ds, valid_ds = make_datasets()

STEPS_PER_EPOCH = int(math.ceil(len(train) / BATCH_SIZE_TRAIN))
VALIDATION_STEPS = int(math.ceil(len(valid) / BATCH_SIZE_VALID))
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3))
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3145408453.py in <cell line: 0>()
      1 EPOCHS = 10
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=valid_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Map::Shuffle::MemoryCacheImpl::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_49762]

## === cell 5
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_paths = [
    os.path.join(test_image_dir, img_id) for img_id in sample_sub["image_id"].tolist()
]


def _load_and_preprocess_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    img = preprocess_input(img)
    return img


BATCH_SIZE = 64

options = tf.data.Options()
options.deterministic = True
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths)).with_options(
    options
)
test_ds = test_ds.map(_load_and_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
pred_class_idx = np.argmax(probs, axis=1).astype(np.int64)

pred_label_ids = idx_to_label_arr[pred_class_idx]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_label_ids.astype(int)}
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
assert out_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)

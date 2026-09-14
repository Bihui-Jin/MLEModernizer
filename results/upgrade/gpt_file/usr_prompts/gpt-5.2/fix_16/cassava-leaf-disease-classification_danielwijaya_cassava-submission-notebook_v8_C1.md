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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8898458748866727

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11697) has done: 'I remove the failing dependencies (`kaggle_datasets`, external EfficientNet/keras_applications paths, and missing pretrained .h5 loads) that prevent the notebook from running under your current TensorFlow/Protobuf setup. Then I keep your core inference logic (TFRecord input pipeline + simple averaged ensemble + argmax) but build two equivalent EfficientNet models directly from `tf.keras.applications` and run them with ImageNet weights, which is a minimal, legitimate replacement for the missing models that yields a reasonable accuracy baseline. Finally, I fix the TFRecord decode bug (JPEG decode then forced reshape) by resizing properly, and ensure the generated `submission.csv` has the exact required columns and row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype` from an incompatible protobuf runtime) by forcing the pure-Python protobuf implementation before TensorFlow is imported, which is the minimal change to unblock execution in this environment. Then I keep your exact inference core (TFRecords → EfficientNetB6/B4 with ImageNet weights → average probs → argmax) but correct a logic issue that can silently scramble predictions: your image and id pipelines are iterated separately while `experimental_deterministic=False` is set in `load_dataset` (and only partially overridden), which can desynchronize IDs from images. I make the test dataset strictly deterministic/ordered end-to-end and derive `test_ids` and `predictions` from the same iteration order to align outputs with `sample_submission.csv`, which should move accuracy sharply upward toward the target without changing model architecture or training. Finally, I keep the submission format exactly as required and ensure `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I fix the dataset construction error by removing the unsupported `deterministic=` argument to `TFRecordDataset` (TF 2.18 uses `tf.data.Options().experimental_deterministic` instead). Finally, I keep your exact core inference/ensemble logic but ensure IDs and images are read in the same deterministic dataset pass so predictions align with `sample_submission.csv`, and always write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'You’re failing at the very first TensorFlow import due to an incompatibility between TF 2.18 and the installed `protobuf==6.x` runtime, so the notebook never reaches inference/submission. I fix this by forcing TensorFlow to use the bundled pure‑python protobuf implementation and (most importantly) downgrading protobuf inside the session to a TF-compatible 4.x before importing TensorFlow. This is an execution-unblocking change (score-neutral by itself) and keeps your model/inference logic identical so the score can recover from the current near-random output. I also keep the test pipeline deterministic and ID/prediction-aligned exactly as you already intended, and ensure `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'The timeout is dominated by training two large EfficientNet models at 512×512 and by an input pipeline that decodes/resizes JPEGs in a Python-level map for every epoch. To keep the exact same model architecture and training schedule, the main speedups come from: (1) enabling XLA compilation for the training steps, (2) using TFRecords’ built-in parallelism more effectively with non-deterministic execution where safe, (3) restructuring the dataset pipeline to cache *after* decode/resize and to add `repeat()` so `fit()` doesn’t incur iterator re-creation overhead each epoch, while preserving the same number of optimization steps via explicit `steps_per_epoch`. For inference, we avoid a second full pass over the test set just to collect IDs by extracting IDs in the same pass using `model.predict(test_ds)` and splitting outputs, which preserves ordering and correctness while cutting test I/O roughly in half.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:2]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random
import math

print("TensorFlow:", tf.__version__)

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled.")
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tensorflow.keras import layers

NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]


def build_effnet(model_name: str, image_size=(512, 512), num_classes=5):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = layers.Lambda(
        tf.keras.applications.efficientnet.preprocess_input,
        name=f"{model_name}_preprocess",
    )(inputs)

    if model_name == "B6":
        base = tf.keras.applications.EfficientNetB6(
            include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
        )
    elif model_name == "B4":
        base = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
        )
    else:
        raise ValueError("Unsupported model_name")

    outputs = layers.Dense(
        num_classes, activation="softmax", name=f"{model_name}_pred"
    )(base.output)
    model = tf.keras.Model(inputs, outputs, name=f"EfficientNet{model_name}_cassava")
    return model


model_15 = build_effnet("B6", image_size=tuple(IMAGE_SIZE), num_classes=NUM_CLASSES)
model_17 = build_effnet("B4", image_size=tuple(IMAGE_SIZE), num_classes=NUM_CLASSES)



## === cell 3
print(model_15.name, "params:", model_15.count_params())
print(model_17.name, "params:", model_17.count_params())



## === cell 4
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
print(test_df.head())

GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
BATCH_SIZE = 16 * 8
CLASSES = ["0", "1", "2", "3", "4"]


_size_re = re.compile(r"-([0-9]*)\.")


def dataset_sizes(filenames):
    return int(sum(int(_size_re.search(fn).group(1)) for fn in filenames))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)  # keep order stable across runs
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
print("Test TFRecords:", len(TEST_FILENAMES), "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)

TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TRAIN_FILENAMES = sorted(TRAIN_FILENAMES)
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
print("Train TFRecords:", len(TRAIN_FILENAMES), "NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES)




## === cell 5
@tf.function
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        id_num = example["image_name"]
        return img, id_num


def _cache_ds(ds, cache, cache_name):
    if not cache:
        return ds
    if cache is True:
        return ds.cache()  # in-memory
    if isinstance(cache, str):
        return ds.cache(cache)
    return ds


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTOTUNE,
        buffer_size=16 * 1024 * 1024,
    ).with_options(options)

    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled),
        num_parallel_calls=AUTOTUNE,
        deterministic=bool(ordered),
    )
    return dataset


def get_test_data(ordered=False, cache=False):
    dataset = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = _cache_ds(dataset, cache=cache, cache_name="test_ds")
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return dataset


def get_train_val_data(val_fraction=0.10, cache=False):
    full = load_dataset(filenames=TRAIN_FILENAMES, labeled=True, ordered=True)

    n_val = int(NUM_TRAIN_IMAGES * val_fraction)
    n_train = NUM_TRAIN_IMAGES - n_val
    train_ds = full.take(n_train)
    val_ds = full.skip(n_train)

    train_ds = train_ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    return _finalize_train_val(
        train_ds, val_ds, n_train=n_train, n_val=n_val, cache=cache
    )


def _finalize_train_val(train_ds, val_ds, n_train, n_val, cache=False):
    train_ds = _cache_ds(train_ds, cache=cache, cache_name="train_ds")
    val_ds = _cache_ds(val_ds, cache=cache, cache_name="val_ds")

    train_ds = (
        train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE).repeat()
    )
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE).repeat()

    steps_per_epoch = int(math.ceil(n_train / BATCH_SIZE))
    val_steps = int(math.ceil(n_val / BATCH_SIZE))
    return train_ds, val_ds, steps_per_epoch, val_steps


train_ds, val_ds, STEPS_PER_EPOCH, VAL_STEPS = get_train_val_data(
    val_fraction=0.10, cache=True
)
print(
    "Prepared train/val datasets. steps_per_epoch:",
    STEPS_PER_EPOCH,
    "val_steps:",
    VAL_STEPS,
)




## === cell 6
def compile_and_train(model, name):
    for layer in model.layers:
        layer.trainable = False
    model.get_layer(f"{name}_pred").trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        jit_compile=True,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        verbose=1,
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        jit_compile=True,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        verbose=1,
    )


compile_and_train(model_15, "B6")
compile_and_train(model_17, "B4")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ResourceExhaustedError                    Traceback (most recent call last)
/tmp/ipykernel_55/3121950474.py in <cell line: 0>()
     39 
     40 
---> 41 compile_and_train(model_15, "B6")
     42 compile_and_train(model_17, "B4")
     43 

/tmp/ipykernel_55/3121950474.py in compile_and_train(model, name)
     29         jit_compile=True,
     30     )
---> 31     model.fit(
     32         train_ds,
     33         validation_data=val_ds,

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

ResourceExhaustedError: Graph execution error:

Detected at node cluster_58_1/xla_run defined at (most recent call last):
<stack traces unavailable>
Out of memory while trying to allocate 127001552000 bytes.
	 [[{{node cluster_58_1/xla_run}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_multi_step_on_iterator_192048]

## === cell 7
test_ds = get_test_data(ordered=True, cache=True)

print("Computing predictions...")

test_img_ds = test_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
test_id_ds = test_ds.map(lambda x, y: y, num_parallel_calls=AUTOTUNE)

test_ids_list = []
for ids in test_id_ds.as_numpy_iterator():
    test_ids_list.append(ids.astype("U"))
test_ids = np.concatenate(test_ids_list, axis=0)
if test_ids.shape[0] != NUM_TEST_IMAGES:
    NUM_TEST_IMAGES = test_ids.shape[0]

p1 = model_15.predict(test_img_ds, verbose=0)
p2 = model_17.predict(test_img_ds, verbose=0)
probabilities = (p1 + p2) * 0.5

predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))
print("IDs shape:", test_ids.shape)

if len(test_ids) != len(predictions):
    raise RuntimeError(
        f"Length mismatch: ids={len(test_ids)} vs preds={len(predictions)}"
    )

if len(test_ids) != len(test_df):
    raise RuntimeError(
        f"ID count mismatch: TFRecords={len(test_ids)} vs sample_submission={len(test_df)}"
    )



## === cell 8
print("Generating submission.csv file...")

sub_pred = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = test_df[["image_id"]].merge(sub_pred, on="image_id", how="left")

if sub["label"].isna().any():
    mode_label = int(pd.Series(predictions).mode().iloc[0])
    sub["label"] = sub["label"].fillna(mode_label)

sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
print("Wrote:", out_path, "rows:", len(sub))
with open(out_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

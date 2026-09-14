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

0.8851616802659413

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import numpy as np
import pandas as pd
import random

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def get_strategy():
    """
    Detects and returns the best TensorFlow distribution strategy:
    - TPUStrategy for TPU(s)
    - MirroredStrategy for GPU(s)
    - Default strategy for CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except (ValueError, tf.errors.NotFoundError):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using default strategy:", type(strategy).__name__)
    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy


strategy = get_strategy()



## === cell 2
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(f"Global Batch size: {BATCH_SIZE}")



## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Data dir exists:", tf.io.gfile.exists(DATA_DIR))
print("Train CSV exists:", tf.io.gfile.exists(train_csv_path))
print("Sample sub exists:", tf.io.gfile.exists(sample_sub_path))



## === cell 4
MODEL_DIR = (
    "/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras"
)




## === cell 5
def seed_everthing(SEED=28):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Global seed set to {SEED}")


seed_everthing(28)




## === cell 6
def decode_train_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    label = tf.cast(example["label"], tf.int32)
    return image, label


def decode_test_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    return image, example["image_name"]




## === cell 7
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess(image, label):
    image = preprocess_input(image)
    return image, label




## === cell 8
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(40 / 360),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
    ],
    name="data_augmentation",
)



## === cell 9
train_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))

if len(train_files) == 0:
    raise FileNotFoundError(f"No train TFRecords found in: {TRAIN_TFREC_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No test TFRecords found in: {TEST_TFREC_DIR}")

train_files = sorted(train_files)
test_files = sorted(test_files)

raw_train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTO).map(
    decode_train_example, num_parallel_calls=AUTO
)
raw_test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO).map(
    decode_test_example, num_parallel_calls=AUTO
)

print("Train TFRecord files:", len(train_files))
print("Test TFRecord files:", len(test_files))



## === cell 10
train_count = int(pd.read_csv(train_csv_path).shape[0])
val_fraction = 0.1
val_count = int(train_count * val_fraction)
train_take = train_count - val_count

shuffled = raw_train_ds.shuffle(
    buffer_size=4096, seed=28, reshuffle_each_iteration=True
)

train_ds = (
    shuffled.take(train_take)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = (
    shuffled.skip(train_take)
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = raw_test_ds.batch(BATCH_SIZE).prefetch(AUTO)

print("Datasets ready. Train/Val split:", train_take, "/", val_count)




## === cell 11
def build_fallback_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = inputs
    x = preprocess_input(x)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False  # fast + stable within 600s
    x = base.output
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


with strategy.scope():
    if tf.io.gfile.exists(MODEL_DIR):
        model = tf.keras.models.load_model(MODEL_DIR)
        print("Loaded external model:", MODEL_DIR)
    else:
        print("External model not found, training fallback model instead:", MODEL_DIR)
        model = build_fallback_model()
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3846509534.py in <cell line: 0>()
     28         model = build_fallback_model()
     29         # Keep training short enough for runtime; no early stopping introduced.
---> 30         model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)
     31 
     32 

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
Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::FiniteTake::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16678]

## === cell 12
def apply_test_augmentation_batch(images):
    return data_augmentation(images, training=True)


@tf.function
def tta_predict_batch(model, images, num_tta: int):
    preds = 0.0
    for _ in tf.range(num_tta):
        aug = apply_test_augmentation_batch(images)
        aug = preprocess_input(aug)
        preds += model(aug, training=False)
    return preds / tf.cast(num_tta, tf.float32)




## === cell 13
tta_num_augmentations = 10
all_image_ids = []
all_probs = []

for images, ids in test_ds:
    batch_ids = [x.decode("utf-8") for x in ids.numpy().tolist()]
    probs = tta_predict_batch(model, images, tta_num_augmentations).numpy()
    all_image_ids.extend(batch_ids)
    all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)  # [N,5]
pred_labels = np.argmax(all_probs, axis=1).astype(int)

print(
    "Predictions done. N:", len(pred_labels), "Unique labels:", np.unique(pred_labels)
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/858677398.py in <cell line: 0>()
      7     # ids are bytes; keep as python str
      8     batch_ids = [x.decode("utf-8") for x in ids.numpy().tolist()]
----> 9     probs = tta_predict_batch(model, images, tta_num_augmentations).numpy()
     10     all_image_ids.extend(batch_ids)
     11     all_probs.append(probs)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_fileiar9ip0n.py in tf__tta_predict_batch(model, images, num_tta)
     26                 _ = ag__.Undefined('_')
     27                 aug = ag__.Undefined('aug')
---> 28                 ag__.for_stmt(ag__.converted_call(ag__.ld(tf).range, (ag__.ld(num_tta),), None, fscope), None, loop_body, get_state, set_state, ('preds',), {'iterate_names': '_'})
     29                 try:
     30                     do_return = True

/tmp/__autograph_generated_fileiar9ip0n.py in loop_body(itr)
     20                     nonlocal preds
     21                     _ = itr
---> 22                     aug = ag__.converted_call(ag__.ld(apply_test_augmentation_batch), (ag__.ld(images),), None, fscope)
     23                     aug = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(aug),), None, fscope)
     24                     preds = ag__.ld(preds)

/tmp/__autograph_generated_filej4usn4f7.py in tf__apply_test_augmentation_batch(images)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf__apply_test_augmentation_batch

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: in user code:

    File "/tmp/ipykernel_11/2823118427.py", line 12, in tta_predict_batch  *
        aug = apply_test_augmentation_batch(images)
    File "/tmp/ipykernel_11/2823118427.py", line 4, in apply_test_augmentation_batch  *
        return data_augmentation(images, training=True)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py", line 272, in _adjust_input_rank
        raise ValueError(

    ValueError: Exception encountered when calling Sequential.call().
    
    Invalid input shape for input Tensor("images:0", shape=(8, 512, 512, 3), dtype=float32). Expected shape (512, 512, 3), but input has incompatible shape (8, 512, 512, 3)
    
    Arguments received by Sequential.call():
      • inputs=tf.Tensor(shape=(8, 512, 512, 3), dtype=float32)
      • training=True
      • mask=None


## === cell 14
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_image_ids, "label": pred_labels})

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fallback_label = int(np.bincount(pred_labels, minlength=NUM_CLASSES).argmax())
    submission_df["label"] = submission_df["label"].fillna(fallback_label).astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully! -> submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4215192609.py in <cell line: 0>()
      1 # Create submission aligned to sample_submission order (prevents accidental mis-ordering)
      2 sample_sub = pd.read_csv(sample_sub_path)
----> 3 pred_df = pd.DataFrame({"image_id": all_image_ids, "label": pred_labels})
      4 
      5 submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

NameError: name 'pred_labels' is not defined

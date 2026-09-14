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

0.8792686612269568

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, warnings, random, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, applications, optimizers, losses, metrics

warnings.filterwarnings("ignore")
SEED = 555
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")



## === cell 2
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
N_CLASSES = 5
TTA_STEPS = 3
IMAGE_SIZE = [HEIGHT, WIDTH]
SEED = 555




## === cell 3
def data_augment(image, label):
    image = tf.image.rot90(
        image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    img_sz = IMAGE_SIZE[0]
    image = tf.image.resize_with_crop_or_pad(image, img_sz + 6, img_sz + 6)
    image = tf.image.random_crop(image, size=[img_sz, img_sz, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, lower=0, upper=2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label




## === cell 4
def get_name(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-1]


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def resize_image(image, label):
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])
    return image, label


def process_path(file_path):
    name = get_name(file_path)
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, name




## === cell 5
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["label"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    image_name = example["image_name"]
    return image, image_name


def count_data_items(filenames):
    nums = [int(re.compile(r"-([0-9]*)\.").search(f).group(1)) for f in filenames]
    return np.sum(nums)


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(opts)
    parser = read_labeled_tfrecord if labeled else read_unlabeled_tfrecord
    ds = ds.map(parser, num_parallel_calls=AUTO)
    return ds


def get_dataset(files_path, shuffle=False, tta=False, extension="jpg"):
    dataset = tf.data.Dataset.list_files(f"{files_path}*{extension}", shuffle=shuffle)
    dataset = dataset.map(process_path, num_parallel_calls=AUTO)
    if tta:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False, tta=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    if tta:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTO)
    return dataset




## === cell 6
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
submission_template = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(BASE_PATH, "test_tfrecords/*.tfrec"))
TRAIN_FILENAMES = tf.io.gfile.glob(os.path.join(BASE_PATH, "train_tfrecords/*.tfrec"))
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
NUM_TRAIN_IMAGES = count_data_items(TRAIN_FILENAMES)
print(f"Test TFRecords: {len(TEST_FILENAMES)} files, approx {NUM_TEST_IMAGES} images")
print(
    f"Train TFRecords: {len(TRAIN_FILENAMES)} files, approx {NUM_TRAIN_IMAGES} images"
)



## === cell 7
train_ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
train_ds = train_ds.map(data_augment, num_parallel_calls=AUTO)
train_ds = train_ds.map(resize_image, num_parallel_calls=AUTO)
train_ds = train_ds.shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(AUTO)



## === cell 8
with strategy.scope():
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        pooling="avg",
    )
    x = base_model.output
    x = L.Dense(256, activation="relu")(x)
    output = L.Dense(N_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )
model.summary()



## === cell 9
model.fit(train_ds, epochs=2, verbose=2)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/488695328.py in <cell line: 0>()
      1 # Train briefly (2 epochs) – enough for a reasonable baseline
----> 2 model.fit(train_ds, epochs=2, verbose=2)
      3 

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_49713]

## === cell 10
print(f" TTA_STEPS = {TTA_STEPS} ")
test_ds = get_test_dataset(ordered=True, tta=False)  # base predictions
test_images_ds = test_ds.map(lambda img, name: img)

all_probs = []
for step in range(max(1, TTA_STEPS)):
    if TTA_STEPS > 0:
        tta_ds = get_test_dataset(ordered=True, tta=True)
        imgs = tta_ds.map(lambda img, name: img)
    else:
        imgs = test_images_ds
    probs = model.predict(imgs, verbose=0)
    all_probs.append(probs)

probabilities = np.mean(all_probs, axis=0)
predictions = np.argmax(probabilities, axis=1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1424160533.py in <cell line: 0>()
     11     else:
     12         imgs = test_images_ds
---> 13     probs = model.predict(imgs, verbose=0)
     14     all_probs.append(probs)
     15 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:27 transformation with iterator: Iterator::Root::ParallelMapV2::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2: Input to reshape is a tensor with 786432 values, but the requested shape has 150528
	 [[{{node Reshape}}]] [Op:IteratorGetNext] name: 

## === cell 11
print("Generating submission.csv file...")
test_ids_ds = test_ds.map(lambda img, name: name).unbatch()
test_ids = list(test_ids_ds.as_numpy_iterator())
assert len(test_ids) == len(predictions), "Length mismatch between IDs and predictions"

submission = pd.DataFrame({"image_id": test_ids, "label": predictions.astype(int)})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2090921256.py in <cell line: 0>()
      2 # Extract IDs in the same order as test_ds
      3 test_ids_ds = test_ds.map(lambda img, name: name).unbatch()
----> 4 test_ids = list(test_ids_ds.as_numpy_iterator())
      5 assert len(test_ids) == len(predictions), "Length mismatch between IDs and predictions"
      6 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in __next__(self)
   4786       return numpy
   4787 
-> 4788     return nest.map_structure(to_numpy, next(self._iterator))
   4789 
   4790   def next(self):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:19 transformation with iterator: Iterator::Root::Prefetch::Unbatch::ParallelMapV2::Prefetch::MapAndBatch::ParallelMapV2: Input to reshape is a tensor with 786432 values, but the requested shape has 150528
	 [[{{node Reshape}}]] [Op:IteratorGetNext] name:

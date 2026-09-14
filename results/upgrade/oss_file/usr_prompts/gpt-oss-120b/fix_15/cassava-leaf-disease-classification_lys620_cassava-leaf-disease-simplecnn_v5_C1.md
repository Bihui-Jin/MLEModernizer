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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.6128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.35351) has done: 'We speed up data loading by enabling TensorFlow’s multiprocessing workers and mixed‑precision training, which keep the model architecture and training logic unchanged while reducing I/O and compute time. The only modifications are adding a mixed‑precision policy, passing `workers`/`use_multiprocessing` to `model.fit`, and a brief comment explaining each change.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    print("Mixed‑precision policy could not be set:", e)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs available: {len(gpus)}")




## === cell 2
labels = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
labels["label"] = labels["label"].astype(str)
labels.head()




## === cell 3
train_dir = "../input/cassava-leaf-disease-classification/train_images"
test_dir = "../input/cassava-leaf-disease-classification/test_images"




## === cell 4
train_tfrecord_dir = "../input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecord_files = tf.io.gfile.glob(os.path.join(train_tfrecord_dir, "*.tfrec"))

total_samples = len(labels)
train_samples = int(total_samples * 0.7)  # 70% for training
val_samples = total_samples - train_samples  # 30% for validation
batch_size = 512
num_classes = 5

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_fn(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed["image"], channels=3)
    image = tf.image.resize(image, [150, 150])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(parsed["label"], depth=num_classes)
    return image, label


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    shape = tf.shape(image)
    crop_h = tf.cast(0.9 * tf.cast(shape[0], tf.float32), tf.int32)
    crop_w = tf.cast(0.9 * tf.cast(shape[1], tf.float32), tf.int32)
    crop = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(crop, [150, 150])
    return image, label


raw_dataset = tf.data.TFRecordDataset(
    train_tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
)
parsed_dataset = raw_dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = (
    parsed_dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
    .take(train_samples)
    .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

valid_dataset = (
    parsed_dataset.skip(train_samples)
    .take(val_samples)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)




## === cell 5
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(),
    metrics=["accuracy"],
)




## === cell 6
callback_list = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
    callbacks.ModelCheckpoint(
        filepath="best_model.h5", monitor="val_loss", save_best_only=True
    ),
]




## === cell 7
import math

steps_per_epoch = math.ceil(train_samples / batch_size)
validation_steps = math.ceil(val_samples / batch_size)

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=callback_list,
    verbose=2,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/3675204341.py in <cell line: 0>()
      4 validation_steps = math.ceil(val_samples / batch_size)
      5 
----> 6 history = model.fit(
      7     train_dataset,
      8     steps_per_epoch=steps_per_epoch,

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
Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::FiniteTake::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_4]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::FiniteTake::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_3037]

## === cell 8
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs_range, acc, "bo", label="Train Acc")
plt.plot(epochs_range, val_acc, "b", label="Val Acc")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, "bo", label="Train Loss")
plt.plot(epochs_range, val_loss, "b", label="Val Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/108619729.py in <cell line: 0>()
----> 1 acc = history.history["accuracy"]
      2 val_acc = history.history["val_accuracy"]
      3 loss = history.history["loss"]
      4 val_loss = history.history["val_loss"]
      5 epochs_range = range(1, len(acc) + 1)

NameError: name 'history' is not defined

## === cell 9
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)




## === cell 10
model.load_weights("best_model.h5")

test_generator = test_datagen.flow_from_dataframe(
    dataframe=submission,
    directory=test_dir,
    x_col="image_id",
    y_col=None,
    class_mode=None,
    target_size=(150, 150),
    batch_size=512,
    shuffle=False,
    workers=8,
    use_multiprocessing=True,
    max_queue_size=64,
)

pred_probs = model.predict(
    test_generator,
    steps=int(np.ceil(test_generator.samples / test_generator.batch_size)),
    verbose=2,
)
pred_classes = np.argmax(pred_probs, axis=1).astype(str)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2819040085.py in <cell line: 0>()
----> 1 model.load_weights("best_model.h5")
      2 
      3 test_generator = test_datagen.flow_from_dataframe(
      4     dataframe=submission,
      5     directory=test_dir,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'best_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 11
submission_result = pd.DataFrame(
    {"image_id": submission["image_id"], "label": pred_classes}
)
submission_result.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission_result.shape)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4017465032.py in <cell line: 0>()
      1 submission_result = pd.DataFrame(
----> 2     {"image_id": submission["image_id"], "label": pred_classes}
      3 )
      4 submission_result.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv with shape:", submission_result.shape)

NameError: name 'pred_classes' is not defined

## === cell 12
print("Competition pipeline completed successfully!")

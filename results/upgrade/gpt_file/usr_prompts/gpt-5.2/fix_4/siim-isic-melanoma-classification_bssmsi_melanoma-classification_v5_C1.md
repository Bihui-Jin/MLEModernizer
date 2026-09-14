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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.4909

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from functools import partial
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT_DIR = "../input/siim-isic-melanoma-classification"

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

TFRECORD_DIR = os.path.join(INPUT_DIR, "tfrecords")
train_files = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "train*.tfrec"))
test_files = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "test*.tfrec"))

print("TFRECORD_DIR:", TFRECORD_DIR)
print("Found train tfrecs:", len(train_files))
print("Found test tfrecs:", len(test_files))

if len(train_files) == 0 or len(test_files) == 0:
    raise FileNotFoundError(
        "TFRecord files not found. Expected under ../input/siim-isic-melanoma-classification/tfrecords/"
    )



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 3
train_csv = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

print(train_csv.shape, test_csv.shape)
print(train_csv.columns.tolist())
print(test_csv.columns.tolist())



## === cell 4
train_csv.target.value_counts()



## === cell 5
IMG_SIZE = [1024, 1024]
EPOCHS = 12
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

VAL_SPLIT = 0.2
n_total = len(train_csv)
n_vald = int(VAL_SPLIT * n_total)
n_train = n_total - n_vald
n_test = len(test_csv)

print(
    "n_total:",
    n_total,
    "n_train:",
    n_train,
    "n_vald:",
    n_vald,
    "n_test:",
    n_test,
    "BATCH_SIZE:",
    BATCH_SIZE,
)




## === cell 6
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(
        image, [*IMG_SIZE, 3]
    )  # shape annotation instead of reshape
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    image_name = example["image_name"]
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image, image_name


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False

    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 0  # let TF decide
    options.threading.private_threadpool_size = 32
    options.threading.max_intra_op_parallelism = (
        1  # better for input pipelines; compute uses its own threads
    )

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(options)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTO)
    return ds


def get_train_vald_dataset(n_train, n_vald):
    ds = load_dataset(train_files, labeled=True, ordered=False)
    train_ds = (
        ds.take(n_train)
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )
    vald_ds = ds.skip(n_train).take(n_vald).batch(BATCH_SIZE).cache().prefetch(AUTO)
    return train_ds, vald_ds


def get_test_dataset():
    ds = load_dataset(test_files, labeled=False, ordered=True)
    ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTO)
    return ds


train_dataset, vald_dataset = get_train_vald_dataset(n_train, n_vald)
test_dataset = get_test_dataset()

print(
    f"Dataset: {n_train} training images, {n_vald} validation images, {n_test} unlabeled test images"
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1566761281.py in <cell line: 0>()
     72 
     73 
---> 74 train_dataset, vald_dataset = get_train_vald_dataset(n_train, n_vald)
     75 test_dataset = get_test_dataset()
     76 

/tmp/ipykernel_11/1566761281.py in get_train_vald_dataset(n_train, n_vald)
     54 # Speed: keep same semantics, but ensure caching happens after batching (caches batched tensors => fewer objects).
     55 def get_train_vald_dataset(n_train, n_vald):
---> 56     ds = load_dataset(train_files, labeled=True, ordered=False)
     57     train_ds = (
     58         ds.take(n_train)

/tmp/ipykernel_11/1566761281.py in load_dataset(filenames, labeled, ordered)
     39 
     40     options.experimental_optimization.apply_default_optimizations = True
---> 41     options.experimental_optimization.autotune_buffers = True
     42     options.experimental_optimization.autotune_cpu_budget = 0  # let TF decide
     43     options.threading.private_threadpool_size = 32

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 7
def initialize_model(model_name=""):
    pretrained_model = tf.keras.applications.Xception(
        input_shape=[*IMG_SIZE, 3], include_top=False, weights="imagenet"
    )
    pretrained_model.trainable = False

    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model




## === cell 8
with strategy.scope():
    model = initialize_model()



## === cell 9
TRAIN_STEPS = n_train // BATCH_SIZE
VALID_STEPS = max(1, n_vald // BATCH_SIZE)

print("TRAIN_STEPS:", TRAIN_STEPS, "VALID_STEPS:", VALID_STEPS)



## === cell 10
history = model.fit(
    train_dataset,
    epochs=2,  # keep as original code
    steps_per_epoch=TRAIN_STEPS,
    class_weight={0: 1, 1: 2},
    validation_data=vald_dataset,
    validation_steps=VALID_STEPS,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/736957.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_dataset,
      3     epochs=2,  # keep as original code
      4     steps_per_epoch=TRAIN_STEPS,
      5     class_weight={0: 1, 1: 2},

NameError: name 'train_dataset' is not defined

## === cell 11
model.save("model.h5")



## === cell 12
preds = model.predict(test_dataset.map(lambda x, n: x), verbose=0).reshape(-1)

names_tensor = tf.concat([names for _, names in test_dataset], axis=0)
test_image_names = names_tensor.numpy().astype("U").tolist()

if len(test_image_names) != len(preds):
    raise ValueError(
        f"Prediction count mismatch: got {len(preds)} preds but {len(test_image_names)} image_names"
    )

pred = pd.DataFrame({"image_name": test_image_names, "target": preds})
pred.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3431753955.py in <cell line: 0>()
      1 # Speed: replace Python loop with a single compiled predict pass (still same forward pass semantics).
      2 # Correctness preserved: equivalent to calling model(images, training=False) batch-by-batch.
----> 3 preds = model.predict(test_dataset.map(lambda x, n: x), verbose=0).reshape(-1)
      4 
      5 # Speed/correctness: collect names in-order via a fast concatenation from the cached test_dataset.

NameError: name 'test_dataset' is not defined

## === cell 13
pred = pred.sort_values("image_name").reset_index(drop=True)
pred.to_csv("submission.csv", header=True, index=False)
print("Wrote submission.csv with shape:", pred.shape)
print(pred.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3706535648.py in <cell line: 0>()
----> 1 pred = pred.sort_values("image_name").reset_index(drop=True)
      2 pred.to_csv("submission.csv", header=True, index=False)
      3 print("Wrote submission.csv with shape:", pred.shape)
      4 print(pred.head())

NameError: name 'pred' is not defined

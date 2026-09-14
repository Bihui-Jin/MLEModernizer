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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.6294664873796718

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

import re, math, warnings, random, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential

warnings.filterwarnings("ignore")

print("Imports loaded.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except Exception:
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



## === cell 3
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3

N_CLASSES = 11

TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]
AUG_BATCH = BATCH_SIZE




## === cell 4
def to_float32_2(image, label):
    max_val = tf.reduce_max(label, axis=-1, keepdims=True)
    cond = tf.equal(label, max_val)
    label = tf.where(cond, tf.ones_like(label), tf.zeros_like(label))
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [1024, 1024, 3])  # explicit size needed for TPU
    return image


def read_labeled_tfrecord(example):
    train_feature_description = {
        "CVC - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Normal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Incompletely Imaged": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "Swan Ganz Catheter Present": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, train_feature_description)
    image = decode_image(example["image"])
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

    cvca = example["CVC - Abnormal"]
    cvcb = example["CVC - Borderline"]
    cvcn = example["CVC - Normal"]
    etta = example["ETT - Abnormal"]
    ettb = example["ETT - Borderline"]
    ettn = example["ETT - Normal"]
    ngta = example["NGT - Abnormal"]
    ngtb = example["NGT - Borderline"]
    ngti = example["NGT - Incompletely Imaged"]
    ngtn = example["NGT - Normal"]
    sgcp = example["Swan Ganz Catheter Present"]

    values = [etta, ettb, ettn, ngta, ngtb, ngti, ngtn, cvca, cvcb, cvcn, sgcp]

    label = tf.cast(0, tf.int32)
    for i in range(len(values)):
        if values[i] == 1:
            label = tf.cast(i, tf.int32)

    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    image_name = example["StudyInstanceUID"]
    return image, image_name


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return dataset


def data_augment(image, label):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label


def get_training_dataset(dataset, do_aug=True, do_onehot=False):
    if do_aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.batch(AUG_BATCH)
    if do_onehot:
        raise NameError(
            "onehot() was requested but is not defined in the provided script."
        )
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False, tta=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    if tta:
        dataset = dataset.map(
            lambda image, image_name: (data_augment(image, image_name)[0], image_name),
            num_parallel_calls=AUTO,
        )
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 5
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"
submission_sample_path = f"{database_base_path}sample_submission.csv"
submission = pd.read_csv(submission_sample_path)

TRAIN_FILENAMES = tf.io.gfile.glob(f"{database_base_path}train_tfrecords/*.tfrec")
TRAIN_FILENAMES = sorted(TRAIN_FILENAMES)

TEST_FILENAMES = tf.io.gfile.glob(f"{database_base_path}test_tfrecords/*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)

print("Num train TFRecords:", len(TRAIN_FILENAMES))
print("Num test TFRecords:", len(TEST_FILENAMES))
print("Sample submission columns:", submission.columns.tolist())

if len(TRAIN_FILENAMES) == 0 or len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        "TFRecord files not found under expected *_tfrecords/ directories."
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3226513140.py in <cell line: 0>()
     14 
     15 if len(TRAIN_FILENAMES) == 0 or len(TEST_FILENAMES) == 0:
---> 16     raise FileNotFoundError(
     17         "TFRecord files not found under expected *_tfrecords/ directories."
     18     )

FileNotFoundError: TFRecord files not found under expected *_tfrecords/ directories.

## === cell 6
test_ids_ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=True).map(
    lambda img, uid: uid
)
test_ids_list = [x.numpy().decode("utf-8") for x in test_ids_ds]
NUM_TEST_IMAGES = len(test_ids_list)
print(f"Computed NUM_TEST_IMAGES from TFRecords: {NUM_TEST_IMAGES}")
if NUM_TEST_IMAGES <= 0:
    raise RuntimeError("Failed to read any test IDs from TFRecords.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4175244775.py in <cell line: 0>()
      8 print(f"Computed NUM_TEST_IMAGES from TFRecords: {NUM_TEST_IMAGES}")
      9 if NUM_TEST_IMAGES <= 0:
---> 10     raise RuntimeError("Failed to read any test IDs from TFRecords.")
     11 

RuntimeError: Failed to read any test IDs from TFRecords.

## === cell 7
model_path_list = glob.glob("/kaggle/input/ranzcr-clip/*.h5")
model_path_list.sort()
print("Models to predict (external .h5):")
print(*model_path_list, sep="\n")

USE_EXTERNAL_MODELS = len(model_path_list) > 0
print("USE_EXTERNAL_MODELS:", USE_EXTERNAL_MODELS)



## === cell 8
from tensorflow import keras

models = []
if USE_EXTERNAL_MODELS:
    for model_path in model_path_list:
        print("Loading:", model_path)
        K.clear_session()
        models.append(keras.models.load_model(model_path))
    print("Loaded models:", len(models))
else:
    with strategy.scope():
        model = Sequential(
            [
                L.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
                L.Conv2D(16, 3, padding="same", activation="relu"),
                L.MaxPooling2D(),
                L.Conv2D(32, 3, padding="same", activation="relu"),
                L.MaxPooling2D(),
                L.Conv2D(64, 3, padding="same", activation="relu"),
                L.GlobalAveragePooling2D(),
                L.Dense(64, activation="relu"),
                L.Dense(N_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[],
        )

    train_ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
    train_ds = get_training_dataset(train_ds, do_aug=True, do_onehot=False)

    STEPS_PER_EPOCH = 200
    EPOCHS = 2
    print(
        f"Training fallback model: steps_per_epoch={STEPS_PER_EPOCH}, epochs={EPOCHS}, batch={BATCH_SIZE}"
    )
    model.fit(train_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

    models = [model]
    print("Trained fallback model(s):", len(models))



## === cell 9
print(f" TTA_STEPS = {TTA_STEPS} ")
all_probs = []

if TTA_STEPS > 0:
    for step in range(TTA_STEPS):
        test_ds = get_test_dataset(ordered=True, tta=True)
        print(f"TTA step {step+1}/{TTA_STEPS}")
        test_images_ds = test_ds.map(lambda image, image_name: image)
        probs_step = np.average(
            [m.predict(test_images_ds, verbose=0) for m in models], axis=0
        )
        all_probs.append(probs_step)
    probabilities = np.mean(np.stack(all_probs, axis=0), axis=0)
else:
    test_ds = get_test_dataset(ordered=True, tta=False)
    test_images_ds = test_ds.map(lambda image, image_name: image)
    probabilities = np.average(
        [m.predict(test_images_ds, verbose=0) for m in models], axis=0
    )

print("probabilities shape:", probabilities.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/414565311.py in <cell line: 0>()
      8         test_images_ds = test_ds.map(lambda image, image_name: image)
      9         probs_step = np.average(
---> 10             [m.predict(test_images_ds, verbose=0) for m in models], axis=0
     11         )
     12         all_probs.append(probs_step)

/tmp/ipykernel_11/414565311.py in <listcomp>(.0)
      8         test_images_ds = test_ds.map(lambda image, image_name: image)
      9         probs_step = np.average(
---> 10             [m.predict(test_images_ds, verbose=0) for m in models], axis=0
     11         )
     12         all_probs.append(probs_step)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

## === cell 10
test_ids = np.array(test_ids_list, dtype=str)

if probabilities.ndim != 2:
    raise ValueError(
        f"Expected 2D probabilities (n_samples, n_classes) but got shape {probabilities.shape}"
    )
if len(test_ids) != probabilities.shape[0]:
    raise ValueError(
        f"Mismatch: got {len(test_ids)} ids but {probabilities.shape[0]} predictions"
    )



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/95766454.py in <cell line: 0>()
      2 test_ids = np.array(test_ids_list, dtype=str)
      3 
----> 4 if probabilities.ndim != 2:
      5     raise ValueError(
      6         f"Expected 2D probabilities (n_samples, n_classes) but got shape {probabilities.shape}"

NameError: name 'probabilities' is not defined

## === cell 11
REQUIRED_TARGETS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
REQUIRED_COLUMNS = ["StudyInstanceUID"] + REQUIRED_TARGETS

n_model_outputs = probabilities.shape[1]
if n_model_outputs != len(REQUIRED_TARGETS):
    raise ValueError(
        f"Model outputs {n_model_outputs} do not match required targets ({len(REQUIRED_TARGETS)})."
    )

pred_df = pd.DataFrame({"StudyInstanceUID": test_ids})
for i, col in enumerate(REQUIRED_TARGETS):
    pred_df[col] = probabilities[:, i].astype(np.float32)

pred_df = pred_df[REQUIRED_COLUMNS]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2783227020.py in <cell line: 0>()
     14 REQUIRED_COLUMNS = ["StudyInstanceUID"] + REQUIRED_TARGETS
     15 
---> 16 n_model_outputs = probabilities.shape[1]
     17 if n_model_outputs != len(REQUIRED_TARGETS):
     18     # Keep logic strict: this pipeline's label encoding is 11-way.

NameError: name 'probabilities' is not defined

## === cell 12
print("Generating submission.csv file...")
pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2648793892.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
----> 2 pred_df.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", pred_df.shape)
      4 print(pred_df.head())
      5 

NameError: name 'pred_df' is not defined

## === cell 13
import subprocess

subprocess.run(["bash", "-lc", "head -n 3 submission.csv"], check=False)
subprocess.run(["bash", "-lc", "ls -lah submission.csv"], check=False)

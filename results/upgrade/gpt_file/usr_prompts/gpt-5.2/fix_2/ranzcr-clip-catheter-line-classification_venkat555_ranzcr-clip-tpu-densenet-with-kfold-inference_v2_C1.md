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

0.5084721608910747

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import math, os, re, warnings, random, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")



## === cell 3
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



## === cell 4
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"



## === cell 5
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 5
TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]  # At this size, a GPU will run out of memory. Use the TPU.
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUG_BATCH = BATCH_SIZE




## === cell 6
def count_tfrecord_items(filenames):
    """
    Counts examples by iterating TFRecord(s). For ~3k test examples this is fast enough and robust
    across filename patterns (unlike regex-based count parsing).
    """
    if not filenames:
        return 0
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    return int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy())




## === cell 7
def data_augment(image, label):
    image = tf.image.rot90(image, k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    IMG_SIZE = IMAGE_SIZE[0]
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label




## === cell 9
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
    uid = example["StudyInstanceUID"]

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
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[0]])
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
    image = tf.image.rot90(image, k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    IMG_SIZE = IMAGE_SIZE[0]
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False, tta=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    if tta:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 11
submission = pd.read_csv(f"{database_base_path}sample_submission.csv")
print(submission.head())

TEST_FILENAMES = tf.io.gfile.glob(f"{database_base_path}test_tfrecords/*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)

NUM_TEST_IMAGES = count_tfrecord_items(TEST_FILENAMES)
print(f"TFRecords: {len(TEST_FILENAMES)} files")
print(f"Test images counted: {NUM_TEST_IMAGES}")

if NUM_TEST_IMAGES <= 0:
    NUM_TEST_IMAGES = int(len(submission))
    print(f"Fallback NUM_TEST_IMAGES to len(sample_submission): {NUM_TEST_IMAGES}")



## === cell 13
model_path_list = glob.glob("/kaggle/input/ranzcr-clip/*.h5")
model_path_list.sort()

print("Models to predict:")
print(*model_path_list, sep="\n")

if len(model_path_list) == 0:
    raise FileNotFoundError(
        "No .h5 models found under /kaggle/input/ranzcr-clip/. Cannot run inference."
    )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/330980926.py in <cell line: 0>()
      6 
      7 if len(model_path_list) == 0:
----> 8     raise FileNotFoundError(
      9         "No .h5 models found under /kaggle/input/ranzcr-clip/. Cannot run inference."
     10     )

FileNotFoundError: No .h5 models found under /kaggle/input/ranzcr-clip/. Cannot run inference.

## === cell 15
from tensorflow import keras

models = []
for model_path in model_path_list:
    print(model_path)
    K.clear_session()
    models.append(keras.models.load_model(model_path))



## === cell 17
print(" TTA_STEPS = {} ".format(TTA_STEPS))
if TTA_STEPS > 0:
    probs_steps = []
    for step in range(TTA_STEPS):
        test_ds = get_test_dataset(ordered=True, tta=True)
        print(f"TTA step {step+1}/{TTA_STEPS}")
        test_images_ds = test_ds.map(lambda image, image_name: image)
        probs = np.average(
            [m.predict(test_images_ds, verbose=0) for m in models], axis=0
        )
        probs_steps.append(probs)
    probabilities = np.mean(probs_steps, axis=0)
else:
    test_ds = get_test_dataset(ordered=True, tta=False)
    test_images_ds = test_ds.map(lambda image, image_name: image)
    probabilities = np.average([models[0].predict(test_images_ds, verbose=0)], axis=0)

probabilities = np.asarray(probabilities)
probabilities = np.clip(probabilities, 0.0, 1.0)



## === cell 18
expected_cols = list(submission.columns)
id_col = expected_cols[0]
target_cols = expected_cols[1:]

n_out = probabilities.shape[1]
print(f"Model output shape: {probabilities.shape} (n_out={n_out})")
print(f"Sample submission targets: {len(target_cols)} columns")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2563864894.py in <cell line: 0>()
      6 # If model outputs 11 columns but sample_submission has fewer (as shown in prompt), align safely.
      7 # We'll fill missing required columns (if any) with zeros, and drop extras.
----> 8 n_out = probabilities.shape[1]
      9 print(f"Model output shape: {probabilities.shape} (n_out={n_out})")
     10 print(f"Sample submission targets: {len(target_cols)} columns")

IndexError: tuple index out of range

## === cell 19
print("Generating submission.csv file...")

test_ids = []
for _, batch_ids in get_test_dataset(ordered=True, tta=False):
    test_ids.extend(batch_ids.numpy().astype("U").tolist())
test_ids = np.array(test_ids, dtype="U")

n = min(len(test_ids), probabilities.shape[0], len(submission))
test_ids = test_ids[:n]
probabilities = probabilities[:n]

sub_df = pd.DataFrame({id_col: test_ids})

probs_aligned = np.zeros((n, len(target_cols)), dtype=np.float32)
take = min(len(target_cols), probabilities.shape[1])
probs_aligned[:, :take] = probabilities[:, :take]

for j, c in enumerate(target_cols):
    sub_df[c] = probs_aligned[:, j]

sub_df = sub_df[expected_cols]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns: {list(sub_df.columns)}")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/346550521.py in <cell line: 0>()
      8 
      9 # Truncate/align in case of any mismatch between counted ids and probabilities
---> 10 n = min(len(test_ids), probabilities.shape[0], len(submission))
     11 test_ids = test_ids[:n]
     12 probabilities = probabilities[:n]

IndexError: tuple index out of range

## === cell 20
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4030184905.py in <cell line: 0>()
      1 # Show head of submission for sanity
----> 2 with open("submission.csv", "r") as f:
      3     for _ in range(5):
      4         print(f.readline().rstrip("\n"))

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

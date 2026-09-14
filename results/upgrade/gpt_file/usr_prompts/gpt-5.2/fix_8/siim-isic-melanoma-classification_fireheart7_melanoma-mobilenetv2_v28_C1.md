# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import re
import gc
import sys
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _pip_install_quiet(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    if int(_pb_ver.split(".")[0]) >= 5:
        _pip_install_quiet("protobuf<5")
        import importlib

        importlib.invalidate_caches()
except Exception:
    try:
        _pip_install_quiet("protobuf<5")
    except Exception:
        pass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from functools import partial
import sklearn
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy()

print("Device:", (tpu.master() if tpu is not None else "CPU/GPU"))
print("Number of replicas:", strategy.num_replicas_in_sync)
print("Version of Tensorflow used : ", tf.__version__)



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

LOCAL_DATASET_ROOT = "/kaggle/input/siim-isic-melanoma-classification"
LOCAL_TFRECORD_DIR = os.path.join(LOCAL_DATASET_ROOT, "tfrecords")

GCS_PATH = None
try:
    from kaggle_datasets import KaggleDatasets

    GCS_PATH = KaggleDatasets().get_gcs_path()
except Exception:
    GCS_PATH = None

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
SHAPE = [256, 256]

print("Batch Size = ", BATCH_SIZE)
print("GCS Path = ", GCS_PATH)
print("Local TFRecord dir = ", LOCAL_TFRECORD_DIR)

RUN_EDA = False

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))
_PREFETCH_DEVICE = "/GPU:0" if _HAS_GPU else None
print("Has GPU:", _HAS_GPU)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

_TFDATA_POOL = tf.data.experimental.ThreadPool(  # type: ignore[attr-defined]
    size=min(32, (os.cpu_count() or 8))
)



## === cell 3
train = pd.DataFrame(
    pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
)
train.head()



## === cell 4
test = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/test.csv"))
test.head()



## === cell 5
train.info()



## === cell 6
test.info()



## === cell 7
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"



## === cell 8
image_names = train["image_name"].values + ".jpg"
random_images = [np.random.choice(image_names) for i in range(4)]
random_images



## === cell 9
sample_images = []



## === cell 10
if RUN_EDA:
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        image = cv2.imread(os.path.join(train_dir, random_images[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample_images.append(image)
        plt.imshow(image, cmap="gray")
        plt.grid(True)
    plt.tight_layout()




## === cell 11
def non_local_means_denoising(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image




## === cell 12
if RUN_EDA:
    sample_image = cv2.imread(os.path.join(train_dir, random_images[0]))
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    denoised_image = non_local_means_denoising(sample_image)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image, cmap="gray")
    plt.grid(False)
    plt.title("Normal Image")

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap="gray")
    plt.grid(False)
    plt.title("Denoised image")
    plt.tight_layout()




## === cell 13
def histogram_equalization(image):
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    y_channel = image_ycrcb[:, :, 0]
    cr_channel = image_ycrcb[:, :, 1]
    cb_channel = image_ycrcb[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(y_channel)
    equalized_image = cv2.merge([equalized, cr_channel, cb_channel])
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)
    return equalized_image




## === cell 14
if RUN_EDA:
    equalized_image = histogram_equalization(denoised_image)



## === cell 15
if RUN_EDA:
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image, cmap="gray")
    plt.grid(False)
    plt.title("Normal Image", fontsize=14)

    plt.subplot(1, 3, 2)
    plt.imshow(denoised_image, cmap="gray")
    plt.grid(False)
    plt.title("denoised image after histogram processing", fontsize=14)

    plt.subplot(1, 3, 3)
    plt.imshow(equalized_image, cmap="gray")
    plt.grid(False)
    plt.title("Histogram equalized image", fontsize=14)
    plt.tight_layout()




## === cell 16
def segmentation(image, k, attempts):
    vectorized = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    res, label, center = cv2.kmeans(
        vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image




## === cell 17
if RUN_EDA:
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    plt.imshow(denoised_image, cmap="gray")
    plt.grid(False)
    plt.title("de Noised Image")



## === cell 18
if RUN_EDA:
    plt.figure(figsize=(12, 8))
    segmented_image = segmentation(denoised_image, 3, 10)
    plt.subplot(1, 3, 1)
    plt.imshow(segmented_image, cmap="gray")
    plt.grid(False)
    plt.title("Segmented Image with k = 3")

    segmented_image = segmentation(denoised_image, 4, 10)
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image, cmap="gray")
    plt.grid(False)
    plt.title("Segmented Image with k = 4")

    segmented_image = segmentation(denoised_image, 5, 10)
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image, cmap="gray")
    plt.grid(False)
    plt.title("Segmented Image with k = 5")



## === cell 19
from sklearn.model_selection import train_test_split

if GCS_PATH is not None:
    tfrec_train_glob = GCS_PATH + "/tfrecords/train*.tfrec"
    tfrec_test_glob = GCS_PATH + "/tfrecords/test*.tfrec"
else:
    tfrec_train_glob = os.path.join(LOCAL_TFRECORD_DIR, "train*.tfrec")
    tfrec_test_glob = os.path.join(LOCAL_TFRECORD_DIR, "test*.tfrec")

all_train_files = tf.io.gfile.glob(tfrec_train_glob)
all_test_files = tf.io.gfile.glob(tfrec_test_glob)

training_files, validation_files = train_test_split(
    all_train_files, test_size=0.1, random_state=42
)
testing_files = all_test_files

print("Number of training files = ", len(training_files))
print("Number of validation files = ", len(validation_files))
print("Number of test files = ", len(testing_files))




## === cell 20
@tf.function
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [SHAPE[0], SHAPE[1], 3])
    return image




## === cell 21
if RUN_EDA and len(sample_images) > 0:
    sample_images[0].shape



## === cell 22
training_files[:2], testing_files[:2]



## === cell 23
feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}




## === cell 24
def parse_function(example):
    return tf.io.parse_single_example(example, feature_description)




## === cell 25
if RUN_EDA and len(training_files) > 0:
    file = tf.data.TFRecordDataset(training_files[0])
    parsed_dataset = file.map(parse_function)
    next(iter(parsed_dataset)).keys()




## === cell 26
@tf.function
def read_tfrecord(example, labeled):
    if labeled is True:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled is True:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    else:
        image_name = example["image_name"]
        return image, image_name




## === cell 27
def load_dataset(filenames, labeled, ordered):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    try:
        options.experimental_threading.private_threadpool = _TFDATA_POOL
        options.experimental_threading.max_intra_op_parallelism = 1
    except Exception:
        pass

    try:
        options.experimental_slack = True
    except Exception:
        pass

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset




## === cell 28
def image_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label




## === cell 29
def get_training_dataset():
    dataset = load_dataset(training_files, labeled=True, ordered=False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    if _PREFETCH_DEVICE is not None:
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                _PREFETCH_DEVICE, buffer_size=AUTOTUNE
            )
        )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 30
def get_validation_dataset():
    dataset = load_dataset(validation_files, labeled=True, ordered=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    if _PREFETCH_DEVICE is not None:
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                _PREFETCH_DEVICE, buffer_size=AUTOTUNE
            )
        )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 31
def get_test_dataset():
    dataset = load_dataset(testing_files, labeled=False, ordered=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

    if _PREFETCH_DEVICE is not None:
        dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                _PREFETCH_DEVICE, buffer_size=AUTOTUNE
            )
        )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 32
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()



## === cell 33
_count_re = re.compile(r"-([0-9]*)\.")


def count_data_items(filenames):
    n = [int(_count_re.search(filename).group(1)) for filename in filenames]
    return int(np.sum(n))


num_training_images = count_data_items(training_files)
num_validation_images = count_data_items(validation_files)
num_testing_images = count_data_items(testing_files)

STEPS_PER_EPOCH_TRAIN = num_training_images // BATCH_SIZE
STEPS_PER_EPOCH_VAL = num_validation_images // BATCH_SIZE

print("Number of Training Images = ", num_training_images)
print("Number of Validation Images = ", num_validation_images)
print("Number of Testing Images = ", num_testing_images)
print("\n")
print("Numer of steps per epoch in Train = ", STEPS_PER_EPOCH_TRAIN)
print("Numer of steps per epoch in Validation = ", STEPS_PER_EPOCH_VAL)



## === cell 34
if RUN_EDA:
    image_batch, label_batch = next(iter(training_dataset))




## === cell 35
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(20, 20))
    for n in range(8):
        ax = plt.subplot(2, 4, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n] == 0:
            plt.title("BENIGN")
        else:
            plt.title("MALIGNANT")
    plt.grid(False)
    plt.tight_layout()


if RUN_EDA:
    show_batch(image_batch.numpy(), label_batch.numpy())



## === cell 36
if RUN_EDA:
    del image_batch
    del label_batch
    gc.collect()



## === cell 37
malignant = len(train[train["target"] == 1])
benign = len(train[train["target"] == 0])
total = len(train)

print("Malignant Cases in Train Data = ", malignant)
print("Benign Cases In Train Dataset = ", benign)
print("Total Cases In Train Dataset = ", total)
print("Ratio of Malignant to Benign = ", malignant / benign)



## === cell 38
weight_malignant = (total / malignant) / 2.0
weight_benign = (total / benign) / 2.0

class_weight = {0: weight_benign, 1: weight_malignant}

print("Weight for benign cases = ", class_weight[0])
print("Weight for malignant cases = ", class_weight[1])



## === cell 39
callback_early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=15, verbose=0, restore_best_weights=True
)

callbacks_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.1, patience=10, verbose=0, min_lr=1e-6
)

callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "melanoma_weights.weights.h5",
    save_weights_only=True,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
)



## === cell 40
with strategy.scope():
    bias = np.log(malignant / benign)
    bias = tf.keras.initializers.Constant(bias)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(SHAPE[0], SHAPE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        steps_per_execution=64,
    )
    model.summary()

    EPOCHS = 500
    history = model.fit(
        training_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        validation_data=validation_dataset,
        validation_steps=STEPS_PER_EPOCH_VAL,
        callbacks=[callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
        class_weight=class_weight,
        verbose=1,
    )



## === cell 41
if os.path.exists("melanoma_weights.weights.h5"):
    model.load_weights("melanoma_weights.weights.h5")

n_epochs_it_ran_for = len(history.history["loss"])
n_epochs_it_ran_for



## === cell 42
if RUN_EDA:
    X = np.arange(0, n_epochs_it_ran_for, 1)
    plt.figure(1, figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(X, history.history["loss"], label="Training Loss")
    plt.plot(X, history.history["val_loss"], label="Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.plot(X, history.history["auc"], label="Training AUC")
    plt.plot(X, history.history["val_auc"], label="Validation AUC")
    plt.grid(True)
    plt.legend()



## === cell 43
testing_dataset = get_test_dataset()

testing_dataset_images = testing_dataset.map(
    lambda image, image_name: image, num_parallel_calls=AUTOTUNE, deterministic=True
)

decoded_test_names = test["image_name"].astype(str).values



## === cell 44
try:
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics,
        steps_per_execution=256,
    )
except Exception:
    pass

resulting_probabilities = model.predict(testing_dataset_images, verbose=1)
resulting_probabilities = np.asarray(resulting_probabilities).reshape(-1)

len(resulting_probabilities), resulting_probabilities[:5]



## === cell 45
sample_submission_file = pd.read_csv(
    "../input/siim-isic-melanoma-classification/sample_submission.csv"
)
sample_submission_file.head()



## === cell 46
pred_dataframe = pd.DataFrame(
    {
        "image_name": decoded_test_names[: len(resulting_probabilities)],
        "target": resulting_probabilities.astype(np.float32),
    }
)

submission = sample_submission_file[["image_name"]].merge(
    pred_dataframe, on="image_name", how="left"
)

if submission["target"].isna().any():
    submission["target"] = submission["target"].fillna(
        float(np.mean(resulting_probabilities))
    )

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 47
print("Saved submission.csv with shape:", submission.shape)
print(submission.columns.tolist())
print(submission["target"].describe())
assert os.path.exists("submission.csv")
assert submission.shape[0] == sample_submission_file.shape[0]
assert submission.columns.tolist() == ["image_name", "target"]

# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

# 5. Target score

0.7838

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import math, random, re, time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
import gc
import cv2
import seaborn as sns

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None

from tqdm import tqdm



## === cell 1
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")



## === cell 2
dirname = "../input/siim-isic-melanoma-classification/"
train = pd.read_csv(dirname + "train.csv")
test = pd.read_csv(dirname + "test.csv")
print(train.head())
print(len(train))
print(len(test))
print(train["target"].value_counts())



## === cell 3
sns.countplot(train["target"])



## === cell 4
try:
    if KaggleDatasets is None:
        raise RuntimeError("KaggleDatasets not available")
    GCS_PATH = KaggleDatasets().get_gcs_path()
except Exception:
    GCS_PATH = "../input/siim-isic-melanoma-classification"

train_filenames = tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec")
test_filenames = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")

print("Num train tfrecords:", len(train_filenames))
print("Num test tfrecords:", len(test_filenames))



## === cell 5
if str(GCS_PATH).startswith("gs://"):
    import subprocess, shlex

    subprocess.run(shlex.split(f"gsutil ls {GCS_PATH}"), check=False)
else:
    print("Skipping gsutil ls (not a GCS path):", GCS_PATH)



## === cell 6
train_filenames, valid_filenames = train_test_split(
    train_filenames, test_size=0.2, shuffle=True, random_state=SEED
)



## === cell 7
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
AUTO = tf.data.AUTOTUNE
imSize = 1024




## === cell 8
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])  # explicit size needed for TPU
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(options)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=bool(ordered),
    )
    return ds


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0, 2)
    return image, label


def get_training_dataset():
    ds = load_dataset(train_filenames, labeled=True, ordered=False)
    ds = ds.cache()
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.repeat()
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_val_dataset():
    ds = load_dataset(valid_filenames, labeled=True, ordered=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = int(count_data_items(train_filenames))
NUM_VAL_IMAGES = int(count_data_items(valid_filenames))
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VAL_IMAGES} labeled validation images"
)
print("BATCH_SIZE:", BATCH_SIZE, "STEPS_PER_EPOCH:", STEPS_PER_EPOCH)



## === cell 9
train_ds = get_training_dataset()
val_ds = get_val_dataset()

for image, label in train_ds.take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())




## === cell 10
def res_block(X_in, channels):
    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([X, X_in])
    X = layers.LeakyReLU()(X)
    return X




## === cell 11
def my_model():
    X_in = layers.Input((1024, 1024, 3))

    X = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="avg_pool1")(X_in)

    X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv1")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 64)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool1")(X)
    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool1.2")(X)

    X = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv2")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 128)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool2")(X)

    X = layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool3")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool4")(X)

    X = res_block(X, 256)

    X = layers.MaxPool2D(pool_size=(2, 2), strides=2, name="max_pool5")(X)

    X = layers.Flatten()(X)
    X = layers.Dense(4096, activation="relu", name="fc1")(X)
    X = layers.Dense(1024, activation="relu", name="fc2")(X)
    X_out = layers.Dense(1, activation="sigmoid", name="answer")(X)

    model = Model(inputs=X_in, outputs=X_out, name="pinnet")
    return model




## === cell 12
with strategy.scope():
    model = my_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 13
model.summary()



## === cell 14
from tensorflow.keras.callbacks import ReduceLROnPlateau


def callback():
    cb = []
    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=1,
        min_lr=0.00001,
    )
    cb.append(reduceLROnPlat)
    return cb




## === cell 15
cb = callback()
epochs = 10

history = model.fit(
    train_ds,
    epochs=epochs,
    verbose=True,
    steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,
    validation_data=val_ds,
    validation_steps=NUM_VAL_IMAGES // BATCH_SIZE,
    callbacks=cb,
)



## === cell 16
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()



## === cell 17
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()



## === cell 18
num_test_images = int(count_data_items(test_filenames))
num_test_images




## === cell 19
def get_test_dataset(ordered=False):
    ds = load_dataset(test_filenames, labeled=False, ordered=ordered)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_dataset = get_test_dataset(ordered=True)

print("Computing predictions...")
test_images_ds = test_dataset.map(lambda image, idnum: image, num_parallel_calls=AUTO)
probabilities = model.predict(test_images_ds, verbose=1).flatten()

print("Generating submission.csv file...")
test_ids_ds = test_dataset.map(
    lambda image, idnum: idnum, num_parallel_calls=AUTO
).unbatch()
test_ids = next(iter(test_ids_ds.batch(num_test_images))).numpy().astype("U")

np.savetxt(
    "submission.csv",
    np.rec.fromarrays([test_ids, probabilities]),
    fmt=["%s", "%f"],
    delimiter=",",
    header="image_name,target",
    comments="",
)
print("Saved submission.csv with", len(test_ids), "rows")

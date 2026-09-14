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

# 5. Code solution

## === cell 0
import math, random, os, re, time, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
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



## === cell 2
dirname = "/kaggle/input/siim-isic-melanoma-classification/"
train = pd.read_csv(os.path.join(dirname, "train.csv"))
test = pd.read_csv(os.path.join(dirname, "test.csv"))

print(train.head())
print("train rows:", len(train))
print("test rows:", len(test))
print(train["target"].value_counts(dropna=False))



## === cell 3
pass



## === cell 4
tfrecord_dir = os.path.join(dirname, "tfrecords")
train_filenames = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "train*.tfrec")))
test_filenames = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "test*.tfrec")))

print("Found TFRecords - train:", len(train_filenames), "test:", len(test_filenames))
assert len(train_filenames) > 0, f"No train tfrecords found in {tfrecord_dir}"
assert len(test_filenames) > 0, f"No test tfrecords found in {tfrecord_dir}"



## === cell 5
train_filenames, valid_filenames = train_test_split(
    train_filenames, test_size=0.2, shuffle=True, random_state=42
)
print("TFRecord shards - train:", len(train_filenames), "valid:", len(valid_filenames))



## === cell 6
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
AUTO = tf.data.experimental.AUTOTUNE
imSize = 1024




## === cell 7
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [imSize, imSize, 3])
    return image


@tf.function
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function
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
    options.experimental_slack = True  # semantics unchanged; improves overlap

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=bool(ordered),
    )
    return dataset


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0, 2)
    return image, label


def get_training_dataset():
    dataset = load_dataset(train_filenames, labeled=True, ordered=False)

    dataset = dataset.cache()

    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_val_dataset():
    dataset = load_dataset(valid_filenames, labeled=True, ordered=True)

    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(train_filenames)
NUM_VALID_IMAGES = count_data_items(valid_filenames)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VALID_IMAGES} labeled validation images"
)
print("BATCH_SIZE:", BATCH_SIZE, "STEPS_PER_EPOCH:", STEPS_PER_EPOCH)



## === cell 8
for image, label in get_training_dataset().take(1):
    print(image.shape, label.shape)
    tf.print("Training data label examples:", label[:20])




## === cell 9
def res_block(X_in, channels):
    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X_in)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([X, X_in])
    X = layers.LeakyReLU()(X)
    return X




## === cell 10
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




## === cell 11
with strategy.scope():
    model = my_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=False,
        jit_compile=True,
    )



## === cell 12
model.summary()



## === cell 13
from tensorflow.keras.callbacks import ReduceLROnPlateau


def callback():
    cb = []
    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=1,
        min_lr=0.00001,
    )
    cb.append(reduceLROnPlat)
    return cb




## === cell 14
cb = callback()
epochs = 10

train_ds = get_training_dataset()
val_ds = get_val_dataset()

history = model.fit(
    train_ds,
    epochs=epochs,
    verbose=True,
    steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,
    validation_data=val_ds,
    validation_steps=max(1, NUM_VALID_IMAGES // BATCH_SIZE),
    callbacks=cb,
)



## === cell 15
pass



## === cell 16
pass



## === cell 17
num_test_images = count_data_items(test_filenames)
print("num_test_images from TFRecords:", num_test_images)
print("num_test_images from test.csv:", len(test))




## === cell 18
def get_test_dataset(ordered=False):
    dataset = load_dataset(test_filenames, labeled=False, ordered=ordered)
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


test_dataset = get_test_dataset(ordered=True)



## === cell 19
print("Computing predictions...")
test_images_ds = test_dataset.map(
    lambda image, idnum: image, num_parallel_calls=AUTO, deterministic=True
)

probabilities = model.predict(test_images_ds, verbose=1).reshape(-1)

test_ids = test["image_name"].values.astype("U")
pred_df = pd.DataFrame(
    {"image_name": test_ids, "target": probabilities.astype(np.float32)}
)

test_order = test[["image_name"]].copy()
sub = test_order.merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(np.nanmean(probabilities)))

sub["target"] = sub["target"].clip(0.0, 1.0)

assert list(sub.columns) == ["image_name", "target"]
assert len(sub) == len(test)

print("Generating submission.csv file...")
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))

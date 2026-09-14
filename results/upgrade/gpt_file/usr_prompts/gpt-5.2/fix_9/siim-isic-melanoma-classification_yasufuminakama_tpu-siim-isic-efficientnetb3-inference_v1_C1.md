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

No external packages required in the script and installed.

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
import os, sys, re, math, importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)



## === cell 1
from tensorflow.keras.applications import EfficientNetB3



## === cell 2
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 3
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

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 4
AUTO = tf.data.experimental.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
TFRECORD_PATH = os.path.join(BASE_PATH, "tfrecords")

EPOCHS = 1
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [300, 300]



## === cell 5
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
print("Sample submission shape:", sub.shape)
print("Sample submission columns:", list(sub.columns))
print(sub.head())



## === cell 6
TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "train*.tfrec"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_PATH, "test*.tfrec"))

if len(TRAINING_FILENAMES) == 0 or len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        f"TFRecords not found. TRAIN={len(TRAINING_FILENAMES)} TEST={len(TEST_FILENAMES)} "
        f"under {TFRECORD_PATH}"
    )

TRAINING_FILENAMES = sorted(TRAINING_FILENAMES)
TEST_FILENAMES = sorted(TEST_FILENAMES)

print(
    "TFRecord shards:", len(TRAINING_FILENAMES), "train,", len(TEST_FILENAMES), "test"
)
print("Example train shard:", TRAINING_FILENAMES[0])
print("Example test shard:", TEST_FILENAMES[0])




## === cell 7
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_jpeg": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    img_bytes = tf.cond(
        tf.strings.length(example["image"]) > 0,
        lambda: example["image"],
        lambda: example["image_jpeg"],
    )
    image = decode_image(img_bytes)
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_jpeg": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    img_bytes = tf.cond(
        tf.strings.length(example["image"]) > 0,
        lambda: example["image"],
        lambda: example["image_jpeg"],
    )
    image = decode_image(img_bytes)

    idnum = tf.cond(
        tf.strings.length(example["image_name"]) > 0,
        lambda: example["image_name"],
        lambda: example["image_id"],
    )
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opt = tf.data.Options()
    if not ordered:
        opt.experimental_deterministic = False

    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTO,
        compression_type=None,
        buffer_size=8 * 1024 * 1024,
    )
    ds = ds.with_options(opt)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return ds


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset():
    ds = load_dataset(TRAINING_FILENAMES, labeled=True)
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.repeat()
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} unlabeled test images"
)
print(f"BATCH_SIZE={BATCH_SIZE}, STEPS_PER_EPOCH={STEPS_PER_EPOCH}")




## === cell 8
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.000075,
    lr_min=0.000001,
    lr_rampup_epochs=20,
    lr_sustain_epochs=0,
    lr_exp_decay=0.8,
):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn




## === cell 9
with strategy.scope():
    backbone = EfficientNetB3(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    model = tf.keras.Sequential(
        [
            backbone,
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)
print("Model built. Backbone weights:", "imagenet")



## === cell 10
train_ds = get_training_dataset()
print("Training ...")
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
)
print("Training done.")



## === cell 11
test_ds = get_test_dataset(ordered=True)

print("Collecting test ids (ordered) ...")
id_ds = test_ds.map(lambda x, y: y, num_parallel_calls=AUTO).unbatch()

test_ids_list = []
for x in id_ds:
    v = x.numpy()
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        test_ids_list.append(v.decode("utf-8"))
    else:
        test_ids_list.append(str(v))
test_ids = np.asarray(test_ids_list, dtype=object)

if (len(test_ids) == 0) or (np.mean([len(s) for s in test_ids]) < 3):
    print("TFRecord ids look empty; falling back to test.csv image_name order.")
    test_csv = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
    test_ids = test_csv["image_name"].astype(str).values

print("Computing predictions with model.predict (streamed) ...")
preds = model.predict(
    test_ds.map(lambda x, y: x, num_parallel_calls=AUTO),
    verbose=0,
)
probabilities = preds.reshape(-1)
print("Predictions:", probabilities.shape, "IDs:", test_ids.shape)



## === cell 12
print("Generating submission.csv file...")
preds = probabilities.astype(np.float32)
preds = np.clip(preds, 0.0, 1.0)

if len(preds) != len(test_ids):
    min_len = min(len(preds), len(test_ids))
    preds = preds[:min_len]
    test_ids = test_ids[:min_len]

pred_df = pd.DataFrame({"image_name": test_ids, "target": preds})
print("Pred df shape:", pred_df.shape)
print(pred_df.head())

sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub_out["target"].isna().any():
    fill_val = float(np.nanmean(pred_df["target"].values)) if len(pred_df) else 0.5
    sub_out["target"] = sub_out["target"].fillna(fill_val)

sub_out = sub_out[["image_name", "target"]]
assert (
    sub_out.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission"

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
print("submission.csv columns:", list(sub_out.columns))
print("Missing targets after merge:", int(sub_out["target"].isna().sum()))

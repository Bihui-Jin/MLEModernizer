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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

print("TF version:", tf.__version__)
print("Working dir:", os.getcwd())



## === cell 1
for p in [
    "/kaggle/input/submissionb3/submissionB3.csv",
    "/kaggle/input/submissionb7/submissionB7-2.csv",
    "/kaggle/input/subeffnetb0/submissionB0.csv",
]:
    print(p, "exists?", os.path.exists(p))



## === cell 2
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



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

INPUT_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "jpeg", "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "jpeg", "test")

TFREC_DIR = os.path.join(INPUT_DIR, "tfrecords")
TRAIN_TFRECS = sorted(
    [
        os.path.join(TFREC_DIR, f)
        for f in os.listdir(TFREC_DIR)
        if f.startswith("train") and f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TFREC_DIR, f)
        for f in os.listdir(TFREC_DIR)
        if f.startswith("test") and f.endswith(".tfrec")
    ]
)

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [384, 384]

print("TRAIN_CSV exists?", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists?", os.path.exists(TEST_CSV))
print("TFREC_DIR exists?", os.path.exists(TFREC_DIR))
print("Num train tfrecs:", len(TRAIN_TFRECS), "Num test tfrecs:", len(TEST_TFRECS))
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE)



## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3



## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
sub.head()



## === cell 6
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "test:", test.shape)
train.head()



## === cell 7
print(train["target"].value_counts(dropna=False))
print("pos rate:", train["target"].mean())



## === cell 8
test_ids = test["image_name"].astype(str).values

NUM_TRAINING_IMAGES = len(train)
NUM_TEST_IMAGES = len(test)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    "NUM_TRAINING_IMAGES:",
    NUM_TRAINING_IMAGES,
    "NUM_TEST_IMAGES:",
    NUM_TEST_IMAGES,
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
)

CLASSES = [0, 1]



## === cell 9
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.optimizer.set_jit(True)




## === cell 10
def _dataset_options():
    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.threading.private_threadpool_size = max(8, os.cpu_count() or 8)
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    return opts


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),  # train only
    "image_name": tf.io.FixedLenFeature([], tf.string),  # present in both train/test
}


def _decode_image_from_tfrecord(example):
    img = tf.image.decode_jpeg(example["image"], channels=3)
    img = tf.image.resize(img, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    img = _decode_image_from_tfrecord(ex)
    lab = tf.cast(ex["target"], tf.int32)
    return img, lab


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    img = _decode_image_from_tfrecord(ex)
    name = ex["image_name"]
    return img, name


def get_training_dataset():
    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTO)
    ds = ds.with_options(_dataset_options())
    ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.map(data_augment, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTO)
    ds = ds.with_options(_dataset_options())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


print(f"Dataset: {NUM_TRAINING_IMAGES} training rows, {NUM_TEST_IMAGES} test rows")




## === cell 11
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.0001,
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




## === cell 12
with strategy.scope():
    backbone_b6 = tf.keras.applications.EfficientNetB6(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    backbone_b6.trainable = False

    model = tf.keras.Sequential(
        [
            backbone_b6,
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 13
with strategy.scope():
    backbone_b3 = tf.keras.applications.EfficientNetB3(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    backbone_b3.trainable = False

    model2 = tf.keras.Sequential(
        [
            backbone_b3,
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dropout(0.3),
            L.Dense(256, activation="relu"),
            L.Dropout(0.25),
            L.Dense(128, activation="relu"),
            L.Dropout(0.2),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model2.summary()



## === cell 14
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)



## === cell 15
train_ds = get_training_dataset()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=1,
)



## === cell 16
history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=1,
)



## === cell 17
test_ds = get_test_dataset(ordered=True)
print("Test batches:", int(np.ceil(NUM_TEST_IMAGES / BATCH_SIZE)))



## === cell 18
print("Computing predictions...")

test_images_ds = test_ds.map(
    lambda image, name: image, num_parallel_calls=AUTO, deterministic=True
)

prob1 = model.predict(test_images_ds, verbose=1).reshape(-1)
prob2 = model2.predict(test_images_ds, verbose=1).reshape(-1)

probabilities = 0.5 * prob1 + 0.5 * prob2
probabilities = probabilities[:NUM_TEST_IMAGES]  # safety

print(
    "Preds shape:",
    probabilities.shape,
    "min/max:",
    float(probabilities.min()),
    float(probabilities.max()),
)



## === cell 19
final_image_names = test_ids

pred_df = pd.DataFrame(
    {
        "image_name": final_image_names.astype("U"),
        "target": probabilities.astype(np.float32),
    }
)
pred_df.head()



## === cell 20
print("Generating submission.csv file...")

subm = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if subm["target"].isna().any():
    fill_value = float(pred_df["target"].mean())
    subm["target"] = subm["target"].fillna(fill_value)

subm["target"] = subm["target"].astype(np.float32)
subm["target"] = np.clip(subm["target"].values, 0.0, 1.0).astype(np.float32)

assert (
    subm.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert list(subm.columns) == ["image_name", "target"], "Submission columns mismatch."
assert np.isfinite(subm["target"].values).all(), "Non-finite predictions in submission."

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", subm.shape)
print(subm.head())



## === cell 21
sub1 = sub.copy()
del sub1["target"]
sub1 = sub1.merge(pred_df, on="image_name", how="left")
if sub1["target"].isna().any():
    sub1["target"] = sub1["target"].fillna(float(pred_df["target"].mean()))
sub1.to_csv("submission-sample.csv", index=False)
print("Wrote submission-sample.csv:", sub1.shape)
sub1.head()



## === cell 22
print("Done.")



## === cell 23
sub_es = subm.copy()
sub_es.to_csv("submission.csv", index=False)
print("Final submission.csv written:", sub_es.shape)



## === cell 24
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved models.")

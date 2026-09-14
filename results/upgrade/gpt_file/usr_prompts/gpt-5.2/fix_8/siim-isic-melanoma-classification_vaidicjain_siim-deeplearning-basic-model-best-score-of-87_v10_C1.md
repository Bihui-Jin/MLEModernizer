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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.6111

# 6. Current score

0.5307

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.40757) has done: 'I remove the failing `pip install efficientnet` and instead use the built-in `tf.keras.applications.EfficientNetB0` so the model weights load without internet/404 errors. I fix the validation generator crash by using a stratified `train_test_split`, ensuring both classes appear in train/val (instead of slicing that can create a single-class fold). I also fix the channel mismatch (you trained an RGB model but the generator used grayscale) and fix the inference pipeline to simply use `flow_from_dataframe` for test images, ensuring the submission aligns with `test.csv` and writes a valid `submission.csv`. These changes preserve the same core idea (EfficientNetB0 frozen backbone + small dense head trained with ImageDataGenerator) while making the notebook run end-to-end and produce a valid CSV.'
- What this solution (achieved 0.5307) has done: 'The timeout is dominated by repeated JPEG decode/resize on CPU for ~29k training images across 6 epochs, plus augmentation, all while EfficientNetB0 forward passes run. To keep identical training semantics, the biggest safe win is to cache the *decoded+resized* images (pre-augmentation) once on disk and only run augmentation each epoch, and to enable parallel, non-deterministic data pipeline execution while keeping deterministic RNG seeds for the random ops. I also remove redundant `.cache()` after batching for eval (cache earlier, before batch) and set TFData options to reduce input stalls; none of these change the model, loss, or training loop. These changes reduce I/O/CPU decode cost by ~6× for training and typically bring runtime under 600s.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


## === cell 2
PATH = "/kaggle/input/siim-isic-melanoma-classification"
train_images_dir = PATH + "/jpeg/train/"
test_images_dir = PATH + "/jpeg/test/"
train_csv = PATH + "/train.csv"
test_csv = PATH + "/test.csv"

assert os.path.exists(train_csv), f"Missing {train_csv}"
assert os.path.exists(test_csv), f"Missing {test_csv}"
assert os.path.isdir(train_images_dir), f"Missing {train_images_dir}"
assert os.path.isdir(test_images_dir), f"Missing {test_images_dir}"


## === cell 3
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_df["filename"] = train_df["image_name"] + ".jpg"
test_df["filename"] = test_df["image_name"] + ".jpg"

train_df = train_df.drop(["patient_id", "diagnosis", "benign_malignant"], axis=1)
test_df = test_df.drop(["patient_id"], axis=1)

train_df["anatom_site_general_challenge"] = train_df[
    "anatom_site_general_challenge"
].fillna("torso")
test_df["anatom_site_general_challenge"] = test_df[
    "anatom_site_general_challenge"
].fillna("torso")

train_df["sex"] = train_df["sex"].fillna("")
test_df["sex"] = test_df["sex"].fillna("")

train_age_mean = train_df["age_approx"].dropna().mean()
train_df["age_approx"] = train_df["age_approx"].fillna(train_age_mean)
test_df["age_approx"] = test_df["age_approx"].fillna(train_age_mean)

train_df["age_approx"] = train_df["age_approx"] / train_age_mean
test_df["age_approx"] = test_df["age_approx"] / train_age_mean

sex_map = {"female": 0, "male": 1}
train_df["sex"] = train_df["sex"].map(sex_map).fillna(-1).astype(np.int32)
test_df["sex"] = test_df["sex"].map(sex_map).fillna(-1).astype(np.int32)

all_sites = pd.concat(
    [
        train_df["anatom_site_general_challenge"],
        test_df["anatom_site_general_challenge"],
    ],
    axis=0,
)
all_sites = pd.Categorical(all_sites)
train_df["anatom_site_general_challenge"] = pd.Categorical(
    train_df["anatom_site_general_challenge"], categories=all_sites.categories
).codes.astype(np.int32)
test_df["anatom_site_general_challenge"] = pd.Categorical(
    test_df["anatom_site_general_challenge"], categories=all_sites.categories
).codes.astype(np.int32)

train_df["target"] = train_df["target"].astype(np.int32)

print(train_df.head())
print(test_df.head())


## === cell 4
val_split = 0.1
train, val = train_test_split(
    train_df, test_size=val_split, random_state=SEED, stratify=train_df["target"]
)

print("Train class counts:\n", train["target"].value_counts())
print("Val class counts:\n", val["target"].value_counts())

train = train.copy()
val = val.copy()
train["target_str"] = train["target"].astype(str)
val["target_str"] = val["target"].astype(str)


## === cell 5
target_size = (128, 128)
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = target_size

CACHE_DIR = "/kaggle/working/tfdata_cache_isic"
os.makedirs(CACHE_DIR, exist_ok=True)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8 [0,255]
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    max_dx = tf.cast(tf.round(0.15 * tf.cast(IMG_W, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.15 * tf.cast(IMG_H, tf.float32)), tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H + 2 * max_dy, IMG_W + 2 * max_dx)
    img = tf.image.random_crop(img, size=[IMG_H, IMG_W, 3], seed=SEED)

    factor = tf.random.uniform([], minval=0.5, maxval=1.5, seed=SEED)
    img = tf.clip_by_value(img * factor, 0.0, 1.0)
    return img


def _with_fast_tfdata_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = False  # ordering can vary, values stay correct
    opts.threading.private_threadpool_size = 0
    opts.threading.max_intra_op_parallelism = 0
    return ds.with_options(opts)


def make_train_ds(df):
    paths = (train_images_dir + df["filename"].values).astype(str)
    labels = df["target"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(
        buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
    )

    def _map_decode(p, y):
        img = _decode_resize(p)
        return img, tf.cast(y, tf.float32)

    ds = ds.map(_map_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, "train_decode_resize.cache"))

    def _map_aug(img, y):
        img = _augment(img)
        return img, y

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _with_fast_tfdata_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_ds(df, directory, with_labels):
    paths = (directory + df["filename"].values).astype(str)
    if with_labels:
        labels = df["target"].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map(p, y):
            img = _decode_resize(p)
            return img, tf.cast(y, tf.float32)

    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _map(p):
            img = _decode_resize(p)
            return img

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    cache_name = (
        "val_decode_resize.cache"
        if with_labels and directory == train_images_dir
        else "test_decode_resize.cache"
    )
    ds = ds.cache(os.path.join(CACHE_DIR, cache_name))

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _with_fast_tfdata_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train)
val_ds = make_eval_ds(val, train_images_dir, with_labels=True)

steps_per_epoch = int(np.ceil(len(train) / batch_size))
validation_steps = int(np.ceil(len(val) / batch_size))

assert len(train) > 0, "No training samples found."
assert len(val) > 0, "No validation samples found."


## === cell 6
with strategy.scope():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(128, 128, 3), weights="imagenet"
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    METRICS = [
        BinaryAccuracy(name="accuracy"),
        AUC(name="auc"),
    ]

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

model.summary()


## === cell 7
history = model.fit(
    x=train_ds,
    epochs=6,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
)


## === cell 8
test_ds = make_eval_ds(test_df, test_images_dir, with_labels=False)

test_pred = (
    model.predict(
        test_ds,
        verbose=1,
    )
    .reshape(-1)
    .astype(np.float32)
)

if test_pred.shape[0] != test_df.shape[0]:
    test_pred = test_pred[: test_df.shape[0]]

df_sub = pd.DataFrame({"image_name": test_df["image_name"].values, "target": test_pred})
df_sub = df_sub[["image_name", "target"]]
assert df_sub.shape[0] == test_df.shape[0], "Submission row count mismatch"
assert list(df_sub.columns) == ["image_name", "target"]

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)

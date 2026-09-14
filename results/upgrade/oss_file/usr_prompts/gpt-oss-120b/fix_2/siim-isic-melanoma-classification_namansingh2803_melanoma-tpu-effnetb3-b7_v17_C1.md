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

# 5. Target score

0.8936599643953211

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
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

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
GCS_PATH = (
    "/kaggle/input/siim-isic-melanoma-classification"  # local path, no auth needed
)

EPOCHS = 3
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [224, 224]  # reduced size for speed



## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 5
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_PATH, pre, file))




## === cell 6
sub = pd.read_csv(os.path.join(GCS_PATH, "sample_submission.csv"))



## === cell 7
train = pd.read_csv(os.path.join(GCS_PATH, "train.csv"))



## === cell 9
TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(GCS_PATH, "jpeg/train/*.jpg"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(GCS_PATH, "jpeg/test/*.jpg"))




## === cell 10
def transform_rotation(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2
    rotation = 15.0 * tf.random.normal([1], dtype="float32")
    rotation = math.pi * rotation / 180.0
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3]
    )
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])
    idx2 = tf.linalg.matvec(rotation_matrix, tf.cast(idx, dtype="float32"))
    idx2 = tf.cast(idx2, dtype="int32")
    idx2 = tf.clip_by_value(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def transform_shear(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2
    shear = 5.0 * tf.random.normal([1], dtype="float32")
    shear = math.pi * shear / 180.0
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3]
    )
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])
    idx2 = tf.linalg.matvec(shear_matrix, tf.cast(idx, dtype="float32"))
    idx2 = tf.cast(idx2, dtype="int32")
    idx2 = tf.clip_by_value(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def transform_shift(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2
    height_shift = 16.0 * tf.random.normal([1], dtype="float32")
    width_shift = 16.0 * tf.random.normal([1], dtype="float32")
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    shift_matrix = tf.reshape(
        tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0
        ),
        [3, 3],
    )
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])
    idx2 = tf.linalg.matvec(shift_matrix, tf.cast(idx, dtype="float32"))
    idx2 = tf.cast(idx2, dtype="int32")
    idx2 = tf.clip_by_value(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def transform_zoom(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2
    height_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    width_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    zoom_matrix = tf.reshape(
        tf.concat(
            [
                one / height_zoom,
                zero,
                zero,
                zero,
                one / width_zoom,
                zero,
                zero,
                zero,
                one,
            ],
            axis=0,
        ),
        [3, 3],
    )
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])
    idx2 = tf.linalg.matvec(zoom_matrix, tf.cast(idx, dtype="float32"))
    idx2 = tf.cast(idx2, dtype="int32")
    idx2 = tf.clip_by_value(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])




## === cell 11
def data_augment_chaotic(image, label):
    p_spatial = tf.random.uniform([], 0, 1, dtype="float32")
    p_spatial2 = tf.random.uniform([], 0, 1, dtype="float32")
    p_pixel = tf.random.uniform([], 0, 1, dtype="float32")
    p_crop = tf.random.uniform([], 0, 1, dtype="float32")

    if p_spatial >= 0.2:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    if p_crop >= 0.7:
        if p_crop >= 0.95:
            sz = 0.6
        elif p_crop >= 0.85:
            sz = 0.7
        elif p_crop >= 0.8:
            sz = 0.8
        else:
            sz = 0.9
        image = tf.image.random_crop(
            image, size=[int(HEIGHT * sz), int(WIDTH * sz), CHANNELS]
        )
        image = tf.image.resize(image, [HEIGHT, WIDTH])

    if p_spatial2 >= 0.6:
        if p_spatial2 >= 0.9:
            image = transform_rotation(image)
        elif p_spatial2 >= 0.8:
            image = transform_zoom(image)
        elif p_spatial2 >= 0.7:
            image = transform_shift(image)
        else:
            image = transform_shear(image)

    if p_pixel >= 0.4:
        if p_pixel >= 0.85:
            image = tf.image.random_saturation(image, lower=0, upper=2)
        elif p_pixel >= 0.65:
            image = tf.image.random_contrast(image, lower=0.8, upper=2)
        elif p_pixel >= 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)
        else:
            image = tf.image.adjust_gamma(image, gamma=0.6)

    return image, label




## === cell 12
train_df, val_df = train_test_split(
    train, test_size=0.1, stratify=train["target"], random_state=42
)


def make_path(df, split):
    return (
        df["image_name"]
        .apply(lambda x: os.path.join(GCS_PATH, "jpeg", split, f"{x}.jpg"))
        .tolist()
    )


train_paths = make_path(train_df, "train")
val_paths = make_path(val_df, "train")
train_labels = train_df["target"].astype("float32").tolist()
val_labels = val_df["target"].astype("float32").tolist()


def path_label_dataset(paths, labels, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTO)
    if augment:
        ds = ds.map(data_augment_chaotic, num_parallel_calls=AUTO)
    ds = ds.shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO)
    return ds


def get_training_dataset():
    return path_label_dataset(train_paths, train_labels, augment=True)


def get_validation_dataset():
    return path_label_dataset(val_paths, val_labels, augment=False)


test_df = pd.read_csv(os.path.join(GCS_PATH, "test.csv"))
test_paths = make_path(test_df, "test")


def test_dataset():
    ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = ds.map(_load, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds


NUM_TRAINING_IMAGES = len(train_paths)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(f"Dataset: {NUM_TRAINING_IMAGES} training images")




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1445682531.py in <cell line: 0>()
      1 # split train dataframe into train/val
----> 2 train_df, val_df = train_test_split(
      3     train, test_size=0.1, stratify=train["target"], random_state=42
      4 )
      5 

NameError: name 'train_test_split' is not defined

## === cell 13
def build_lrfn(
    lr_start=1e-5,
    lr_max=1e-4,
    lr_min=1e-6,
    lr_rampup_epochs=2,
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




## === cell 14
with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            efn.EfficientNetB0(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4040590357.py in <cell line: 0>()
      2     model2 = tf.keras.Sequential(
      3         [
----> 4             efn.EfficientNetB0(
      5                 input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
      6             ),

NameError: name 'efn' is not defined

## === cell 15
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

history2 = model2.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=get_validation_dataset(),
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2402958332.py in <cell line: 0>()
      2 lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
      3 
----> 4 history2 = model2.fit(
      5     get_training_dataset(),
      6     epochs=EPOCHS,

NameError: name 'model2' is not defined

## === cell 16
test_ds = test_dataset()
print("Computing predictions...")
probabilities2 = model2.predict(test_ds, verbose=0).flatten()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3067420387.py in <cell line: 0>()
----> 1 test_ds = test_dataset()
      2 print("Computing predictions...")
      3 probabilities2 = model2.predict(test_ds, verbose=0).flatten()
      4 

NameError: name 'test_dataset' is not defined

## === cell 17
print("Generating submission.csv file...")
pred_df = pd.DataFrame({"image_name": test_df["image_name"], "target": probabilities2})
submission_path = "submission_EffnetB0.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/963813390.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
----> 2 pred_df = pd.DataFrame({"image_name": test_df["image_name"], "target": probabilities2})
      3 submission_path = "submission_EffnetB0.csv"
      4 pred_df.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}")

NameError: name 'test_df' is not defined

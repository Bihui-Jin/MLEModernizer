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

0.65363

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.65363) has done: 'I fix the protobuf import issue, replace the unavailable EfficientNet wrapper with the built‑in Keras version, and adjust the model definition accordingly. These changes resolve the import errors and ensure all variables (e.g., `train_test_split`, `model2`, `test_dataset`, `test_df`) are properly defined so the script runs end‑to‑end and writes a valid `submission_EffnetB0.csv` file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE
GCS_PATH = (
    "/kaggle/input/siim-isic-melanoma-classification"  # local path, no auth needed
)

EPOCHS = 3
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [224, 224]  # reduced size for speed




## === cell 3
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 4
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_PATH, pre, file))




## === cell 5
sub = pd.read_csv(os.path.join(GCS_PATH, "sample_submission.csv"))




## === cell 6
train = pd.read_csv(os.path.join(GCS_PATH, "train.csv"))




## === cell 7
TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(GCS_PATH, "jpeg/train/*.jpg"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(GCS_PATH, "jpeg/test/*.jpg"))




## === cell 8
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




## === cell 9
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




## === cell 10
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




## === cell 11
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




## === cell 12
with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            EfficientNetB0(
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




## === cell 13
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

history2 = model2.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=get_validation_dataset(),
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4172603844.py in <cell line: 0>()
      3 
      4 history2 = model2.fit(
----> 5     get_training_dataset(),
      6     epochs=EPOCHS,
      7     callbacks=[lr_schedule],

/tmp/ipykernel_11/2858258874.py in get_training_dataset()
     36 
     37 def get_training_dataset():
---> 38     return path_label_dataset(train_paths, train_labels, augment=True)
     39 
     40 

/tmp/ipykernel_11/2858258874.py in path_label_dataset(paths, labels, augment)
     30     ds = ds.map(_load, num_parallel_calls=AUTO)
     31     if augment:
---> 32         ds = ds.map(data_augment_chaotic, num_parallel_calls=AUTO)
     33     ds = ds.shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO)
     34     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_file4qjydnmm.py in tf__data_augment_chaotic(image, label)
    156                     nonlocal image
    157                     pass
--> 158                 ag__.if_stmt(ag__.ld(p_spatial2) >= 0.6, if_body_8, else_body_8, get_state_8, set_state_8, ('image',), 1)
    159 
    160                 def get_state_12():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1213   # Note: tf.cond doesn't support SparseTensor.
   1214   if tensors.is_dense_tensor(cond):
-> 1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
   1217     _py_if_stmt(cond, body, orelse)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1259     return new_orelse_vars
   1260 
-> 1261   final_cond_vars = tf_cond.cond(
   1262       cond, aug_body, aug_orelse, strict=True)
   1263   final_cond_vars = final_cond_vars + init_vars[nouts:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file4qjydnmm.py in if_body_8()
    151                             ag__.if_stmt(ag__.ld(p_spatial2) >= 0.7, if_body_5, else_body_5, get_state_5, set_state_5, ('image',), 1)
    152                         ag__.if_stmt(ag__.ld(p_spatial2) >= 0.8, if_body_6, else_body_6, get_state_6, set_state_6, ('image',), 1)
--> 153                     ag__.if_stmt(ag__.ld(p_spatial2) >= 0.9, if_body_7, else_body_7, get_state_7, set_state_7, ('image',), 1)
    154 
    155                 def else_body_8():

/tmp/__autograph_generated_file4qjydnmm.py in if_body_7()
    116                     def if_body_7():
    117                         nonlocal image
--> 118                         image = ag__.converted_call(ag__.ld(transform_rotation), (ag__.ld(image),), None, fscope)
    119 
    120                     def else_body_7():

/tmp/__autograph_generated_filecc8kw89b.py in tf__transform_rotation(image)
     21                 z = ag__.converted_call(ag__.ld(tf).ones, ([ag__.ld(DIM) * ag__.ld(DIM)],), dict(dtype='int32'), fscope)
     22                 idx = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(x), ag__.ld(y), ag__.ld(z)],), None, fscope)
---> 23                 idx2 = ag__.converted_call(ag__.ld(tf).linalg.matvec, (ag__.ld(rotation_matrix), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(idx),), dict(dtype='float32'), fscope)), None, fscope)
     24                 idx2 = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(idx2),), dict(dtype='int32'), fscope)
     25                 idx2 = ag__.converted_call(ag__.ld(tf).clip_by_value, (ag__.ld(idx2), -ag__.ld(DIM) // 2 + ag__.ld(XDIM) + 1, ag__.ld(DIM) // 2), None, fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/2968636675.py", line 27, in data_augment_chaotic  *
        image = transform_rotation(image)
    File "/tmp/ipykernel_11/689329623.py", line 17, in transform_rotation  *
        idx2 = tf.linalg.matvec(rotation_matrix, tf.cast(idx, dtype="float32"))

    ValueError: Dimensions must be equal, but are 3 and 50176 for '{{node cond_2/cond/MatVec/MatMul}} = BatchMatMulV2[T=DT_FLOAT, adj_x=false, adj_y=false, grad_x=false, grad_y=false](cond_2/cond/Reshape, cond_2/cond/MatVec/ExpandDims)' with input shapes: [3,3], [3,50176,1].


## === cell 14
test_ds = test_dataset()
print("Computing predictions...")
probabilities2 = model2.predict(test_ds, verbose=0).flatten()




## === cell 15
print("Generating submission.csv file...")
pred_df = pd.DataFrame({"image_name": test_df["image_name"], "target": probabilities2})
submission_path = "submission_EffnetB0.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

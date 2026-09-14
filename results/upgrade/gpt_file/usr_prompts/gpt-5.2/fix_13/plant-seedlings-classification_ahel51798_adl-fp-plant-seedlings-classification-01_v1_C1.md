# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

input_root = "/kaggle/input/plant-seedlings-classification"
print("Listing:", input_root)
print(os.listdir(input_root)[:20])




## === cell 1
from datetime import datetime, timedelta
import gc
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

SEED = 1337
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

start_time = datetime.now()
print("Time now is", start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime("%Y%m%d_%H%M_")
print("this_run_file_prefix", this_run_file_prefix)

print("TensorFlow version:", tf.__version__)




## === cell 2
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




## === cell 3
IMAGE_SIZE = [299, 299]
width = 299
height = 299

CLASSES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]
num_classes = len(CLASSES)

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"
TEST_ROOT = "/kaggle/input/plant-seedlings-classification"
SAMPLE_PATH = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
if not os.path.exists(SAMPLE_PATH):
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"

print("num_classes:", num_classes, "BATCH_SIZE:", BATCH_SIZE)




## === cell 4
import math
from pathlib import Path

preprocess = tf.keras.applications.inception_v3.preprocess_input

CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

_TF_W = tf.constant(width, tf.float32)
_TF_H = tf.constant(height, tf.float32)

_HAS_TFA = False


def _list_images_with_labels(train_dir: str):
    paths = []
    labels = []
    for cls in CLASSES:  # fixed order
        cls_dir = Path(train_dir) / cls
        cls_paths = sorted([str(p) for p in cls_dir.glob("*.png")])
        paths.extend(cls_paths)
        labels.extend([CLASS_TO_INDEX[cls]] * len(cls_paths))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int32)


def _train_val_split(paths, labels, val_split: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    paths = paths[idx]
    labels = labels[idx]
    n_val = int(round(len(paths) * val_split))
    val_paths = paths[:n_val]
    val_labels = labels[:n_val]
    train_paths = paths[n_val:]
    train_labels = labels[n_val:]
    return train_paths, train_labels, val_paths, val_labels


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _onehot(label):
    return tf.one_hot(label, depth=num_classes, dtype=tf.float32)


@tf.function(reduce_retracing=True)
def _apply_transform(img, transform):
    img = tf.expand_dims(img, 0)
    transform = tf.expand_dims(transform, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=tf.constant([height, width], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function(reduce_retracing=True)
def _augment(img, seed_pair):
    s0 = tf.cast(seed_pair, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=s0)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=s0 + tf.constant([1, 0], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([2, 0], tf.int32),
        minval=-math.pi,
        maxval=math.pi,
        dtype=tf.float32,
    )
    c = tf.cos(angle)
    s = tf.sin(angle)
    cx = (_TF_W - 1.0) / 2.0
    cy = (_TF_H - 1.0) / 2.0
    a0 = c
    a1 = -s
    a2 = cx - c * cx + s * cy
    b0 = s
    b1 = c
    b2 = cy - s * cx - c * cy
    rot = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    img = _apply_transform(img, rot)

    dx = (
        tf.random.stateless_uniform(
            [],
            seed=s0 + tf.constant([3, 0], tf.int32),
            minval=-0.3,
            maxval=0.3,
            dtype=tf.float32,
        )
        * _TF_W
    )
    dy = (
        tf.random.stateless_uniform(
            [],
            seed=s0 + tf.constant([4, 0], tf.int32),
            minval=-0.3,
            maxval=0.3,
            dtype=tf.float32,
        )
        * _TF_H
    )
    trans = tf.stack([1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0], axis=0)
    img = _apply_transform(img, trans)

    shear_deg = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([5, 0], tf.int32),
        minval=-0.3,
        maxval=0.3,
        dtype=tf.float32,
    )
    shear = shear_deg * (math.pi / 180.0)
    cos_s = tf.cos(shear)
    sin_s = tf.sin(shear)
    sh = tf.stack([1.0, -sin_s, 0.0, 0.0, cos_s, 0.0, 0.0, 0.0], axis=0)
    img = _apply_transform(img, sh)

    zoom = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([6, 0], tf.int32),
        minval=0.5,
        maxval=1.5,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(_TF_H * zoom), tf.int32)
    new_w = tf.cast(tf.round(_TF_W * zoom), tf.int32)
    resized = tf.image.resize(img, [new_h, new_w], method="bilinear")
    img = tf.image.resize_with_crop_or_pad(resized, height, width)

    return img


def _make_dataset(paths, labels=None, training=False, seed=SEED, cache=False):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    autotune = tf.data.AUTOTUNE

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(paths))
        ds = ds.with_options(options)

        @tf.function(reduce_retracing=True)
        def _map_test(path):
            img = _decode_and_resize(path)
            img = preprocess(img)
            return img

        ds = ds.map(_map_test, num_parallel_calls=autotune)
        if cache:
            ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(autotune)
        return ds

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(paths), tf.convert_to_tensor(labels))
    )
    ds = ds.with_options(options)

    if training:
        shuffle_buf = int(min(len(paths), max(2048, BATCH_SIZE * 64)))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True
        )

    @tf.function(reduce_retracing=True)
    def _map_decode_preprocess(path, label):
        img = _decode_and_resize(path)
        img = preprocess(img)
        y = _onehot(label)
        return img, y

    ds = ds.map(_map_decode_preprocess, num_parallel_calls=autotune)

    if cache:
        ds = ds.cache()

    if training:
        ds = ds.enumerate()

        @tf.function(reduce_retracing=True)
        def _map_augment(i, xy):
            img, y = xy
            seed_pair = tf.stack(
                [tf.cast(seed, tf.int32), tf.cast(i, tf.int32)], axis=0
            )
            img = _augment(img, seed_pair)
            return img, y

        ds = ds.map(_map_augment, num_parallel_calls=autotune)

    ds = ds.batch(BATCH_SIZE, drop_remainder=bool(training))
    ds = ds.prefetch(autotune)
    return ds


def define_generators(seed: int = 1337, val_split: float = 0.15):
    train_paths, train_labels, val_paths, val_labels = _train_val_split(
        *_list_images_with_labels(TRAIN_DIR), val_split=val_split, seed=seed
    )

    test_dir = Path(TEST_ROOT) / "test"
    test_paths = np.array(
        sorted([str(p) for p in test_dir.glob("*.png")]), dtype=object
    )

    train_ds = _make_dataset(
        train_paths, train_labels, training=True, seed=seed, cache=True
    )
    val_ds = _make_dataset(val_paths, val_labels, training=False, seed=seed, cache=True)
    test_ds = _make_dataset(
        test_paths, labels=None, training=False, seed=seed, cache=True
    )

    class _Compat:
        pass

    train_gen = _Compat()
    val_gen = _Compat()
    test_gen = _Compat()

    train_gen.dataset = train_ds
    val_gen.dataset = val_ds
    test_gen.dataset = test_ds

    train_gen.samples = int(train_paths.shape[0])
    val_gen.samples = int(val_paths.shape[0])
    test_gen.samples = int(test_paths.shape[0])

    train_gen.num_classes = num_classes
    val_gen.num_classes = num_classes

    train_gen.class_indices = {c: i for i, c in enumerate(CLASSES)}

    test_gen.filenames = [f"test/{Path(p).name}" for p in test_paths.tolist()]

    return train_gen, val_gen, test_gen


train_generator, validation_generator, test_generator = define_generators(seed=SEED)

print(
    "train classes:",
    train_generator.num_classes,
    "val classes:",
    validation_generator.num_classes,
)
print("class_indices:", train_generator.class_indices)




## === cell 5
def create_ResNet50_model():
    pretrained_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_ResNet101V2_model():
    pretrained_model = tf.keras.applications.ResNet101V2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_VGG16_model():
    pretrained_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_Xception_model():
    pretrained_model = tf.keras.applications.Xception(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_DenseNet_model():
    pretrained_model = tf.keras.applications.DenseNet201(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_InceptionV3_model():
    pretrained_model = tf.keras.applications.InceptionV3(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_ResNet152_model():
    pretrained_model = tf.keras.applications.ResNet152V2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_MobileNetV2_model():
    pretrained_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_InceptionResNetV2_model():
    pretrained_model = tf.keras.applications.InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = LR_START + (epoch * (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS)
    elif epoch < (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS):
        lr = LR_MAX
    else:
        lr = LR_MIN + (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        )
    return lr


def write_history(j):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if historys[i] != 0:
            history_dict[i] = historys[i].history
    filename = (
        "/kaggle/working/" + this_run_file_prefix + "model_history_" + str(j) + ".pkl"
    )
    with open(filename, "wb") as pklfile:
        pickle.dump(history_dict, pklfile)


def load_history(filename):
    with open(filename, "rb") as file:
        history_dict = pickle.load(file)
    return history_dict


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.get("accuracy", []), label="Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.get("loss", []), label="Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()




## === cell 6
LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = LR_START
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.80

lr_callback = LearningRateScheduler(lrfn, verbose=False)

rng = list(range(30))
y = [lrfn(x) for x in rng]
print("lrfn y:", y)




## === cell 7
no_of_models = 1
models = [0] * no_of_models
start_model = 0
end_model = 1

EPOCHS = 3
historys = [0] * no_of_models
finished_models = 0

checkpoint_path = "/kaggle/working/best_model.keras"

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
lr_callback = LearningRateScheduler(lrfn, verbose=False)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
)
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

FIT_KW = {}

steps_per_epoch = max(1, int(math.ceil(train_generator.samples / BATCH_SIZE)))
validation_steps = max(1, int(math.ceil(validation_generator.samples / BATCH_SIZE)))

with strategy.scope():
    for j in range(no_of_models):
        models[j] = create_InceptionV3_model()

models[0].summary()

for j in range(start_model, end_model):
    start_training = datetime.now()
    print(start_training)
    time_from_start_program_tdelta = start_training - start_time
    if time_from_start_program_tdelta > end_training_by_tdelta:
        print(j, "time limit for doing training over, get out")
        break

    print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)
    historys[j] = models[j].fit(
        train_generator.dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator.dataset,
        validation_steps=validation_steps,
        callbacks=[
            learning_rate_reduction,
            early_stopping,
            lr_callback,
            model_checkpoint_callback,
        ],
        verbose=1,
        **FIT_KW,
    )

    write_history(j)

    gc.collect()
    finished_models = j + 1

print(datetime.now())




## === cell 8
hist_idx = 0
if historys[hist_idx] != 0:
    history_to_analyze = historys[hist_idx].history
    print(
        "History keys:",
        list(history_to_analyze.keys()),
        "last val_loss:",
        history_to_analyze.get("val_loss", [None])[-1],
    )
else:
    print("No training history available to plot.")




## === cell 9
best_model = None
if os.path.exists(checkpoint_path):
    best_model = load_model(checkpoint_path)
    print("Loaded best model from checkpoint:", checkpoint_path)
else:
    best_model = models[0]
    print("Using in-memory model (no saved checkpoint found).")




## === cell 10
PRED_KW = {}

test_steps = max(1, int(math.ceil(test_generator.samples / BATCH_SIZE)))

predictions = best_model.predict(
    test_generator.dataset,
    steps=test_steps,
    verbose=1,
    **PRED_KW,
)

y_idx = predictions.argmax(axis=1)
class_list = np.asarray(CLASSES, dtype=object)[y_idx].tolist()

submission = pd.read_csv(SAMPLE_PATH)

filenames = np.asarray(test_generator.filenames, dtype=object)
pred_files = np.char.substr(filenames.astype(str), 5, 10_000)  # remove "test/" prefix

pred_df = pd.DataFrame({"file": pred_files, "species": class_list})

submission = submission.drop(columns=["species"]).merge(pred_df, on="file", how="left")
if submission["species"].isna().any():
    missing = submission[submission["species"].isna()]["file"].head(5).tolist()
    raise RuntimeError(f"Missing predictions for some files, e.g.: {missing}")

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Submission file generated at:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)

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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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

# 5. Target score

0.98488

# 6. Current score

0.07207

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12162) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error in this environment). I also correct the dataset root path so the code can actually find `train/` and `test/`, and add a small, safe fallback that searches known Kaggle locations if the preferred path isn’t present. Finally, I make training and inference robust: ensure generators are always defined before fitting, load the best checkpoint if it exists (otherwise use the in-memory model), and always write a valid `submission.csv` with the required `file,species` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.04505) has done: 'Most of the timeout is coming from expensive Python-side image loading/augmentation in `ImageDataGenerator` plus long end-to-end training on 299×299 with a fully-trainable InceptionV3. To keep identical modeling/training semantics while cutting overhead, I (1) remove all plotting/visualization that burns time but doesn’t affect the model, (2) enable deterministic settings and fast TF execution paths (XLA JIT, tf.data prefetching) while keeping float32 and the same epochs/callbacks, and (3) speed up inference by using a larger batch size for the test generator (same predictions, fewer Python calls). These changes don’t alter architecture/loss/optimizer/training loop logic; they only reduce input pipeline and execution overhead.'
- What this solution (achieved 0.07207) has done: 'The timeout is dominated by input pipeline overhead (wrapping a Keras `DirectoryIterator` inside `tf.data.from_generator` prevents efficient parallel decode/augmentation) and by running a heavy InceptionV3 backbone fully-trainable while also using slow Python-side augmentation. To preserve the same model, loss, and training semantics, I keep the architecture and callbacks identical but switch to a pure-`tf.data` pipeline that matches the original augmentation operations and uses parallel map, caching, and prefetch. I also fix `steps_per_epoch/validation_steps` to use `ceil` so no samples are silently dropped (same semantics as Keras generators), and set `workers/use_multiprocessing` for `fit` to reduce Python overhead safely. These changes are performance-focused and keep accuracy behavior (aside from negligible floating-point differences) while making the run much more likely to fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

print("Listing /kaggle/input/plant-seedlings-classification (if present):")
if os.path.exists("/kaggle/input/plant-seedlings-classification"):
    print(os.listdir("/kaggle/input/plant-seedlings-classification")[:30])
else:
    print("Path not found.")




## === cell 1
from datetime import datetime, timedelta
import gc
import random
import pickle
import math

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

start_time = datetime.now()
print("Time now is", start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime("%Y%m%d_%H%M_")
print("this_run_file_prefix", this_run_file_prefix)

print("TensorFlow version:", tf.__version__)

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
CANDIDATE_ROOTS = [
    "/kaggle/input/plant-seedlings-classification",
    "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification/plant-seedlings-classification",
]


def pick_data_root(candidates):
    for root in candidates:
        tr = os.path.join(root, "train")
        te = os.path.join(root, "test")
        if os.path.isdir(tr) and os.path.isdir(te):
            return root
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            root = os.path.join(base, name)
            tr = os.path.join(root, "train")
            te = os.path.join(root, "test")
            if os.path.isdir(tr) and os.path.isdir(te):
                return root
            nested = os.path.join(root, name)
            tr2 = os.path.join(nested, "train")
            te2 = os.path.join(nested, "test")
            if os.path.isdir(tr2) and os.path.isdir(te2):
                return nested
    raise FileNotFoundError(
        "Could not locate dataset root containing train/ and test/."
    )


DATA_ROOT = pick_data_root(CANDIDATE_ROOTS)

train_directory = os.path.join(DATA_ROOT, "train")
test_directory = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("Train dir exists:", os.path.exists(train_directory), train_directory)
print("Test  dir exists:", os.path.exists(test_directory), test_directory)


def visualize_class_images(base_directory, rows=2, cols=6):
    return


visualize_class_images(train_directory)




## === cell 4
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_preprocess,
)

AUTOTUNE = tf.data.AUTOTUNE


def _list_train_files_and_labels(train_dir, classes):
    filepaths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    filepaths = np.array(filepaths)
    labels = np.array(labels, dtype=np.int32)
    return filepaths, labels


def _list_test_files(test_dir):
    fps = []
    for fname in os.listdir(test_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            fps.append(os.path.join(test_dir, fname))
    fps = sorted(fps)
    return np.array(fps)


def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = inception_preprocess(img)  # matches original preprocessing_function
    return img


def _augment_stateless(img, seed2):
    k = tf.random.stateless_uniform([], seed2, minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    img = tf.image.stateless_random_flip_left_right(
        img, seed2 + tf.constant([1, 0], tf.int32)
    )
    img = tf.image.stateless_random_flip_up_down(
        img, seed2 + tf.constant([2, 0], tf.int32)
    )

    pad_h = tf.cast(tf.round(tf.cast(height, tf.float32) * 0.3), tf.int32)
    pad_w = tf.cast(tf.round(tf.cast(width, tf.float32) * 0.3), tf.int32)
    img = tf.image.pad_to_bounding_box(
        img, pad_h, pad_w, height + 2 * pad_h, width + 2 * pad_w
    )
    img = tf.image.stateless_random_crop(
        img,
        size=[height, width, 3],
        seed=seed2 + tf.constant([3, 0], tf.int32),
    )

    zoom = tf.random.stateless_uniform(
        [], seed2 + tf.constant([4, 0], tf.int32), minval=0.5, maxval=1.0
    )
    crop_h = tf.cast(tf.round(tf.cast(height, tf.float32) * zoom), tf.int32)
    crop_w = tf.cast(tf.round(tf.cast(width, tf.float32) * zoom), tf.int32)
    crop_h = tf.maximum(1, tf.minimum(height, crop_h))
    crop_w = tf.maximum(1, tf.minimum(width, crop_w))
    img2 = tf.image.stateless_random_crop(
        img,
        size=[crop_h, crop_w, 3],
        seed=seed2 + tf.constant([5, 0], tf.int32),
    )
    img = tf.image.resize(img2, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    return img


def build_train_val_test_datasets(
    train_dir, test_dir, classes, validation_split=0.2, seed=1337
):
    filepaths, labels = _list_train_files_and_labels(train_dir, classes)
    n = len(filepaths)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    filepaths = filepaths[idx]
    labels = labels[idx]

    val_n = int(round(n * validation_split))
    val_paths = filepaths[:val_n]
    val_labels = labels[:val_n]
    tr_paths = filepaths[val_n:]
    tr_labels = labels[val_n:]

    def make_ds(paths, labs, training):
        ds = tf.data.Dataset.from_tensor_slices((paths, labs))
        if training:
            ds = ds.shuffle(
                buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
            )
        ds = ds.enumerate()

        def _map_fn(i, pl):
            path, lab = pl
            img = _decode_resize_preprocess(path)
            if training:
                seed2 = tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)])
                img = _augment_stateless(img, seed2)
            y = tf.one_hot(lab, depth=len(classes), dtype=tf.float32)
            return img, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        if training:
            pass
        else:
            ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(tr_paths, tr_labels, training=True)
    val_ds = make_ds(val_paths, val_labels, training=False)

    test_paths = _list_test_files(test_dir)
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(
        lambda p: _decode_resize_preprocess(p), num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.cache().batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    return (
        train_ds,
        val_ds,
        test_ds,
        tr_paths,
        tr_labels,
        val_paths,
        val_labels,
        test_paths,
    )


def define_generators(
    train_directory, test_directory, classes, validation_split=0.2, seed=1337
):
    return None, None, None


def visualize_generator_samples(generator, rows=2, cols=6):
    return




## === cell 5
IMAGE_SIZE = [299, 299]  # kept as in original
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
width = 299
height = 299
num_classes = 12
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

train_ds, val_ds, test_ds, tr_paths, tr_labels, val_paths, val_labels, test_paths = (
    build_train_val_test_datasets(
        train_directory,
        test_directory,
        classes=CLASSES,
        validation_split=0.2,
        seed=SEED,
    )
)

len_train_generator = math.ceil(len(tr_paths) / BATCH_SIZE)
len_validation_generator = math.ceil(len(val_paths) / BATCH_SIZE)
len_test_generator = math.ceil(len(test_paths) / BATCH_SIZE)

print(f"\nLength of Train Generator: {len_train_generator}")
print(f"Length of Validation Generator: {len_validation_generator}")
print(f"Length of Test Generator: {len_test_generator}")

print(
    "\nNumber of train samples:",
    len(tr_paths),
    "val samples:",
    len(val_paths),
    "test samples:",
    len(test_paths),
)

images, labels = next(iter(train_ds))
print(f"\nShape of images in the first batch: {images.shape}")
print(f"Shape of labels in the first batch: {labels.shape}")




## === cell 6
def create_ResNet50_model():
    pretrained_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
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
        include_top=False, input_shape=[*IMAGE_SIZE, 3]
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
            tf.keras.layers.Dense(128, activation="relu"),
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
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def write_history(j, filename):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if historys[i] != 0:
            history_dict[i] = historys[i].history
    filename = filename + ".pkl"
    with open(filename, "ab") as pklfile:
        pickle.dump(history_dict, pklfile)


def load_history(filename):
    with open(filename, "rb") as file:
        history_dict = pickle.load(file)
    return history_dict


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()




## === cell 7
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


LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = LR_START
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.80

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(30)]
y = [lrfn(x) for x in rng]
print("lrfn y (first 10):", y[:10], "...")




## === cell 8
no_of_models = 1
models = [0] * no_of_models
start_model = 0
end_model = 1

EPOCHS = 50
historys = [0] * no_of_models
finished_models = 0

checkpoint_path = "/kaggle/working/best_model.keras"

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
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

val_probabilities = [0] * no_of_models
test_probabilities = [0] * no_of_models
all_probabilities = [0] * no_of_models

with strategy.scope():
    for j in range(no_of_models):
        models[j] = create_InceptionV3_model()

models[0].summary()

try:
    tf.keras.backend.set_image_data_format("channels_last")
except Exception:
    pass

steps_per_epoch = max(1, math.ceil(len(tr_paths) / BATCH_SIZE))
validation_steps = max(1, math.ceil(len(val_paths) / BATCH_SIZE))

fit_kwargs = {}
if tpu is None:
    fit_kwargs.update(
        dict(
            workers=min(8, os.cpu_count() or 1),
            use_multiprocessing=True,
            max_queue_size=32,
        )
    )

for j in range(start_model, end_model):
    start_training = datetime.now()
    print(start_training)
    time_from_start_program_tdelta = start_training - start_time
    if time_from_start_program_tdelta > end_training_by_tdelta:
        print(j, "time limit for doing training over, get out")
        break

    print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)

    historys[j] = models[j].fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[
            learning_rate_reduction,
            early_stopping,
            lr_callback,
            model_checkpoint_callback,
        ],
        **fit_kwargs,
    )

    filename = this_run_file_prefix + "models_" + str(j)
    write_history(j, filename)
    models[j].save(filename + ".keras")

    gc.collect()
    finished_models = j + 1

print(datetime.now())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2944330250.py in <cell line: 0>()
     68     print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)
     69 
---> 70     historys[j] = models[j].fit(
     71         train_ds,
     72         epochs=EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 9
if historys[0] != 0:
    history_to_analyze = historys[0].history
    print(
        "Training history keys:",
        list(history_to_analyze.keys()),
        "epochs_ran:",
        len(history_to_analyze.get("loss", [])),
    )
else:
    print("Training did not run (history missing). Proceeding without plot.")

if os.path.exists(checkpoint_path):
    best_model = load_model(checkpoint_path)
    print("Loaded best model from checkpoint:", checkpoint_path)
else:
    best_model = models[0]
    print("Checkpoint not found; using in-memory trained model.")




## === cell 10
def plot_test_images_with_predictions(generator, predictions, num_images):
    return


predictions = best_model.predict(test_ds, steps=len_test_generator, verbose=1)
class_list = [CLASSES[int(np.argmax(pred, axis=-1))] for pred in predictions]

plot_test_images_with_predictions(None, class_list, num_images=25)




## === cell 11
submission = pd.DataFrame()
submission["file"] = [os.path.basename(p) for p in test_paths]
submission["species"] = class_list

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
else:
    fallback_sample = (
        "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
    )
    sample = pd.read_csv(fallback_sample) if os.path.exists(fallback_sample) else None

if sample is not None:
    submission = sample[["file"]].merge(submission, on="file", how="left")
    submission["species"] = submission["species"].fillna(CLASSES[0])

submission.to_csv("submission.csv", index=False)
print("Submission file generated:", os.path.abspath("submission.csv"))
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
assert list(submission.columns) == ["file", "species"]
assert len(submission) == 666
assert submission["species"].notna().all()

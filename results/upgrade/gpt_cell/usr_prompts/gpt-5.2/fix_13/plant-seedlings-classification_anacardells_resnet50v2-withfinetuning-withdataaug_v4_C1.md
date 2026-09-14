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

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0

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

0.08753

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.07508) has done: 'Diagnosis: The crash happens during the TensorFlow/Keras imports in cell 8, triggered by an incompatibility between the forced pure-Python protobuf runtime (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) and the installed `protobuf==6.33.0`, which breaks TensorFlow’s internal proto handling (`MessageFactory.GetPrototype` missing). This is not a model-code issue; it’s an environment configuration issue introduced in cell 8.  

Patch summary: Remove the forced protobuf implementation overrides and the manual unloading of `google.protobuf` modules, letting TensorFlow use its default compatible protobuf runtime. Keep all ML-related imports and downstream variables unchanged so cell 9 continues to work identically.  

Updated cells: Only cell 8 is modified.  

Compatibility notes for cell k+1: All imported symbols used in cell 9 (`ImageDataGenerator`, `preprocess_input`, etc.) remain available with the same names and behavior. No changes to data paths, generators, or model logic are introduced.  

Assumptions: TensorFlow 2.18 in this environment is compatible with the default protobuf runtime provided by the installed packages, and no other notebook cells depend on the protobuf environment variables being set to the pure-Python implementation.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os



## === cell 1
import csv

csv_trainfile = "/kaggle/working/train.csv"
train_root = "/kaggle/input/plant-seedlings-classification/train"

if not os.path.exists(csv_trainfile):
    with open(csv_trainfile, "w") as f:
        for class_name in sorted(os.listdir(train_root)):
            class_dir = os.path.join(train_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                f.write(f"{class_dir}/{filename};{class_name};{filename}\n")



## === cell 2
column_names = ["path", "specie", "file"]
if os.path.exists(csv_trainfile):
    dataFrameTrain = pd.read_csv(
        csv_trainfile, delimiter=";", header=None, names=column_names
    )
else:
    dataFrameTrain = pd.DataFrame(columns=column_names)

print(dataFrameTrain.shape)
print(dataFrameTrain.head())



## === cell 3
print(
    "Null counts:\n",
    (
        dataFrameTrain.isna().sum()
        if len(dataFrameTrain)
        else "train.csv not loaded (skipped)"
    ),
)



## === cell 4
classes = sorted(
    [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
)
print(f"Number of classes: {len(classes)}")
if len(dataFrameTrain):
    datos_classes = dataFrameTrain.groupby("specie").count()
    print(datos_classes)



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
from math import exp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 9
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

AUTOTUNE = tf.data.AUTOTUNE

class_names = sorted(
    [
        d
        for d in os.listdir(PROYECT_FOLDER_TRAIN)
        if os.path.isdir(os.path.join(PROYECT_FOLDER_TRAIN, d))
    ]
)
num_classes = len(class_names)
class_to_index = {name: i for i, name in enumerate(class_names)}

all_paths = []
all_labels = []
for cname in class_names:
    cdir = os.path.join(PROYECT_FOLDER_TRAIN, cname)
    for fn in os.listdir(cdir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            all_paths.append(os.path.join(cdir, fn))
            all_labels.append(class_to_index[cname])

all_paths = np.array(all_paths)
all_labels = np.array(all_labels, dtype=np.int32)

rng = np.random.RandomState(seed)
perm = rng.permutation(len(all_paths))
all_paths = all_paths[perm]
all_labels = all_labels[perm]

n_total = len(all_paths)
n_val = int(round(n_total * val_split))
val_paths, val_labels = all_paths[:n_val], all_labels[:n_val]
train_paths, train_labels = all_paths[n_val:], all_labels[n_val:]


def _decode_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img, tf.one_hot(label, depth=num_classes)


def _augment(img, y):
    img = tf.image.random_flip_left_right(img, seed=seed)
    img = tf.image.random_flip_up_down(img, seed=seed)

    img = tf.image.stateless_random_brightness(
        img,
        max_delta=0.3 * 255.0,
        seed=tf.random.experimental.stateless_split([seed, 1], 2)[0],
    )

    shift = int(0.2 * image_size[0])
    img = tf.image.resize_with_crop_or_pad(
        img, image_size[0] + 2 * shift, image_size[1] + 2 * shift
    )
    img = tf.image.random_crop(img, size=[image_size[0], image_size[1], 3], seed=seed)

    z = tf.random.stateless_uniform([], seed=[seed, 2], minval=0.8, maxval=1.2)

    def zoom_in():
        crop_frac = 1.0 / z
        crop_h = tf.cast(
            tf.round(crop_frac * tf.cast(image_size[0], tf.float32)), tf.int32
        )
        crop_w = tf.cast(
            tf.round(crop_frac * tf.cast(image_size[1], tf.float32)), tf.int32
        )
        cropped = tf.image.random_crop(img, size=[crop_h, crop_w, 3], seed=seed)
        return tf.image.resize(
            cropped, image_size, method=tf.image.ResizeMethod.BILINEAR
        )

    def zoom_out():
        pad_h = tf.cast(
            tf.round((z - 1.0) * tf.cast(image_size[0], tf.float32)), tf.int32
        )
        pad_w = tf.cast(
            tf.round((z - 1.0) * tf.cast(image_size[1], tf.float32)), tf.int32
        )
        padded = tf.image.resize_with_crop_or_pad(
            img, image_size[0] + pad_h, image_size[1] + pad_w
        )
        return tf.image.resize(
            padded, image_size, method=tf.image.ResizeMethod.BILINEAR
        )

    img = tf.cond(z >= 1.0, zoom_out, zoom_in)

    angle = tf.random.stateless_uniform(
        [], seed=[seed, 3], minval=-30.0, maxval=30.0
    ) * (np.pi / 180.0)
    img = tfa_image_rotate(img, angle)

    img = img * 0.9

    img = preprocess_input(img)
    return img, y


def tfa_image_rotate(image, angles):
    angles = tf.cast(angles, tf.float32)
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0
    cos_a = tf.cos(angles)
    sin_a = tf.sin(angles)
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    image4 = image[None, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image4,
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


def make_ds(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(train_paths, train_labels, training=True)
val_ds = make_ds(val_paths, val_labels, training=False)


class _GenLike:
    def __init__(self, class_indices, n, batch_size, filenames=None):
        self.class_indices = class_indices
        self.n = n
        self.batch_size = batch_size
        self.num_classes = len(class_indices)
        self.filenames = filenames or []


train_generator = _GenLike(class_to_index, n=len(train_paths), batch_size=batch_size)
val_generator = _GenLike(class_to_index, n=len(val_paths), batch_size=batch_size)



## === cell 10
input_shape_c = tuple((image_size[0], image_size[1], 3))
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 11
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)

pre_trained_model.add(layers.GlobalAveragePooling2D())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(12, activation="softmax"))
pre_trained_model.summary()



## === cell 12
epochs = 200

n_classes = getattr(train_generator, "num_classes", None)
if n_classes is None:
    n_classes = len(train_generator.class_indices)

last_layer = pre_trained_model.layers[-1]
if isinstance(last_layer, Dense) and getattr(last_layer, "units", None) != n_classes:
    pre_trained_model.pop()
    pre_trained_model.add(layers.Dense(n_classes, activation="softmax"))

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

modelsave = ModelCheckpoint(filepath=file + ".h5", save_best_only=True, verbose=1)

print("[INFO]: Entrenando la red...")

steps_per_epoch = int(np.ceil(train_generator.n / train_generator.batch_size))
validation_steps = int(np.ceil(val_generator.n / val_generator.batch_size))

H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 13
pass



## === cell 14
csv_testfile = "/kaggle/working/test.csv"
test_root = "/kaggle/input/plant-seedlings-classification/test"

if not os.path.exists(csv_testfile):
    with open(csv_testfile, "w") as f:
        for filename in sorted(os.listdir(test_root)):
            if filename.lower().endswith(".png"):
                f.write(f"{test_root}/{filename};{filename}\n")



## === cell 15
column_names = ["path", "file"]
dataFrameTest = pd.read_csv(
    csv_testfile, delimiter=";", header=None, names=column_names
)

print(dataFrameTest.shape)
print(dataFrameTest.head())
print(dataFrameTest["path"][0])



## === cell 16
batch_size = 32
image_size = (256, 256)
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/"

test_paths = dataFrameTest["path"].to_numpy()


def _decode_resize_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (
    test_ds.map(_decode_resize_test, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

predicted_class = pre_trained_model.predict(
    test_ds,
    verbose=0,
)



## === cell 17
predicted_class_number = np.argmax(predicted_class, axis=1)

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
classes_in_index_order = [idx_to_class[i] for i in range(len(idx_to_class))]
print(classes_in_index_order)

csv_resultsfile = "/kaggle/working/submission.csv"

sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sub = pd.read_csv(sample_path)

pred_files = dataFrameTest["file"].tolist()
pred_labels = [classes_in_index_order[i] for i in predicted_class_number]

pred_df = pd.DataFrame({"file": pred_files, "species": pred_labels})

sub = sub[["file"]].merge(pred_df, on="file", how="left")
sub["species"] = sub["species"].fillna(classes_in_index_order[0])

sub = sub[["file", "species"]]
sub.to_csv(csv_resultsfile, index=False)

dataFrameResults = pd.read_csv(csv_resultsfile, delimiter=",")
print(dataFrameResults.shape)
print(dataFrameResults.head())
print("Missing predictions:", dataFrameResults["species"].isna().sum())
print("Saved submission to:", csv_resultsfile)

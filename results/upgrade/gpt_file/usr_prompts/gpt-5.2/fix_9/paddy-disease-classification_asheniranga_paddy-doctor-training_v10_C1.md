# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.pop("TF_USE_LEGACY_KERAS", None)

import tensorflow as tf
import cv2

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add, Activation
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.disable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
train_meta_data = "/kaggle/input/paddy-disease-classification/train.csv"
train_data_dir = "/kaggle/input/paddy-disease-classification/train_images"
epochs = 25
lr = 1e-4
valid_split = 0.2
input_size = 128
batch_size = 16
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy


def _resolve_image_dir(path, expected_subdirs=None):
    candidates = [path, os.path.join(path, os.path.basename(path))]
    if path.endswith("/kaggle/input") or path.endswith("/kaggle/input/"):
        candidates.append("/kaggle/input/paddy-disease-classification/train_images")
    for p in candidates:
        if os.path.isdir(p):
            if expected_subdirs is None:
                return p
            if all(os.path.isdir(os.path.join(p, d)) for d in expected_subdirs):
                return p
    return path


_expected_classnames = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

train_data_dir = _resolve_image_dir(
    train_data_dir, expected_subdirs=_expected_classnames
)

_detected = [
    d for d in _expected_classnames if os.path.isdir(os.path.join(train_data_dir, d))
]
if len(_detected) == 10:
    classnames = _detected
else:
    dirs = [
        d
        for d in sorted(os.listdir(train_data_dir))
        if os.path.isdir(os.path.join(train_data_dir, d))
        and d not in {"train_images", "test_images"}
    ]
    classnames = dirs[:10]

assert (
    len(classnames) == 10
), f"Expected 10 classes but got {len(classnames)}: {classnames}"
train_data_dir, classnames



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)




## === cell 3
def _first_existing_jpg(class_dir):
    for f in sorted(os.listdir(class_dir)):
        if f.lower().endswith(".jpg"):
            return os.path.join(class_dir, f)
    return None


src = _first_existing_jpg(os.path.join(train_data_dir, "dead_heart"))
if src is None:
    for c in classnames:
        src = _first_existing_jpg(os.path.join(train_data_dir, c))
        if src is not None:
            break

img = img_to_array(load_img(src), dtype="uint8")
img.shape




## === cell 4
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[1])
            if rv not in anchors_y:
                anchors_y.append(rv)

        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0

        return image
    else:
        return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    else:
        return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])

        if ax == 0:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            return np.row_stack(slices)
        else:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            return np.column_stack(slices)
    else:
        return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    with tf.device("/CPU:0"):
        image = tf.image.random_crop(image, (input_size, input_size, 3))
        image = image.numpy()
        image = random_cutout(image, 8, 16)
        image = random_displacment(image)
        image = random_gaus_blur(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 2.0)
        image = tf.image.random_saturation(image, 0.75, 1.25)
        image = tf.image.random_hue(image, 0.1).numpy()
    return image


def test_time_augmentation_fn(image):
    image = tf.convert_to_tensor(image)
    with tf.device("/CPU:0"):
        image = tf.image.random_crop(image, (input_size, input_size, 3))
        image = image.numpy()
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 2.0)
        image = image.numpy() if hasattr(image, "numpy") else image
    return image




## === cell 5
generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
    preprocessing_function=center_crop_and_random_augmentations_fn,
)

train_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    class_mode="categorical",
    classes=classnames,
)

valid_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    class_mode="categorical",
    classes=classnames,
)

for it in (train_datagen, valid_datagen):
    if hasattr(it, "_prefetch"):
        it._prefetch = max(1, it.batch_size * 2)
    if hasattr(it, "workers"):
        it.workers = 4
    if hasattr(it, "use_multiprocessing"):
        it.use_multiprocessing = True




## === cell 6
def _list_all_jpgs(root_dir):
    out = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                out.append(os.path.join(root, f))
    return out


all_train_jpgs = _list_all_jpgs(train_data_dir)
if len(all_train_jpgs) == 0:
    raise FileNotFoundError(f"No .jpg images found under {train_data_dir}")

rng = np.random.RandomState(SEED)
n_fit = min(128, len(all_train_jpgs))
fit_files = rng.choice(all_train_jpgs, size=n_fit, replace=False).tolist()

to_gen_fit = []
for file in fit_files:
    arr = img_to_array(
        load_img(file, target_size=(input_size, input_size)), dtype="uint8"
    )
    to_gen_fit.append(arr)

to_gen_fit = np.stack(to_gen_fit, axis=0)
to_gen_fit.shape



## === cell 7
generator.fit(to_gen_fit)



## === cell 8
train_batch = next(iter(train_datagen))
valid_batch = next(iter(valid_datagen))
len(train_batch[0]), len(valid_batch[0])



## === cell 9
if False:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()

    train_batch = next(iter(train_datagen))
    for i, arr in enumerate(train_batch[0][: len(axes)]):
        img = array_to_img(arr)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()



## === cell 10
if False:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()

    valid_batch = next(iter(valid_datagen))
    for i, arr in enumerate(valid_batch[0][: len(axes)]):
        img = array_to_img(arr)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()



## === cell 11
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(input_size, input_size, 3),
    pooling="avg",
)
base.trainable = True
model = tf.keras.Sequential(
    [base, tf.keras.layers.Dense(classes, activation="softmax")]
)

model.build([None, input_size, input_size, 3])
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
_using_hub = False
_using_hub



## === cell 12
model.summary()



## === cell 13
history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)



## === cell 14
tta_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
    class_mode="categorical",
    classes=classnames,
)
if hasattr(tta_datagen, "_prefetch"):
    tta_datagen._prefetch = max(1, tta_datagen.batch_size * 2)
if hasattr(tta_datagen, "workers"):
    tta_datagen.workers = 4
if hasattr(tta_datagen, "use_multiprocessing"):
    tta_datagen.use_multiprocessing = True



## === cell 15
n_val = tta_datagen.samples

val_files = [os.path.join(tta_datagen.directory, f) for f in tta_datagen.filenames]
val_x = np.empty((n_val, input_size, input_size, 3), dtype=np.uint8)
for i, fp in enumerate(val_files):
    val_x[i] = img_to_array(
        load_img(fp, target_size=(input_size, input_size)), dtype="uint8"
    )


_std_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)
_std_gen.mean = getattr(generator, "mean", None)
_std_gen.std = getattr(generator, "std", None)
_std_gen.principal_components = getattr(generator, "principal_components", None)
_std_gen.zca_whitening = getattr(generator, "zca_whitening", False)
if _std_gen.mean is None or _std_gen.std is None:
    _std_gen.fit(to_gen_fit)


def _standardize_batch_uint8(x_uint8):
    x = x_uint8.astype(np.float32, copy=False)
    x *= 1.0 / 255.0
    x = _std_gen.standardize(x)
    return x


def _predict_in_chunks(m, x, batch_size=16, verbose=0):
    outs = []
    n = x.shape[0]
    for i in range(0, n, max(1, batch_size * 32)):
        outs.append(
            m.predict(
                x[i : i + batch_size * 32], batch_size=batch_size, verbose=verbose
            )
        )
    return (
        np.concatenate(outs, axis=0)
        if len(outs)
        else np.zeros((0, classes), dtype=np.float32)
    )


eve_encodings = np.zeros((n_val, classes), dtype=np.float32)
for _ in range(5):
    aug = np.empty_like(val_x)
    for i in range(n_val):
        aug[i] = test_time_augmentation_fn(val_x[i])
    aug = _standardize_batch_uint8(aug)
    encodings = _predict_in_chunks(model, aug, batch_size=batch_size, verbose=0)
    eve_encodings += encodings

eve_encodings /= 5.0
eve_encodings[:3]



## === cell 16
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes
accuracy_score(true_classes, pred_classes)



## === cell 17
if False:
    plt.figure(figsize=[12, 6], dpi=300)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.legend()
    plt.show()



## === cell 18
if False:
    plt.figure(figsize=[12, 6], dpi=300)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.legend()
    plt.show()



## === cell 19
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_effnetb4_history.csv", index=False)
temp_hist.tail()



## === cell 20
model.save("model_effnet_b4.hdf5")



## === cell 21
model.save_weights("model_effnet_b4_weights.weights.h5")



## === cell 22
test_loc = "/kaggle/input/paddy-disease-classification/test_images"
if os.path.isdir(os.path.join(test_loc, "test_images")):
    test_loc = os.path.join(test_loc, "test_images")

test_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)



## === cell 23
test_generator.mean = getattr(generator, "mean", None)
test_generator.std = getattr(generator, "std", None)
test_generator.principal_components = getattr(generator, "principal_components", None)
test_generator.zca_whitening = getattr(generator, "zca_whitening", False)
if test_generator.mean is None or test_generator.std is None:
    test_generator.fit(to_gen_fit)



## === cell 24
test_datagen = ImageDataGenerator().flow_from_directory(
    directory=test_loc,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=["."],
    shuffle=False,
    class_mode=None,
)
if hasattr(test_datagen, "_prefetch"):
    test_datagen._prefetch = max(1, test_datagen.batch_size * 2)
if hasattr(test_datagen, "workers"):
    test_datagen.workers = 4
if hasattr(test_datagen, "use_multiprocessing"):
    test_datagen.use_multiprocessing = True



## === cell 25
if False:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()

    test_batch = next(iter(test_datagen))
    for i, arr in enumerate(test_batch[: len(axes)]):
        img = array_to_img(arr)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()



## === cell 26
n_test = test_datagen.samples
test_files = [os.path.join(test_datagen.directory, f) for f in test_datagen.filenames]

test_x = np.empty((n_test, input_size, input_size, 3), dtype=np.uint8)
for i, fp in enumerate(test_files):
    test_x[i] = img_to_array(
        load_img(fp, target_size=(input_size, input_size)), dtype="uint8"
    )


def _standardize_test_batch_uint8(x_uint8):
    x = x_uint8.astype(np.float32, copy=False)
    x *= 1.0 / 255.0
    x = test_generator.standardize(x)
    return x


test_encodings = np.zeros((n_test, classes), dtype=np.float32)
for _ in range(5):
    aug = np.empty_like(test_x)
    for i in range(n_test):
        aug[i] = test_time_augmentation_fn(test_x[i])
    aug = _standardize_test_batch_uint8(aug)
    encodings = _predict_in_chunks(model, aug, batch_size=batch_size, verbose=0)
    test_encodings += encodings

test_encodings /= 5.0
test_encodings[:3]



## === cell 27
predict_max = np.argmax(test_encodings, axis=1)
predict_max[:10]



## === cell 28
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]
predictions[:10]



## === cell 29
files = test_datagen.filenames
image_ids = [os.path.basename(f) for f in files]

sub = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
    if sub["label"].isna().any():
        fill_value = (
            "normal" if len(predictions) == 0 else pd.Series(predictions).mode().iloc[0]
        )
        sub["label"] = sub["label"].fillna(fill_value)

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 30
sub.label.value_counts()

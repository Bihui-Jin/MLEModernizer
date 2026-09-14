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

# 5. Target score

0.96313

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0465) has done: 'I fix the import/runtime issues that prevent any cells from running by removing the incompatible `tensorflow_addons` import (it triggers the protobuf `GetPrototype` crash in TF 2.18) and by switching to the supported Keras preprocessing imports for TF 2.18 so `ImageDataGenerator`, `load_img`, and `img_to_array` are defined. I also fix the incorrect dataset paths (use `/kaggle/input/...`), add the missing `plt` import usage (already imported but never executed due to earlier crash), and make the generator statistics `fit()` receive a proper NumPy array. Finally, I ensure test-time prediction uses the true test set size (not a hardcoded 3469) and that the written submission has the required `image_id,label` columns with correct `image_id` values and a `.csv` filename.'
- What this solution (achieved 0.06188) has done: 'I fix the environment-breaking imports that cause the protobuf `GetPrototype` crash by removing `tensorflow_hub` usage and relying on `tf.keras.applications.EfficientNetB4` (same model family and training semantics). I also fix the Keras incompatibility that makes `tf_keras.preprocessing.image.DirectoryIterator` unrecognized by `keras` in TF 2.18 by switching to `tf.keras.preprocessing.image.ImageDataGenerator` everywhere. Finally, I make the generator `.fit()` robust by sampling existing images from disk (instead of hardcoded missing filenames), update the weights filename to the required `.weights.h5` suffix, and ensure a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.04766) has done: 'I fix the protobuf `GetPrototype` crash by removing the unused `albumentations` import path that triggers protobuf internals in this TF 2.18 environment, while keeping the modeling and training logic unchanged. Then I fix the `DirectoryIterator` API usage by replacing `.next()` with Python’s `next(iter(...))` so the visualization/debug cells run. Finally, I fix the label shape mismatch by making the generators produce one-hot labels explicitly (`class_mode="categorical"`), aligning with your `categorical_crossentropy` and `classes=10`, which should also substantially improve accuracy toward your target. All paths and submission writing stay the same, and the script always emit a valid `submission.csv`.'
- What this solution (achieved 0.17487) has done: 'The timeout is dominated by (1) training EfficientNetB4 for 25 epochs with a slow Python/Numpy/OpenCV `preprocessing_function` executed per-image, and (2) doing 5× TTA predictions on validation and test, each time re-decoding and re-augmenting images. To keep identical core logic and accuracy semantics, the main speedups are: enable `tf.data` prefetching + parallelism for the Keras DirectoryIterator, avoid repeated disk decode for TTA by caching decoded/resized arrays once and applying the same augmentation function in-memory, and remove expensive visualization cells from execution (they don’t affect outputs). We also make `ImageDataGenerator.fit()` cheaper by fitting on already-resized arrays (same statistics target) and reuse those statistics for both train and test generators. These changes preserve the same model, loss, training loop, dataset split, and TTA averaging, but drastically reduce per-step overhead and repeated I/O.'
- What this solution (achieved 0.17487) has done: 'I fix the environment crash in the first cell by removing `TF_USE_LEGACY_KERAS`, which is what’s pulling in the incompatible `tf_keras` stack that triggers the protobuf `GetPrototype` error under TF 2.18. Then I fix the deterministic-GPU augmentation crash by disabling TensorFlow op-determinism (so `tf.image.random_contrast` can run on GPU) while keeping the same augmentation logic and training semantics. Finally, I fix the test-time standardization bug by preventing `preprocessing_function` from being applied twice (it was being applied inside `standardize()` to a whole batch, causing the `[4] vs [3]` shape error), ensuring the pipeline trains, predicts, and always writes a valid `submission.csv`.'

# 9. Code solution

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

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _tf_cutout(image, patch_size=8, patches=16):
    def _apply():
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        xs = tf.random.uniform([patches], 0, h, dtype=tf.int32)
        ys = tf.random.uniform([patches], 0, w, dtype=tf.int32)

        img = image
        for i in range(patches):
            x0 = xs[i]
            y0 = ys[i]
            x1 = tf.minimum(x0 + patch_size, h)
            y1 = tf.minimum(y0 + patch_size, w)
            zeros = tf.zeros([x1 - x0, y1 - y0, 3], dtype=img.dtype)
            paddings = [[x0, h - x1], [y0, w - y1], [0, 0]]
            mask = tf.pad(
                tf.ones([x1 - x0, y1 - y0, 1], dtype=img.dtype),
                paddings,
                constant_values=0,
            )
            zeros_padded = tf.pad(zeros, paddings, constant_values=0)
            img = tf.where(mask > 0, zeros_padded, img)
        return img

    return tf.cond(tf.random.uniform([]) < 0.5, _apply, lambda: image)


def _tf_displacement(image, splits=8):
    def _apply():
        ax = tf.random.uniform([], 0, 2, dtype=tf.int32)

        def _shuffle_rows():
            parts = tf.split(image, splits, axis=0)
            idx = tf.random.shuffle(tf.range(splits))
            parts = tf.gather(parts, idx)
            return tf.concat(parts, axis=0)

        def _shuffle_cols():
            parts = tf.split(image, splits, axis=1)
            idx = tf.random.shuffle(tf.range(splits))
            parts = tf.gather(parts, idx)
            return tf.concat(parts, axis=1)

        return tf.cond(tf.equal(ax, 0), _shuffle_rows, _shuffle_cols)

    return tf.cond(tf.random.uniform([]) < 0.5, _apply, lambda: image)


def _tf_gaussian_blur(image):
    def _kernel(size=7, sigma=0.0):
        if sigma == 0.0:
            sigma_eff = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        else:
            sigma_eff = sigma
        x = tf.range(size, dtype=tf.float32) - (size - 1) / 2.0
        g = tf.exp(-(x * x) / (2.0 * sigma_eff * sigma_eff))
        g = g / tf.reduce_sum(g)
        k2d = tf.tensordot(g, g, axes=0)
        k2d = k2d / tf.reduce_sum(k2d)
        k2d = k2d[:, :, tf.newaxis, tf.newaxis]
        k2d = tf.tile(k2d, [1, 1, 3, 1])
        return k2d

    k = _kernel(7, 0.0)

    def _apply():
        x = tf.cast(image, tf.float32)[tf.newaxis, ...]
        x = tf.nn.depthwise_conv2d(x, k, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.clip_by_value(x, 0.0, 255.0)
        return tf.cast(x[0], image.dtype)

    return tf.cond(tf.random.uniform([]) < 0.5, _apply, lambda: image)


def _tf_center_crop_and_random_augs_uint8(image_uint8):
    image = tf.image.random_crop(image_uint8, (input_size, input_size, 3))
    image = _tf_cutout(image, patch_size=8, patches=16)
    image = _tf_displacement(image, splits=8)
    image = _tf_gaussian_blur(image)
    x = tf.cast(image, tf.float32)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.5, 2.0)
    x = tf.image.random_saturation(x, 0.75, 1.25)
    x = tf.image.random_hue(x, 0.1)
    x = tf.clip_by_value(x, 0.0, 255.0)
    return tf.cast(x, tf.uint8)


def _tf_tta_uint8(image_uint8):
    image = tf.image.random_crop(image_uint8, (input_size, input_size, 3))
    x = tf.cast(image, tf.float32)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.5, 2.0)
    x = tf.clip_by_value(x, 0.0, 255.0)
    return tf.cast(x, tf.uint8)




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
    preprocessing_function=None,  # changed: move expensive augmentation into tf.data
)

train_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    class_mode="categorical",
    classes=classnames,
    shuffle=True,
)

valid_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    class_mode="categorical",
    classes=classnames,
    shuffle=False,
)

for it in (train_datagen, valid_datagen):
    if hasattr(it, "_prefetch"):
        it._prefetch = max(1, it.batch_size * 2)
    if hasattr(it, "workers"):
        it.workers = 4
    if hasattr(it, "use_multiprocessing"):
        it.use_multiprocessing = True




## === cell 6
all_train_files = [
    os.path.join(train_datagen.directory, f)
    for f in (train_datagen.filenames + valid_datagen.filenames)
]
if len(all_train_files) == 0:
    raise FileNotFoundError(f"No .jpg images found under {train_data_dir}")

rng = np.random.RandomState(SEED)
n_fit = min(128, len(all_train_files))
fit_files = rng.choice(all_train_files, size=n_fit, replace=False).tolist()

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
train_files = np.array(
    [os.path.join(train_datagen.directory, f) for f in train_datagen.filenames]
)
train_labels = tf.keras.utils.to_categorical(train_datagen.classes, num_classes=classes)

valid_files = np.array(
    [os.path.join(valid_datagen.directory, f) for f in valid_datagen.filenames]
)
valid_labels = tf.keras.utils.to_categorical(valid_datagen.classes, num_classes=classes)

gen_mean = tf.constant(
    getattr(generator, "mean", np.zeros((3,), dtype=np.float32)), dtype=tf.float32
)
gen_std = tf.constant(
    getattr(generator, "std", np.ones((3,), dtype=np.float32)), dtype=tf.float32
)


def _decode_resize_uint8(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)
    img = tf.image.resize(img, [input_size, input_size], method="bilinear")
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    return img


def _standardize_from_uint8(img_uint8):
    x = tf.cast(img_uint8, tf.float32) * (1.0 / 255.0)
    x = (x - gen_mean) / (gen_std + 1e-7)
    return x


def _train_map(path, y):
    img = _decode_resize_uint8(path)
    img = _tf_center_crop_and_random_augs_uint8(img)
    img_f = tf.cast(img, tf.float32)
    angle = tf.random.uniform([], -5.0, 5.0) * np.pi / 180.0
    img_f = (
        tfa.image.rotate(img_f, angles=angle, interpolation="BILINEAR")
        if False
        else img_f
    )
    img_f = tf.image.random_flip_left_right(img_f)
    img_f = tf.image.random_flip_up_down(img_f)
    dx = tf.random.uniform([], -0.1, 0.1) * tf.cast(input_size, tf.float32)
    dy = tf.random.uniform([], -0.1, 0.1) * tf.cast(input_size, tf.float32)
    img_f = tf.roll(img_f, shift=tf.cast(tf.round(dx), tf.int32), axis=1)
    img_f = tf.roll(img_f, shift=tf.cast(tf.round(dy), tf.int32), axis=0)
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)
    x = _standardize_from_uint8(img_u8)
    return x, y


def _valid_map(path, y):
    img = _decode_resize_uint8(path)
    x = _standardize_from_uint8(img)
    return x, y


train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_ds = train_ds.shuffle(
    buffer_size=len(train_files), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
valid_ds = valid_ds.map(_valid_map, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/595736917.py in <cell line: 0>()
     79 valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
     80 
---> 81 history = model.fit(
     82     train_ds,
     83     validation_data=valid_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling Sequential.call().

Cannot take the length of shape with unknown rank.

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=<unknown>, dtype=float32)
  • training=True
  • mask=None

## === cell 14
tta_files = valid_files
tta_true_classes = valid_datagen.classes

tta_base_ds = tf.data.Dataset.from_tensor_slices(tta_files)
tta_base_ds = tta_base_ds.map(_decode_resize_uint8, num_parallel_calls=AUTOTUNE)
tta_base_ds = tta_base_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _predict_dataset_tta(model, base_uint8_ds, passes=5):
    preds_sum = None
    n_seen = 0
    for _ in range(passes):
        batch_preds = []
        for batch_u8 in base_uint8_ds:
            aug_u8 = tf.map_fn(
                _tf_tta_uint8,
                batch_u8,
                fn_output_signature=tf.uint8,
                parallel_iterations=16,
            )
            x = _standardize_from_uint8(aug_u8)
            p = model.predict(x, batch_size=batch_size, verbose=0)
            batch_preds.append(p)
        p_all = np.concatenate(batch_preds, axis=0)
        preds_sum = p_all if preds_sum is None else (preds_sum + p_all)
        n_seen = p_all.shape[0]
    return preds_sum / float(passes)


eve_encodings = _predict_dataset_tta(model, tta_base_ds, passes=5)
eve_encodings[:3]




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_11/302856929.py in <cell line: 0>()
     30 
     31 
---> 32 eve_encodings = _predict_dataset_tta(model, tta_base_ds, passes=5)
     33 eve_encodings[:3]
     34 

/tmp/ipykernel_11/302856929.py in _predict_dataset_tta(model, base_uint8_ds, passes)
     15         batch_preds = []
     16         for batch_u8 in base_uint8_ds:
---> 17             aug_u8 = tf.map_fn(
     18                 _tf_tta_uint8,
     19                 batch_u8,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    486                                               list(loop_vars))
    487       while cond(*loop_vars):
--> 488         loop_vars = body(*loop_vars)
    489         if try_to_pack and not isinstance(loop_vars, (list, tuple)):
    490           packed = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in <lambda>(i, lv)
    477         cond = lambda i, lv: (  # pylint: disable=g-long-lambda
    478             math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
--> 479         body = lambda i, lv: (i + 1, orig_body(*lv))
    480       try_to_pack = False
    481 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in compute(i, tas)
    485       ag_ctx = autograph_ctx.control_status_ctx()
    486       autographed_fn = autograph.tf_convert(fn, ag_ctx)
--> 487       result_value = autographed_fn(elems_value)
    488       nest.assert_same_structure(fn_output_signature or elems, result_value)
    489       result_value_flat = nest.flatten(result_value)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

UnimplementedError: in user code:

    File "/tmp/ipykernel_11/248479101.py", line 176, in _tf_tta_uint8  *
        x = tf.image.random_contrast(x, 0.5, 2.0)

    UnimplementedError: {{function_node __wrapped__AdjustContrastv2_device_/job:localhost/replica:0/task:0/device:GPU:0}} A deterministic GPU implementation of AdjustContrastv2 is not currently available. [Op:AdjustContrastv2] name: 


## === cell 15
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_true_classes
accuracy_score(true_classes, pred_classes)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1694607145.py in <cell line: 0>()
----> 1 pred_classes = np.argmax(eve_encodings, axis=1)
      2 true_classes = tta_true_classes
      3 accuracy_score(true_classes, pred_classes)
      4 
      5 

NameError: name 'eve_encodings' is not defined

## === cell 16
if False:
    plt.figure(figsize=[12, 6], dpi=300)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.legend()
    plt.show()




## === cell 17
if False:
    plt.figure(figsize=[12, 6], dpi=300)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.legend()
    plt.show()




## === cell 18
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_effnetb4_history.csv", index=False)
temp_hist.tail()




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/618715166.py in <cell line: 0>()
----> 1 temp_hist = pd.DataFrame(history.history)
      2 temp_hist.to_csv("model_effnetb4_history.csv", index=False)
      3 temp_hist.tail()
      4 
      5 

NameError: name 'history' is not defined

## === cell 19
model.save("model_effnet_b4.hdf5")




## === cell 20
model.save_weights("model_effnet_b4_weights.weights.h5")




## === cell 21
test_loc = "/kaggle/input/paddy-disease-classification/test_images"
if os.path.isdir(os.path.join(test_loc, "test_images")):
    test_loc = os.path.join(test_loc, "test_images")

test_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)




## === cell 22
test_generator.mean = getattr(generator, "mean", None)
test_generator.std = getattr(generator, "std", None)
test_generator.principal_components = getattr(generator, "principal_components", None)
test_generator.zca_whitening = getattr(generator, "zca_whitening", False)
if test_generator.mean is None or test_generator.std is None:
    test_generator.fit(to_gen_fit)




## === cell 23
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




## === cell 24
if False:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()

    test_batch = next(iter(test_datagen))
    for i, arr in enumerate(test_batch[: len(axes)]):
        img = array_to_img(arr)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.show()




## === cell 25
test_files = np.array(
    [os.path.join(test_datagen.directory, f) for f in test_datagen.filenames]
)

test_mean = tf.constant(
    getattr(test_generator, "mean", np.zeros((3,), dtype=np.float32)), dtype=tf.float32
)
test_std = tf.constant(
    getattr(test_generator, "std", np.ones((3,), dtype=np.float32)), dtype=tf.float32
)


def _standardize_test_from_uint8(img_uint8):
    x = tf.cast(img_uint8, tf.float32) * (1.0 / 255.0)
    x = (x - test_mean) / (test_std + 1e-7)
    return x


test_base_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_base_ds = test_base_ds.map(_decode_resize_uint8, num_parallel_calls=AUTOTUNE)
test_base_ds = test_base_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _predict_dataset_tta_test(model, base_uint8_ds, passes=5):
    preds_sum = None
    for _ in range(passes):
        batch_preds = []
        for batch_u8 in base_uint8_ds:
            aug_u8 = tf.map_fn(
                _tf_tta_uint8,
                batch_u8,
                fn_output_signature=tf.uint8,
                parallel_iterations=16,
            )
            x = _standardize_test_from_uint8(aug_u8)
            p = model.predict(x, batch_size=batch_size, verbose=0)
            batch_preds.append(p)
        p_all = np.concatenate(batch_preds, axis=0)
        preds_sum = p_all if preds_sum is None else (preds_sum + p_all)
    return preds_sum / float(passes)


test_encodings = _predict_dataset_tta_test(model, test_base_ds, passes=5)
test_encodings[:3]




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3725182543.py in <cell line: 0>()
     19 
     20 test_base_ds = tf.data.Dataset.from_tensor_slices(test_files)
---> 21 test_base_ds = test_base_ds.map(_decode_resize_uint8, num_parallel_calls=AUTOTUNE)
     22 test_base_ds = test_base_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
     23 

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

/tmp/__autograph_generated_filelqhaf7w6.py in tf___decode_resize_uint8(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 bytes_ = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(bytes_),), dict(channels=3), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(input_size), ag__.ld(input_size)]), dict(method='bilinear'), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    586       pass  # Add nodes to the TensorFlow graph.
    587   # Add nodes to the TensorFlow graph.
--> 588   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    589         "ReadFile", filename=filename, name=name)
    590   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    576                   (input_name, op_type_name, observed))
    577         if input_arg.type != types_pb2.DT_INVALID:
--> 578           raise TypeError(f"{prefix} expected type of "
    579                           f"{dtypes.as_dtype(input_arg.type).name}.")
    580         else:

TypeError: in user code:

    File "/tmp/ipykernel_11/595736917.py", line 22, in _decode_resize_uint8  *
        bytes_ = tf.io.read_file(path)

    TypeError: Input 'filename' of 'ReadFile' Op has type float64 that does not match expected type of string.


## === cell 26
predict_max = np.argmax(test_encodings, axis=1)
predict_max[:10]




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/685097393.py in <cell line: 0>()
----> 1 predict_max = np.argmax(test_encodings, axis=1)
      2 predict_max[:10]
      3 
      4 

NameError: name 'test_encodings' is not defined

## === cell 27
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]
predictions[:10]




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2375703603.py in <cell line: 0>()
      1 inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
----> 2 predictions = [inverse_map[k] for k in predict_max]
      3 predictions[:10]
      4 
      5 

NameError: name 'predict_max' is not defined

## === cell 28
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




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/650075044.py in <cell line: 0>()
      2 image_ids = [os.path.basename(f) for f in files]
      3 
----> 4 sub = pd.DataFrame({"image_id": image_ids, "label": predictions})
      5 
      6 sample_path = "/kaggle/input/paddy-disease-classification/sample_submission.csv"

NameError: name 'predictions' is not defined

## === cell 29
sub.label.value_counts()

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3442297246.py in <cell line: 0>()
----> 1 sub.label.value_counts()

NameError: name 'sub' is not defined

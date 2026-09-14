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
- What this solution (achieved 0.0661) has done: 'I fix the protobuf `GetPrototype` crash by forcing the Python protobuf implementation before importing TensorFlow, which is the minimal environment-level change that unblocks all cells. Then I fix the training `EnsureShape` error by correcting the `tf.roll` axes in `_train_map` (it was rolling the channel axis, producing width=16), which is a pure bugfix and restores the intended augmentation behavior. Finally, I fix test image discovery by making `test_loc` resolution robust to nested `test_images/test_images` folders and by always aligning predictions back to `sample_submission.csv` order so a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
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
            parts = tf.split(
                image, splits, axis=0
            )  # list length splits, each [H/s, W, C]
            idx = tf.random.shuffle(tf.range(splits))
            gathered = [parts[i] for i in tf.unstack(idx)]
            out = tf.concat(gathered, axis=0)
            return out

        def _shuffle_cols():
            parts = tf.split(
                image, splits, axis=1
            )  # list length splits, each [H, W/s, C]
            idx = tf.random.shuffle(tf.range(splits))
            gathered = [parts[i] for i in tf.unstack(idx)]
            out = tf.concat(gathered, axis=1)
            return out

        out = tf.cond(tf.equal(ax, 0), _shuffle_rows, _shuffle_cols)
        out = tf.ensure_shape(out, (input_size, input_size, 3))
        return out

    out = tf.cond(tf.random.uniform([]) < 0.5, _apply, lambda: image)
    out = tf.ensure_shape(out, (input_size, input_size, 3))
    return out


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

    out = tf.cond(tf.random.uniform([]) < 0.5, _apply, lambda: image)
    out = tf.ensure_shape(out, (input_size, input_size, 3))
    return out


def _tf_center_crop_and_random_augs_uint8(image_uint8):
    image = tf.image.random_crop(image_uint8, (input_size, input_size, 3))
    image = tf.ensure_shape(image, (input_size, input_size, 3))
    image = _tf_cutout(image, patch_size=8, patches=16)
    image = tf.ensure_shape(image, (input_size, input_size, 3))
    image = _tf_displacement(image, splits=8)
    image = tf.ensure_shape(image, (input_size, input_size, 3))
    image = _tf_gaussian_blur(image)
    image = tf.ensure_shape(image, (input_size, input_size, 3))
    x = tf.cast(image, tf.float32)
    x = tf.image.random_brightness(x, 0.2)
    with tf.device("/CPU:0"):
        x = tf.image.random_contrast(x, 0.5, 2.0)
    x = tf.image.random_saturation(x, 0.75, 1.25)
    x = tf.image.random_hue(x, 0.1)
    x = tf.clip_by_value(x, 0.0, 255.0)
    out = tf.cast(x, tf.uint8)
    out = tf.ensure_shape(out, (input_size, input_size, 3))
    return out


def _tf_tta_uint8(image_uint8):
    image = tf.image.random_crop(image_uint8, (input_size, input_size, 3))
    image = tf.ensure_shape(image, (input_size, input_size, 3))
    x = tf.cast(image, tf.float32)
    x = tf.image.random_brightness(x, 0.2)
    with tf.device("/CPU:0"):
        x = tf.image.random_contrast(x, 0.5, 2.0)
    x = tf.clip_by_value(x, 0.0, 255.0)
    out = tf.cast(x, tf.uint8)
    out = tf.ensure_shape(out, (input_size, input_size, 3))
    return out




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
    preprocessing_function=None,  # expensive augmentation moved to tf.data
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
    [os.path.join(train_datagen.directory, f) for f in train_datagen.filenames],
    dtype=str,
)
train_labels = tf.keras.utils.to_categorical(train_datagen.classes, num_classes=classes)

valid_files = np.array(
    [os.path.join(valid_datagen.directory, f) for f in valid_datagen.filenames],
    dtype=str,
)
valid_labels = tf.keras.utils.to_categorical(valid_datagen.classes, num_classes=classes)

gen_mean = tf.constant(
    getattr(generator, "mean", np.zeros((3,), dtype=np.float32)), dtype=tf.float32
)
gen_std = tf.constant(
    getattr(generator, "std", np.ones((3,), dtype=np.float32)), dtype=tf.float32
)


def _decode_resize_uint8(path):
    path = tf.cast(path, tf.string)
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)
    img = tf.image.resize(img, [input_size, input_size], method="bilinear")
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    img = tf.ensure_shape(img, (input_size, input_size, 3))
    return img


def _standardize_single_from_uint8(img_uint8):
    img = tf.ensure_shape(img_uint8, (input_size, input_size, 3))
    x = tf.cast(img, tf.float32) * (1.0 / 255.0)
    x = (x - gen_mean) / (gen_std + 1e-7)
    x = tf.ensure_shape(x, (input_size, input_size, 3))
    return x


def _standardize_batch_from_uint8(img_uint8):
    img = tf.ensure_shape(img_uint8, (None, input_size, input_size, 3))
    x = tf.cast(img, tf.float32) * (1.0 / 255.0)
    x = (x - gen_mean) / (gen_std + 1e-7)
    x = tf.ensure_shape(x, (None, input_size, input_size, 3))
    return x


def _train_map(path, y):
    img = _decode_resize_uint8(path)
    img = _tf_center_crop_and_random_augs_uint8(img)
    img_f = tf.cast(img, tf.float32)

    img_f = tf.image.random_flip_left_right(img_f)
    img_f = tf.image.random_flip_up_down(img_f)

    dx = tf.random.uniform([], -0.1, 0.1) * tf.cast(input_size, tf.float32)
    dy = tf.random.uniform([], -0.1, 0.1) * tf.cast(input_size, tf.float32)

    img_f = tf.roll(img_f, shift=tf.cast(tf.round(dx), tf.int32), axis=1)
    img_f = tf.roll(img_f, shift=tf.cast(tf.round(dy), tf.int32), axis=0)
    img_f = tf.ensure_shape(img_f, (input_size, input_size, 3))

    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)
    img_u8 = tf.ensure_shape(img_u8, (input_size, input_size, 3))
    x = _standardize_single_from_uint8(img_u8)
    return x, y


def _valid_map(path, y):
    img = _decode_resize_uint8(path)
    x = _standardize_single_from_uint8(img)
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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/681275670.py in <cell line: 0>()
     76     buffer_size=len(train_files), seed=SEED, reshuffle_each_iteration=True
     77 )
---> 78 train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
     79 train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
     80 

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

/tmp/__autograph_generated_filesml6z_lu.py in tf___train_map(path, y)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_decode_resize_uint8), (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_tf_center_crop_and_random_augs_uint8), (ag__.ld(img),), None, fscope)
     12                 img_f = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img_f = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img_f),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file47sumsjk.py in tf___tf_center_crop_and_random_augs_uint8(image_uint8)
     12                 image = ag__.converted_call(ag__.ld(_tf_cutout), (ag__.ld(image),), dict(patch_size=8, patches=16), fscope)
     13                 image = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(image), (ag__.ld(input_size), ag__.ld(input_size), 3)), None, fscope)
---> 14                 image = ag__.converted_call(ag__.ld(_tf_displacement), (ag__.ld(image),), dict(splits=8), fscope)
     15                 image = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(image), (ag__.ld(input_size), ag__.ld(input_size), 3)), None, fscope)
     16                 image = ag__.converted_call(ag__.ld(_tf_gaussian_blur), (ag__.ld(image),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filexue_gklx.py in tf___tf_displacement(image, splits)
     58                             raise
     59                         return fscope_1.ret(retval__1, do_return_1)
---> 60                 out = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).random.uniform, ([],), None, fscope) < 0.5, ag__.ld(_apply), ag__.autograph_artifact(lambda: ag__.ld(image))), None, fscope)
     61                 out = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(out), (ag__.ld(input_size), ag__.ld(input_size), 3)), None, fscope)
     62                 try:

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filexue_gklx.py in _apply()
     49                                     raise
     50                                 return fscope_3.ret(retval__3, do_return_3)
---> 51                         out = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).equal, (ag__.ld(ax), 0), None, fscope_1), ag__.ld(_shuffle_rows), ag__.ld(_shuffle_cols)), None, fscope_1)
     52                         out = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(out), (ag__.ld(input_size), ag__.ld(input_size), 3)), None, fscope_1)
     53                         try:

/tmp/__autograph_generated_filexue_gklx.py in _shuffle_rows()
     23                                 parts = ag__.converted_call(ag__.ld(tf).split, (ag__.ld(image), ag__.ld(splits)), dict(axis=0), fscope_2)
     24                                 idx = ag__.converted_call(ag__.ld(tf).random.shuffle, (ag__.converted_call(ag__.ld(tf).range, (ag__.ld(splits),), None, fscope_2),), None, fscope_2)
---> 25                                 gathered = [ag__.ld(parts)[ag__.ld(i)] for i in ag__.converted_call(ag__.ld(tf).unstack, (ag__.ld(idx),), None, fscope_2)]
     26                                 out = ag__.converted_call(ag__.ld(tf).concat, (ag__.ld(gathered),), dict(axis=0), fscope_2)
     27                                 try:

/tmp/__autograph_generated_filexue_gklx.py in <listcomp>(.0)
     23                                 parts = ag__.converted_call(ag__.ld(tf).split, (ag__.ld(image), ag__.ld(splits)), dict(axis=0), fscope_2)
     24                                 idx = ag__.converted_call(ag__.ld(tf).random.shuffle, (ag__.converted_call(ag__.ld(tf).range, (ag__.ld(splits),), None, fscope_2),), None, fscope_2)
---> 25                                 gathered = [ag__.ld(parts)[ag__.ld(i)] for i in ag__.converted_call(ag__.ld(tf).unstack, (ag__.ld(idx),), None, fscope_2)]
     26                                 out = ag__.converted_call(ag__.ld(tf).concat, (ag__.ld(gathered),), dict(axis=0), fscope_2)
     27                                 try:

TypeError: in user code:

    File "/tmp/ipykernel_11/681275670.py", line 49, in _train_map  *
        img = _tf_center_crop_and_random_augs_uint8(img)
    File "/tmp/ipykernel_11/3068936733.py", line 166, in _tf_center_crop_and_random_augs_uint8  *
        image = _tf_displacement(image, splits=8)
    File "/tmp/ipykernel_11/3068936733.py", line 111, in _shuffle_rows  *
        gathered = [parts[i] for i in tf.unstack(idx)]

    TypeError: list indices must be integers or slices, not SymbolicTensor


## === cell 14
tta_files = valid_files
tta_true_classes = valid_datagen.classes

tta_base_ds = tf.data.Dataset.from_tensor_slices(
    tf.constant(tta_files, dtype=tf.string)
)
tta_base_ds = tta_base_ds.map(_decode_resize_uint8, num_parallel_calls=AUTOTUNE)
tta_base_ds = tta_base_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _predict_dataset_tta(model, base_uint8_ds, passes=5):
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
            x = _standardize_batch_from_uint8(aug_u8)
            p = model.predict(x, batch_size=batch_size, verbose=0)
            batch_preds.append(p)
        p_all = np.concatenate(batch_preds, axis=0)
        preds_sum = p_all if preds_sum is None else (preds_sum + p_all)
    return preds_sum / float(passes)


eve_encodings = _predict_dataset_tta(model, tta_base_ds, passes=5)
eve_encodings[:3]



## === cell 15
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_true_classes
accuracy_score(true_classes, pred_classes)



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
if "history" in globals() and hasattr(history, "history"):
    temp_hist = pd.DataFrame(history.history)
    temp_hist.to_csv("model_effnetb4_history.csv", index=False)
    temp_hist.tail()
else:
    pd.DataFrame().to_csv("model_effnetb4_history.csv", index=False)
    "history not available; wrote empty model_effnetb4_history.csv"



## === cell 19
model.save("model_effnet_b4.hdf5")



## === cell 20
model.save_weights("model_effnet_b4_weights.weights.h5")




## === cell 21
def _resolve_test_dir(base):
    candidates = [
        base,
        os.path.join(base, "test_images"),
        os.path.join(base, os.path.basename(base)),
        "/kaggle/input/paddy-disease-classification/test_images",
        "/kaggle/input/paddy-disease-classification/test_images/test_images",
    ]
    for c in candidates:
        if os.path.isdir(c):
            try:
                if any(f.lower().endswith(".jpg") for f in os.listdir(c)):
                    return c
            except Exception:
                pass
    return base


test_loc = _resolve_test_dir("/kaggle/input/paddy-disease-classification/test_images")

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
sample_path = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)
test_image_ids = sample["image_id"].astype(str).tolist()

test_files = []
missing = 0
for img_id in test_image_ids:
    p = os.path.join(test_loc, img_id)
    if os.path.exists(p):
        test_files.append(p)
    else:
        missing += 1

if len(test_files) == 0:
    test_files = [
        os.path.join(test_loc, f)
        for f in sorted(os.listdir(test_loc))
        if f.lower().endswith(".jpg")
    ]
    test_image_ids = [os.path.basename(p) for p in test_files]

assert len(test_files) > 0, f"No test images found under {test_loc}"
len(test_files), test_files[0], missing



## === cell 24
test_mean = tf.constant(
    getattr(test_generator, "mean", np.zeros((3,), dtype=np.float32)), dtype=tf.float32
)
test_std = tf.constant(
    getattr(test_generator, "std", np.ones((3,), dtype=np.float32)), dtype=tf.float32
)


def _standardize_test_single_from_uint8(img_uint8):
    img = tf.ensure_shape(img_uint8, (input_size, input_size, 3))
    x = tf.cast(img, tf.float32) * (1.0 / 255.0)
    x = (x - test_mean) / (test_std + 1e-7)
    x = tf.ensure_shape(x, (input_size, input_size, 3))
    return x


def _standardize_test_batch_from_uint8(img_uint8):
    img = tf.ensure_shape(img_uint8, (None, input_size, input_size, 3))
    x = tf.cast(img, tf.float32) * (1.0 / 255.0)
    x = (x - test_mean) / (test_std + 1e-7)
    x = tf.ensure_shape(x, (None, input_size, input_size, 3))
    return x


test_base_ds = tf.data.Dataset.from_tensor_slices(
    tf.constant(np.array(test_files, dtype=str), dtype=tf.string)
)
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
            x = _standardize_test_batch_from_uint8(aug_u8)
            p = model.predict(x, batch_size=batch_size, verbose=0)
            batch_preds.append(p)
        if len(batch_preds) == 0:
            raise RuntimeError(
                f"Test prediction produced no batches. Check test_loc={test_loc} and discovered files."
            )
        p_all = np.concatenate(batch_preds, axis=0)
        preds_sum = p_all if preds_sum is None else (preds_sum + p_all)
    return preds_sum / float(passes)


test_encodings = _predict_dataset_tta_test(model, test_base_ds, passes=5)
test_encodings[:3]



## === cell 25
predict_max = np.argmax(test_encodings, axis=1)
predict_max[:10]



## === cell 26
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[int(k)] for k in predict_max]
predictions[:10]



## === cell 27
sub = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill_value = pd.Series(predictions).mode().iloc[0] if len(predictions) else "normal"
    sub["label"] = sub["label"].fillna(fill_value)

sub = sub[["image_id", "label"]]
assert list(sub.columns) == ["image_id", "label"]
sub.to_csv("submission.csv", index=False)
sub.head()

## --- ERROR in outputing the csv:
Invalid submission: Expected columns {'label', 'image_id'}, but got set()

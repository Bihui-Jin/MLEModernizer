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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    if Version(_pb.__version__) >= Version("5.0.0"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
        importlib.reload(_pb)
except Exception:
    pass

import random
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow_hub as hub
import seaborn as sns
import cv2
import albumentations as A

from albumentations.core.composition import Compose
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 25
lr = 1e-4
valid_split = 0.2
input_size = 128
batch_size = 16
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy

N_WORKERS = min(8, (os.cpu_count() or 2))
USE_MULTIPROCESSING = True
MAX_QUEUE_SIZE = 32



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)



## === cell 3
src = "../input/paddy-disease-classification/train_images/dead_heart/100008.jpg"
img = img_to_array(load_img(src), dtype="uint8")




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
            rv = np.random.randint(0, image.shape[0])

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
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 8, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1).numpy()

    return image


def test_time_augmentation_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)

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
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
)

valid_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
)



## === cell 6
to_gen_fit = []

train_images_root = "../input/paddy-disease-classification/train_images"

class_dirs = [
    d
    for d in sorted(os.listdir(train_images_root))
    if os.path.isdir(os.path.join(train_images_root, d))
]

files_to_fit = []
for cls in class_dirs:
    cls_dir = os.path.join(train_images_root, cls)
    jpgs = sorted(
        f
        for f in os.listdir(cls_dir)
        if os.path.isfile(os.path.join(cls_dir, f)) and f.lower().endswith(".jpg")
    )
    if jpgs:
        files_to_fit.append(os.path.join(cls_dir, jpgs[0]))

for file in files_to_fit:
    to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))



## === cell 7
generator.fit(to_gen_fit)



## === cell 8
train_batch = next(train_datagen)[0]
valid_batch = next(valid_datagen)[0]
len(train_batch), len(valid_batch)



## === cell 9
DO_PLOTS = False

if DO_PLOTS:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()
    for i, arr in enumerate(train_batch):
        img = array_to_img(arr)
        axes[i].imshow(img)
    plt.show()



## === cell 10
if DO_PLOTS:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()
    for i, arr in enumerate(valid_batch):
        img = array_to_img(arr)
        axes[i].imshow(img)
    plt.show()



## === cell 11
HUB_CACHE_DIR = os.path.join(os.getcwd(), "tfhub_cache")
os.makedirs(HUB_CACHE_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = HUB_CACHE_DIR


class HubFeatureVector(tf.keras.layers.Layer):
    def __init__(self, handle, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self._handle = handle
        self._module = hub.load(handle)

    def call(self, inputs, training=None):
        out = self._module(inputs)
        if isinstance(out, dict):
            out = out.get("default", next(iter(out.values())))
        return out


inputs = tf.keras.Input(shape=(input_size, input_size, 3))
x = HubFeatureVector(
    "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    trainable=True,
)(inputs)
outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)



## === cell 12
model.build([None, input_size, input_size, 3])

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=True)



## === cell 13
model.summary()



## === cell 14
expected_classes = [
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

train_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    classes=expected_classes,
)

valid_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    classes=expected_classes,
)

history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
)


## === cell 15
tta_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
)



## === cell 16
n_val = tta_datagen.samples
eve_encodings = np.zeros((n_val, classes), dtype=np.float32)

for _ in range(5):
    encodings = model.predict(
        tta_datagen,
        verbose=1,
        workers=N_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        max_queue_size=MAX_QUEUE_SIZE,
    )
    eve_encodings += encodings.astype(np.float32, copy=False)

eve_encodings /= 5.0
eve_encodings



## === cell 17
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes

accuracy_score(true_classes, pred_classes)



## === cell 18
if DO_PLOTS:
    plt.figure(figsize=[12, 6], dpi=300)
    sns.lineplot(
        x=list(range(len(history.history["accuracy"]))),
        y=history.history["accuracy"],
        label="train",
    )
    sns.lineplot(
        x=list(range(len(history.history["val_accuracy"]))),
        y=history.history["val_accuracy"],
        label="validation",
    )
    plt.show()



## === cell 19
if DO_PLOTS:
    plt.figure(figsize=[12, 6], dpi=300)
    sns.lineplot(
        x=list(range(len(history.history["loss"]))),
        y=history.history["loss"],
        label="train",
    )
    sns.lineplot(
        x=list(range(len(history.history["val_loss"]))),
        y=history.history["val_loss"],
        label="validation",
    )
    plt.show()



## === cell 20
temp = pd.DataFrame(history.history)
temp.to_csv("model_effnetb4_history.csv", index=False)



## === cell 21
model.save("model_effnet_b4.hdf5")



## === cell 22
model.save_weights("model_effnet_b4_weights.hdf5")



## === cell 23
test_loc = "../input/paddy-disease-classification/test_images"

test_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn,
)



## === cell 24
test_generator.fit(to_gen_fit)



## === cell 25
test_datagen = test_generator.flow_from_directory(
    directory=test_loc,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=["."],
    shuffle=False,
)



## === cell 26
if DO_PLOTS:
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
    axes = axes.ravel()
    test_batch = next(test_datagen)[0]
    for i, arr in enumerate(test_batch):
        img = array_to_img(arr)
        axes[i].imshow(img)
    plt.show()



## === cell 27
n_test = test_datagen.samples
test_encodings = np.zeros((n_test, classes), dtype=np.float32)

for _ in range(5):
    encodings = model.predict(
        test_datagen,
        verbose=1,
        workers=N_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
        max_queue_size=MAX_QUEUE_SIZE,
    )
    test_encodings += encodings.astype(np.float32, copy=False)

test_encodings /= 5.0
test_encodings



## === cell 28
predict_max = np.argmax(test_encodings, axis=1)
predict_max



## === cell 29
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]



## === cell 30
files = test_datagen.filenames

temp = pd.DataFrame({"image_id": files, "label": predictions})

temp.image_id = temp.image_id.str.replace("./", "", regex=False)
temp.to_csv("model_submission_v17.csv", index=False)
temp



## === cell 31
temp.label.value_counts()

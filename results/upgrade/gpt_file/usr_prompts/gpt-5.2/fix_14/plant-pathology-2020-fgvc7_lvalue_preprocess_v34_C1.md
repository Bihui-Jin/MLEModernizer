# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras

tqdm.pandas()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
USE_TPU = "TPU_NAME" in os.environ
if USE_TPU:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.MirroredStrategy()

print("Using strategy:", type(strategy).__name__)



## === cell 2
IMAGE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
assert (
    sub.columns.tolist() == ["image_id"] + TARGET_COLS
), "Unexpected submission columns"
assert (
    train_data.columns.tolist() == ["image_id"] + TARGET_COLS
), "Unexpected train columns"
assert test_data.columns.tolist() == ["image_id"], "Unexpected test columns"

print("Train:", train_data.shape, "Test:", test_data.shape)




## === cell 3
def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = cv2.GC_PR_FGD
    mask[2 * h // 5 : 3 * h // 5, 2 * w // 5 : 3 * w // 5] = cv2.GC_FGD
    return mask


def remove_background(image, h=136, w=205):
    orig_image = image
    image = cv2.resize(image, (w, h))
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    h0, w0 = orig_image.shape[:2]
    mask_binary = cv2.resize(mask_binary, (w0, h0))
    result = cv2.bitwise_and(orig_image, orig_image, mask=mask_binary)
    return result




## === cell 4
def rotate(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]
    x = tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    return tf.image.resize(x, shape)


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


_SCALES = np.arange(0.8, 1.0, 0.01, dtype=np.float32)
_BOXES = np.zeros((len(_SCALES), 4), dtype=np.float32)
for i, scale in enumerate(_SCALES):
    x1 = y1 = 0.5 - (0.5 * float(scale))
    x2 = y2 = 0.5 + (0.5 * float(scale))
    _BOXES[i] = [x1, y1, x2, y2]
_BOX_INDICES = np.zeros(len(_SCALES), dtype=np.int32)

_ZOOM_BOXES_T = tf.constant(_BOXES, dtype=tf.float32)
_ZOOM_BOXIND_T = tf.constant(_BOX_INDICES, dtype=tf.int32)
_NUM_SCALES = len(_SCALES)


def zoom(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]

    def random_crop(img):
        crops = tf.image.crop_and_resize(
            [img], boxes=_ZOOM_BOXES_T, box_indices=_ZOOM_BOXIND_T, crop_size=shape
        )
        return crops[
            tf.random.uniform(shape=[], minval=0, maxval=_NUM_SCALES, dtype=tf.int32)
        ]

    choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
    return x




## === cell 5
def get_train_val_test_datasets(
    val_fold: int,
    preprocess=True,
    augment=True,
    IMAGE_SIZE=(408, 615),
    nfolds=5,
    batch_size=32,
):
    image_h, image_w = IMAGE_SIZE

    @tf.function
    def _read_decode_resize(image_id):
        path = tf.strings.join([IMAGE_PATH, image_id, ".jpg"])
        bytes_ = tf.io.read_file(path)
        img = tf.io.decode_jpeg(bytes_, channels=3)  # RGB
        img = tf.image.resize(
            img, [image_h, image_w], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        img = tf.ensure_shape(img, [image_h, image_w, 3])
        return img

    def _maybe_remove_bg(img):
        def _cv_remove(x):
            x = x.numpy().astype(np.uint8)
            y = remove_background(x)
            return y.astype(np.uint8)

        out = tf.py_function(_cv_remove, [tf.cast(img, tf.uint8)], Tout=tf.uint8)
        out = tf.ensure_shape(out, [None, None, 3])
        out = tf.image.resize(
            out, [image_h, image_w], method=tf.image.ResizeMethod.BILINEAR
        )
        out = tf.cast(out, tf.float32)
        out = tf.ensure_shape(out, [image_h, image_w, 3])
        return out

    @tf.function
    def map_base_from_id(image_id, label):
        img = _read_decode_resize(image_id)
        if preprocess:
            img = _maybe_remove_bg(img)
        img = img / 255.0
        label = (tf.cast(label, tf.float32) + 0.01) / 1.04  # keep original smoothing
        return img, label

    @tf.function
    def aug_map_single(image, label):
        image = flip(image)
        image = color(image)
        image = rotate(image)
        image = zoom(image)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    n = len(train_data)
    fold_size = n // nfolds
    start = val_fold * fold_size
    end = (val_fold + 1) * fold_size if val_fold < nfolds - 1 else n

    train_ids = train_data["image_id"].values.astype(str)
    labels = train_data[TARGET_COLS].values.astype(np.float32)

    idx = np.arange(n, dtype=np.int32)
    val_mask = (idx >= start) & (idx < end)
    train_mask = ~val_mask

    train_ids_sel = train_ids[train_mask]
    train_y_sel = labels[train_mask]
    val_ids_sel = train_ids[val_mask]
    val_y_sel = labels[val_mask]

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.threading.private_threadpool_size = max(1, (os.cpu_count() or 4) // 2)
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_ids_sel, train_y_sel)
    ).with_options(options)
    val_ds = tf.data.Dataset.from_tensor_slices((val_ids_sel, val_y_sel)).with_options(
        options
    )
    test_ids = test_data["image_id"].values.astype(str)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ids).with_options(options)

    @tf.function
    def map_test(image_id):
        img = _read_decode_resize(image_id)
        if preprocess:
            img = _maybe_remove_bg(img)
        img = img / 255.0
        return img

    train_ds = train_ds.map(
        map_base_from_id, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    val_ds = val_ds.map(
        map_base_from_id, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.map(
        map_test, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        int(train_mask.sum()),
        int(val_mask.sum()),
        aug_map_single,
    )




## === cell 6
BATCH_SIZE = 8
VAL_FOLD = 0

train_ds, val_ds, test_ds, n_train, n_val, _AUG_MAP_SINGLE = (
    get_train_val_test_datasets(
        val_fold=VAL_FOLD, preprocess=False, augment=True, batch_size=BATCH_SIZE
    )
)




## === cell 7
def get_model():
    model = keras.Sequential()
    model.add(
        keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            input_shape=(408, 615, 3),
            pooling=None,
        )
    )
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.summary()
    return model




## === cell 8
with strategy.scope():
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy],
        steps_per_execution=32,  # unchanged; reduces Python overhead, same math/updates
    )



## === cell 9
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

callbacks = [
    EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=0.000001),
    CSVLogger("log.csv", append=True, separator=";"),
]

train_ds_epoch = train_ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
train_ds_epoch = train_ds_epoch.take(n_train)
train_ds_epoch = train_ds_epoch.map(
    _AUG_MAP_SINGLE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)
train_ds_epoch = train_ds_epoch.batch(BATCH_SIZE, drop_remainder=True).prefetch(
    tf.data.AUTOTUNE
)

val_ds_epoch = val_ds.take(n_val)
val_ds_epoch = val_ds_epoch.batch(BATCH_SIZE, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
validation_steps = math.ceil(n_val / BATCH_SIZE)

history = None
try:
    history = model.fit(
        train_ds_epoch,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_ds_epoch,
        validation_steps=validation_steps,
        validation_freq=1,
        verbose=1,
        callbacks=callbacks,
    )
except tf.errors.ResourceExhaustedError as e:
    print(
        "WARNING: OOM during training; proceeding to prediction with current weights."
    )
    print(str(e)[:500])



## === cell 10
test_ds_full = test_ds.take(len(test_data))
test = test_ds_full.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

test_pr = model.predict(test, verbose=1)
test_pr = np.asarray(test_pr)

if test_pr.ndim != 2 or test_pr.shape[1] != 4:
    raise ValueError(f"Unexpected prediction shape: {test_pr.shape}")

if test_pr.shape[0] != len(test_data):
    raise ValueError(
        f"Prediction count mismatch: got {test_pr.shape[0]} preds for {len(test_data)} test rows"
    )

submission = pd.DataFrame({"image_id": test_data["image_id"].values})
for i, c in enumerate(TARGET_COLS):
    submission[c] = test_pr[:, i]

submission = submission[["image_id"] + TARGET_COLS]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to have 183 rows but got 7

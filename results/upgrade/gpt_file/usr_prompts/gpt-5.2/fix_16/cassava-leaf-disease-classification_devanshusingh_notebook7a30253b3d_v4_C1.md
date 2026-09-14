# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)




## === cell 1
def _pick_existing_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_csv_path = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
)
label_json_path = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    "/kaggle/data/cassava-leaf-disease-classification/label_num_to_disease_map.json",
)
images_dir_path = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
)

train_tfrecords_dir = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/train_tfrecords",
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords",
    "/kaggle/data/cassava-leaf-disease-classification/train_tfrecords",
)
test_tfrecords_dir = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/test_tfrecords",
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords",
    "/kaggle/data/cassava-leaf-disease-classification/test_tfrecords",
)

print("train_csv_path:", train_csv_path)
print("label_json_path:", label_json_path)
print("images_dir_path:", images_dir_path)
print("train_tfrecords_dir:", train_tfrecords_dir)
print("test_tfrecords_dir:", test_tfrecords_dir)



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print(train_csv.head())
print(train_csv.shape)



## === cell 4
train_csv = train_csv.reset_index(drop=True)



## === cell 5
print(train_csv.dtypes)



## === cell 6
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 7
BATCH_SIZE = 50
IMG_SIZE = 200



## === cell 8
train_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1 / 255.0,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2,
)



## === cell 9
AUTOTUNE = tf.data.AUTOTUNE

n_total = len(train_csv)
perm = np.random.RandomState(SEED).permutation(n_total)
n_valid = int(round(0.2 * n_total))
valid_idx = perm[:n_valid]
train_idx = perm[n_valid:]

train_df = train_csv.iloc[train_idx].copy()
valid_df = train_csv.iloc[valid_idx].copy()

train_paths = (images_dir_path.rstrip("/") + "/" + train_df["image_id"]).to_numpy()
valid_paths = (images_dir_path.rstrip("/") + "/" + valid_df["image_id"]).to_numpy()
train_labels = train_df["label"].astype(np.int32).to_numpy()
valid_labels = valid_df["label"].astype(np.int32).to_numpy()

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=40.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augment",
)

USE_TFDATA_FOR_TRAINING = True  # keep as in original (intended speed path)


@tf.function
def _decode_resize_and_scale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_tfrecord(example_proto, with_label: bool):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    if with_label:
        y = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=5)
        return img, y
    return img, ex["image_name"]


def _list_tfrec_files(tfrec_dir: str, prefix: str):
    if not os.path.isdir(tfrec_dir):
        return []
    files = []
    for f in sorted(os.listdir(tfrec_dir)):
        if f.startswith(prefix) and f.endswith(".tfrec"):
            files.append(os.path.join(tfrec_dir, f))
    return files


def _make_tfrec_ds(tfrec_files, training: bool):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = tf.data.Dataset.from_tensor_slices(tfrec_files)
    if training:
        ds = ds.shuffle(len(tfrec_files), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
        cycle_length=min(16, len(tfrec_files)) if tfrec_files else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda e: _parse_tfrecord(e, with_label=True), num_parallel_calls=AUTOTUNE
    )

    ds = ds.cache()

    if training:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _make_ds(paths, labels, training: bool, cache_path: str = None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
        )

    def _load_and_label(p, y):
        x = _decode_resize_and_scale(p)
        y = tf.one_hot(y, depth=5)
        return x, y

    ds = ds.map(_load_and_label, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    if training:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_tfrec_files = _list_tfrec_files(train_tfrecords_dir, "ld_train")
use_tfrec = len(train_tfrec_files) > 0

if USE_TFDATA_FOR_TRAINING:
    if use_tfrec:
        n_files = len(train_tfrec_files)
        n_valid_files = max(1, int(round(0.2 * n_files)))
        valid_tfrec_files = train_tfrec_files[:n_valid_files]
        train_tfrec_files_ = train_tfrec_files[n_valid_files:]
        train_ds = _make_tfrec_ds(train_tfrec_files_, training=True)
        valid_ds = _make_tfrec_ds(valid_tfrec_files, training=False)
    else:
        train_ds = _make_ds(train_paths, train_labels, training=True, cache_path=None)
        valid_ds = _make_ds(valid_paths, valid_labels, training=False, cache_path=None)
    train_generator = None
    valid_generator = None
else:
    train_ds = None
    valid_ds = None
    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_csv,
        directory=images_dir_path,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        subset="training",
        validate_filenames=False,
    )
    valid_generator = valid_gen.flow_from_dataframe(
        dataframe=train_csv,
        directory=images_dir_path,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
        subset="validation",
        validate_filenames=False,
    )



## === cell 10
SHOW_BATCH = False

if SHOW_BATCH and train_generator is not None:
    import matplotlib.pyplot as plt

    batch = next(train_generator)
    images = batch[0]
    labels = batch[1]

    plt.figure(figsize=(12, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(2, 3, i % 6 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[int(np.argmax(label))])
        if i == 15:
            break



## === cell 11
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 12
def scheduler(epoch, lr):
    if epoch > 6 and epoch % 2 == 0:
        lr = lr / 1.5
        return lr
    else:
        return lr


callback0 = tf.keras.callbacks.ModelCheckpoint(
    "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)
callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 13
PRETRAINED_PATH = _pick_existing_path(
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5",
    "/kaggle/input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5",
)
loaded_pretrained = False
try:
    model = tf.keras.models.load_model(PRETRAINED_PATH)
    loaded_pretrained = True
    print("Loaded saved model:", PRETRAINED_PATH)
except Exception as e:
    print("No saved model. So Train the model !")
    print("Load error:", repr(e))



## === cell 14
his = None
if not loaded_pretrained:
    import math
    import inspect

    x_train = train_ds if USE_TFDATA_FOR_TRAINING else train_generator
    x_valid = valid_ds if USE_TFDATA_FOR_TRAINING else valid_generator

    fit_sig = inspect.signature(model.fit)
    fit_kwargs = {}
    workers = max(2, (os.cpu_count() or 4) // 2)
    for k, v in {
        "workers": workers,
        "use_multiprocessing": False,
        "max_queue_size": 20,
    }.items():
        if k in fit_sig.parameters:
            fit_kwargs[k] = v

    if USE_TFDATA_FOR_TRAINING:
        steps_per_epoch = None
        validation_steps = None
    else:
        steps_per_epoch = max(1, int(math.ceil(train_generator.n / BATCH_SIZE)))
        validation_steps = max(1, int(math.ceil(valid_generator.n / BATCH_SIZE)))

    his = model.fit(
        x=x_train,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=x_valid,
        validation_steps=validation_steps,
        callbacks=[callback0, callback1],
        verbose=2,
        **fit_kwargs,
    )



## === cell 15
RUN_VALID_PRED_DEBUG = False
if RUN_VALID_PRED_DEBUG:
    if USE_TFDATA_FOR_TRAINING:
        xb, yb = next(iter(valid_ds))
        print(model.predict(xb))
    else:
        print(model.predict(next(valid_generator)[0]))



## === cell 16
if his is not None:
    stats = pd.DataFrame(his.history)
    print(stats.tail())
else:
    print("Training skipped (pretrained model loaded).")



## === cell 17
SHOW_TEST_EXAMPLE = False

test_dir = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
)
if SHOW_TEST_EXAMPLE:
    import cv2
    import matplotlib.pyplot as plt

    test_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    test_img_path = os.path.join(test_dir, test_images[0]) if test_images else None
    print("Example test image:", test_img_path)

    if test_img_path is not None:
        img = cv2.imread(test_img_path)
        if img is not None:
            resized_img = (
                cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                / 255.0
            )
            plt.figure(figsize=(8, 4))
            plt.title("TEST IMAGE")
            plt.imshow(resized_img[0][:, :, ::-1])  # BGR->RGB for display
            plt.axis("off")
        else:
            print("cv2.imread failed for:", test_img_path)



## === cell 18
ss_path = _pick_existing_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
)
ss = pd.read_csv(ss_path)

test_tfrec_files = _list_tfrec_files(test_tfrecords_dir, "ld_test")

if len(test_tfrec_files) > 0:
    ds = tf.data.Dataset.from_tensor_slices(test_tfrec_files)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
        cycle_length=min(16, len(test_tfrec_files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).with_options(opts)

    ds = ds.map(
        lambda e: _parse_tfrecord(e, with_label=False), num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    probs = model.predict(ds.map(lambda x, name: x), verbose=0)
    preds = np.argmax(probs, axis=1).astype(int).tolist()

    my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
else:
    test_paths = (test_dir.rstrip("/") + "/" + ss["image_id"]).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices(test_paths)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    ds = ds.map(_decode_resize_and_scale, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    probs = model.predict(ds, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int).tolist()

    my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})

my_submission.to_csv("submission.csv", index=False)



## === cell 19
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote:", os.path.abspath("submission.csv"), "rows:", len(my_submission))
assert os.path.basename("submission.csv").endswith(".csv")
assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(pd.read_csv(ss_path))

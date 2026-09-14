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

No external packages required in the script and installed.

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

# 5. Target score

0.8773043215472952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.1136) has done: 'Main bottlenecks are (1) heavy Python-level per-image prediction loops calling `model.predict()` 4× per test image, and (2) rebuilding Albumentations `Compose` objects on every single augmentation call. I keep the exact same TTA set and averaging semantics, but batch the entire test set through the model (4 forward passes total instead of ~10k) and pre-create the Albumentations pipelines once, applying them in a tight loop. I also make test image loading use a `tf.data` pipeline with parallel decode/resize and prefetch to remove Python overhead and improve throughput without changing any values (still RGB, resized, scaled by 1/255). These changes are provably equivalent to the original logic aside from negligible float-order effects, and should bring runtime under 600s.'

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR:", INPUT_DIR)
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

SEED = 100
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("tf:", tf.__version__)
print("cv2:", cv2.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(train.head())
print("train shape:", train.shape)



## === cell 3
import json

with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json"), "r") as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
print(train["class"].value_counts())



## === cell 4
plt.figure(figsize=(15, 7))
sns.countplot(x=train["class"], order=train["class"].value_counts().index)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()



## === cell 5
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))

train_df = train.copy()
train_df["label"] = train_df["label"].astype(str)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.05,
    random_state=SEED,
    stratify=train_df["label"].values,
)

print("train_df:", train_df.shape, "val_df:", val_df.shape)



## === cell 6
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 5


def preprocess_cv2_train(img):
    if img.dtype != np.uint8:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    if np.random.rand() < 0.5:
        img_u8 = cv2.flip(img_u8, 1)

    if np.random.rand() < 0.5:
        img_u8 = cv2.flip(img_u8, 0)

    if np.random.rand() < 0.5:
        angle = float(np.random.uniform(-40.0, 40.0))
        h, w = img_u8.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_u8 = cv2.warpAffine(
            img_u8,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    return img_u8.astype(np.float32)


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_cv2_train,
    rescale=1.0 / 255.0,
).flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_PATH,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
).flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_PATH,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Class indices:", train_gen.class_indices)




## === cell 7
def build_model(input_shape=(512, 512, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


model_path = "../input/mdpa56/initialweightInceptionResnet4.h5"

model2 = None
if os.path.exists(model_path):
    model2 = tf.keras.models.load_model(model_path, compile=False)
    print("Loaded external model:", model_path)
else:
    print("External model not found at:", model_path)
    print("Training a small fallback CNN to produce a valid submission.")
    model2 = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES
    )
    model2.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, verbose=1),
        EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "fallback_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    steps_per_epoch = max(1, train_gen.n // train_gen.batch_size)
    val_steps = max(1, val_gen.n // val_gen.batch_size)

    history = model2.fit(
        train_gen,
        validation_data=val_gen,
        epochs=8,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
        workers=min(4, (os.cpu_count() or 2)),
        use_multiprocessing=True,
        max_queue_size=16,
    )




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1601495791.py in <cell line: 0>()
     51     # This preserves the exact same generator/augmentation logic, but parallelizes the
     52     # Python/cv2 preprocessing so GPU/TF time isn't spent waiting on image augmentation.
---> 53     history = model2.fit(
     54         train_gen,
     55         validation_data=val_gen,

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

## === cell 8
def tta_flip_v_batch(imgs_01):
    return imgs_01[:, ::-1, :, :]


def tta_flip_h_batch(imgs_01):
    return imgs_01[:, :, ::-1, :]


def tta_grid_dropout_batch(imgs_01, ratio=0.5, holes_x=5, holes_y=5):
    imgs = imgs_01.copy()
    n, h, w, c = imgs.shape
    cell_h = max(1, h // holes_y)
    cell_w = max(1, w // holes_x)
    for i in range(n):
        mask = np.ones((h, w), dtype=np.float32)
        for yy in range(holes_y):
            for xx in range(holes_x):
                if np.random.rand() < ratio:
                    y0 = yy * cell_h
                    x0 = xx * cell_w
                    y1 = h if yy == holes_y - 1 else (y0 + cell_h)
                    x1 = w if xx == holes_x - 1 else (x0 + cell_w)
                    mask[y0:y1, x0:x1] = 0.0
        imgs[i] = imgs[i] * mask[..., None]
    return imgs




## === cell 9
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_image_ids = sample_sub["image_id"].tolist()

TEST_DIR = TEST_PATH if TEST_PATH.endswith("/") else (TEST_PATH + "/")
assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"


def _decode_resize_scale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function(jit_compile=False)
def _grid_dropout_tf(imgs, ratio, holes_x, holes_y, seed_pair):
    b = tf.shape(imgs)[0]
    h = tf.shape(imgs)[1]
    w = tf.shape(imgs)[2]

    cell_h = tf.maximum(1, h // holes_y)
    cell_w = tf.maximum(1, w // holes_x)

    rnd = tf.random.stateless_uniform(
        shape=(b, holes_y, holes_x), seed=seed_pair, dtype=tf.float32
    )
    keep = tf.cast(rnd >= ratio, tf.float32)  # 1 means keep, 0 means drop

    mask = tf.repeat(keep, repeats=cell_h, axis=1)
    mask = tf.repeat(mask, repeats=cell_w, axis=2)
    mask = mask[:, :h, :w]  # crop (last cells may overshoot)
    mask = mask[..., None]  # [B,H,W,1]
    return imgs * mask


test_paths = [os.path.join(TEST_DIR, image_id) for image_id in test_image_ids]

options = tf.data.Options()
options.experimental_deterministic = True

path_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
img_ds = (
    path_ds.map(_decode_resize_scale, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


@tf.function(jit_compile=False)
def _infer_probs(x):
    return model2(x, training=False)


pred_chunks = []
for batch_idx, batch_imgs in enumerate(img_ds):
    tta_v = tf.reverse(batch_imgs, axis=[1])  # vertical flip
    tta_h = tf.reverse(batch_imgs, axis=[2])  # horizontal flip
    tta_d = _grid_dropout_tf(
        batch_imgs,
        ratio=tf.constant(0.5, tf.float32),
        holes_x=tf.constant(5, tf.int32),
        holes_y=tf.constant(5, tf.int32),
        seed_pair=tf.constant([SEED, batch_idx], dtype=tf.int32),
    )

    tta_all = tf.concat([batch_imgs, tta_v, tta_h, tta_d], axis=0)  # [4B,H,W,C]
    probs_all = _infer_probs(tta_all)  # [4B,NUM_CLASSES]

    b = tf.shape(batch_imgs)[0]
    probs_all = tf.reshape(probs_all, (4, b, NUM_CLASSES))  # [4,B,C]
    pred_mean = tf.reduce_mean(probs_all, axis=0)  # [B,C]

    pred_chunks.append(pred_mean.numpy())

pred_mean_all = np.concatenate(pred_chunks, axis=0)
pred_labels = np.argmax(pred_mean_all, axis=1).astype(int).tolist()

print("Preds:", len(pred_labels), pred_labels[:10])



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/790276089.py in <cell line: 0>()
     63     tta_v = tf.reverse(batch_imgs, axis=[1])  # vertical flip
     64     tta_h = tf.reverse(batch_imgs, axis=[2])  # horizontal flip
---> 65     tta_d = _grid_dropout_tf(
     66         batch_imgs,
     67         ratio=tf.constant(0.5, tf.float32),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node mul defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/790276089.py", line 65, in <cell line: 0>

  File "/tmp/ipykernel_11/790276089.py", line 34, in _grid_dropout_tf

Incompatible shapes: [32,512,512,3] vs. [32,510,510,1]
	 [[{{node mul}}]] [Op:__inference__grid_dropout_tf_260]

## === cell 10
submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2118804665.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
      2 submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
      3 submission.to_csv(submission_path, index=False)
      4 
      5 print("Wrote:", submission_path)

NameError: name 'pred_labels' is not defined

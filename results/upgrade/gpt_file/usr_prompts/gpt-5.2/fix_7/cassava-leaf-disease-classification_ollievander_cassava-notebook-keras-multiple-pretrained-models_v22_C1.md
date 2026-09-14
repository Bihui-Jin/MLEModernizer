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

# 5. Code solution

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
print("Class distribution (by name):")
print(train["class"].value_counts())



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


def preprocess_tf_train(img):
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, tf.float32)

    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED + 1)

    angle_rad = tf.random.uniform(
        shape=(),
        minval=-40.0 * np.pi / 180.0,
        maxval=40.0 * np.pi / 180.0,
        seed=SEED + 2,
    )
    img = tf.image.rot90(
        img,
        k=tf.random.uniform(
            shape=(), minval=0, maxval=4, dtype=tf.int32, seed=SEED + 3
        ),
    )

    return img


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_tf_train,
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
    )




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

    rnd = tf.random.stateless_uniform(
        shape=(b, holes_y, holes_x, 1), seed=seed_pair, dtype=tf.float32
    )
    keep = tf.cast(rnd >= ratio, tf.float32)  # 1 keep, 0 drop

    mask = tf.image.resize(
        keep, size=(h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
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



## === cell 10
submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())

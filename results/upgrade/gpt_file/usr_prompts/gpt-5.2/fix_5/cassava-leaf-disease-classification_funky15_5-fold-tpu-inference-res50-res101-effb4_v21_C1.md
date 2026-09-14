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
import os, glob, math, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

print("Tensorflow version " + tf.__version__)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.10, seed=SEED),
        layers.RandomContrast(0.10, seed=SEED),
    ],
    name="data_augmentation",
)


def _load_image_pil(path, image_size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.float32)


def _preprocess_np(img_np, normalize=True):
    if normalize:
        img_np = img_np / 255.0
    return img_np




## === cell 1
def build_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs

    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base.trainable = True

    x = base.output
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="effnetb4_cassava")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)

train_df["filepath"] = (TRAIN_DIR + "/" + train_df["image_id"].astype(str)).values

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
trn_df = train_df.iloc[val_size:].reset_index(drop=True)


def make_dataset(df, training, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    paths = df["filepath"].values
    labels = df["label"].values.astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _read(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size], method="bilinear")
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        if training:
            img = data_augmentation(img, training=True)
        return img, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_read, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if not training:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(trn_df, training=True)
val_ds = make_dataset(val_df, training=False)

print("Train size:", len(trn_df), "Val size:", len(val_df))



## === cell 3
model = build_model()

EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

mod_lst = [model]




## === cell 4
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(p) for p in files]
    n = len(files)
    if n == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    paths_tf = tf.constant(files)

    def _read_only(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32)
        if normalize:
            img = img * (1.0 / 255.0)
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.apply_default_optimizations = True

    base_ds = tf.data.Dataset.from_tensor_slices(paths_tf).with_options(options)
    base_ds = base_ds.map(
        _read_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).cache()
    base_ds = base_ds.apply(tf.data.experimental.ignore_errors())

    if TTA:
        infer_ds = base_ds.repeat(aug_num).map(
            lambda img: data_augmentation(img, training=True),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        total = n * aug_num
    else:
        infer_ds = base_ds
        total = n

    infer_ds = infer_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    sum_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    for mod in model_obj_list:
        probs = mod.predict(infer_ds, verbose=0)  # shape: (total, NUM_CLASSES)
        probs = np.asarray(probs, dtype=np.float32)

        if TTA:
            probs = probs.reshape((aug_num, n, NUM_CLASSES)).mean(axis=0)

        sum_probs += probs

    avg_probs = sum_probs / float(len(model_obj_list))
    preds = avg_probs.argmax(axis=1).astype(int)

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 5
sample_df = pd.read_csv(SAMPLE_SUB)

predict_df = get_preds_model_list(
    TEST_DIR, mod_lst, normalize=True, aug_num=5, TTA=True
)

predict_df = sample_df[["image_id"]].merge(predict_df, on="image_id", how="left")
predict_df["label"] = predict_df["label"].fillna(0).astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())



## === cell 6
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(pd.read_csv(SAMPLE_SUB))
print("submission.csv format OK")

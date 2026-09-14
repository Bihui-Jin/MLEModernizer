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
import json
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"




## === cell 2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

print("TF version:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("Labels:", label_list, "NUM_CLASSES:", NUM_CLASSES)




## === cell 4
train_df_tmp = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
sample_sub_tmp = pd.read_csv(SAMPLE_SUB_CSV, usecols=["image_id", "label"])
print(f"Number of train images (from CSV): {len(train_df_tmp)}")
print(f"Number of test images (from sample_submission): {len(sample_sub_tmp)}")




## === cell 5
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/xceptionv8/Cassava_Best_Xception_Model_V08.hdf5"
PRETRAINED_EXISTS = os.path.exists(PRE_TRAINED_MODEL)
print("Pretrained path exists:", PRETRAINED_EXISTS, PRE_TRAINED_MODEL)




## === cell 6
try:
    import albumentations as A  # type: ignore

    AUGMENTATIONS_TRAIN = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
                p=0.5,
            ),
            A.CenterCrop(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
            A.ToFloat(max_value=255.0),
        ]
    )

    AUGMENTATIONS_TEST = A.Compose(
        [
            A.CenterCrop(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
            A.ToFloat(max_value=255.0),
        ]
    )
except Exception as e:
    print("albumentations unavailable; using fallback augmentations. Reason:", repr(e))

    class _FallbackCompose:
        def __init__(self, height, width):
            self.height = int(height)
            self.width = int(width)

        def __call__(self, image):
            h, w = image.shape[:2]
            ch, cw = self.height, self.width
            top = max(0, (h - ch) // 2)
            left = max(0, (w - cw) // 2)
            cropped = image[top : top + ch, left : left + cw]
            if cropped.shape[0] != ch or cropped.shape[1] != cw:
                pil = Image.fromarray(image)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                pil = pil.resize((cw, ch), resample)
                cropped = np.asarray(pil)
            out = cropped.astype(np.float32) / 255.0
            return {"image": out}

    AUGMENTATIONS_TRAIN = _FallbackCompose(IMG_HEIGHT, IMG_WIDTH)
    AUGMENTATIONS_TEST = _FallbackCompose(IMG_HEIGHT, IMG_WIDTH)




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _build_path(data_type, image_id):
    base = TEST_DIR if data_type == "TEST_DATA" else TRAIN_DIR
    return tf.strings.join([base, image_id], separator=os.sep)


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment_train_tf(img, seed):
    apply_aug = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0) > 0.5

    if "albumentations" in globals() and "A" in globals():
        def _alb_aug(x_np):
            x_uint8 = (np.clip(x_np * 255.0, 0, 255)).astype(np.uint8)
            out = AUGMENTATIONS_TRAIN(image=x_uint8)[
                "image"
            ]  # float32 [0,1], center-cropped to 500x500
            return out.astype(np.float32)

        aug = tf.py_function(_alb_aug, [img], Tout=tf.float32)
        aug.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    else:
        aug = img

    return tf.cond(apply_aug, lambda: aug, lambda: img)


def make_train_dataset(image_ids, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))
    ds = ds.shuffle(
        buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
    )

    def _map_fn(idx, x):
        image_id, y = x
        path = _build_path("TRAIN_DATA", image_id)
        img = _decode_resize_normalize(path)
        seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
        img = _augment_train_tf(img, seed)
        return img, tf.cast(y, tf.int64)

    ds = ds.enumerate()
    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_dataset(data_type, image_ids, labels, batch_size):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(image_ids)

        def _map_fn(image_id):
            path = _build_path(data_type, image_id)
            img = _decode_resize_normalize(path)
            return img

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))

    def _map_fn(image_id, y):
        path = _build_path(data_type, image_id)
        img = _decode_resize_normalize(path)
        return img, tf.cast(y, tf.int64)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 8
train_df = train_df_tmp
sample_sub = sample_sub_tmp

print("train_df:", train_df.shape, "sample_sub:", sample_sub.shape)
assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_sub.columns) == {"image_id", "label"}

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
    stratify=train_df["label"].values,
)

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

print("df_train:", df_train.shape, "df_val:", df_val.shape)

train_gen = make_train_dataset(
    df_train["image_id"].values.astype(str),
    df_train["label"].values.astype(np.int64),
    batch_size=batch_size,
)
val_gen = make_eval_dataset(
    "VALIDATE_DATA",
    df_val["image_id"].values.astype(str),
    df_val["label"].values.astype(np.int64),
    batch_size=batch_size,
)




## === cell 9
def build_xception_model(img_h, img_w, num_classes):
    inp = layers.Input(shape=(img_h, img_w, 3))
    base = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    x = base.output
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model


if PRETRAINED_EXISTS:
    model = tf.keras.models.load_model(PRE_TRAINED_MODEL, compile=False)
    print("Loaded pretrained model from:", PRE_TRAINED_MODEL)
else:
    model = build_xception_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
    print("Built new Xception-based model (imagenet weights).")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 10
if not PRETRAINED_EXISTS:
    EPOCHS = 2
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
    )




## === cell 11
test_df = pd.DataFrame({"image_id": sample_sub["image_id"].values})
test_gen = make_eval_dataset(
    "TEST_DATA",
    test_df["image_id"].values.astype(str),
    labels=None,
    batch_size=batch_size,
)




## === cell 12
pred_probs = model.predict(
    test_gen,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred probs shape:", pred_probs.shape, "Pred labels shape:", pred_labels.shape)
assert len(pred_labels) == len(test_df)




## === cell 13
submission = sample_sub.copy()
submission["label"] = pred_labels
submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head(3))
print("Submission shape:", submission.shape)
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(submission.columns) == ["image_id", "label"]
assert submission.shape[0] == sample_sub.shape[0]

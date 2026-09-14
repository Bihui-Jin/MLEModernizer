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
import random
import json

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import cv2
from PIL import Image



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv3/Cassava_Best_InceptionResNet_Model_V03.hdf5"
)



## === cell 7
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    CLAHE,
    HueSaturationValue,
    CenterCrop,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    Cutout,
    ToFloat,
    ShiftScaleRotate,
)



## === cell 8
AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
        ShiftScaleRotate(
            always_apply=False,
            p=0.5,
            shift_limit=0,
            scale_limit=(0.5, 1.50),
            rotate_limit=15,
            interpolation=0,
            border_mode=0,
        ),
        ToFloat(max_value=255),
    ]
)

AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255)])



## === cell 9
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)



## === cell 10
from tensorflow.keras.utils import Sequence

_RESAMPLE = Image.Resampling.LANCZOS  # compatibility fallback only


def _fast_read_resize_rgb_u8(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = (
            Image.open(image_path)
            .convert("RGB")
            .resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
        )
        return np.asarray(img, dtype=np.uint8)
    img_bgr = cv2.resize(
        img_bgr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_test_image_f32(image_id: str) -> np.ndarray:
    image_path = os.path.join(TEST_DIR, image_id)
    try:
        img_bytes = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3", antialias=True
        )
        return tf.cast(img, tf.float32).numpy()
    except Exception:
        return _fast_read_resize_rgb_u8(image_path).astype(np.float32, copy=False)


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        return _read_test_image_f32(image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
        return _fast_read_resize_rgb_u8(image_path)


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        img_list = []
        for x in batch_x:
            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)

            img_list.append(img_data)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## === cell 11
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
test_samples



## === cell 12
test_gen = AugmentedImageSequence(
    "TEST",
    "TEST_DATA",
    test_df["image_id"].values,
    None,
    batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 13
import keras

keras.backend.clear_session()
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) - 1)
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 14
from tensorflow.keras import layers, models

num_classes = 5

PRETRAINED_EXISTS = os.path.exists(PRE_TRAINED_MODEL)

model = None
if PRETRAINED_EXISTS:
    from tensorflow.keras.models import load_model

    model = load_model(PRE_TRAINED_MODEL)
else:
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    x = layers.Dropout(0.2)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

model.summary()




## === cell 15
def old_predict():
    test_results_list = []
    for image_id in test_df["image_id"]:
        image_path = os.path.join(TEST_DIR, image_id)
        image_data = Image.open(image_path).convert("RGB")
        image_data = image_data.resize((IMG_HEIGHT, IMG_WIDTH), _RESAMPLE)
        image_data = np.array(image_data, dtype=np.float32)
        image_data = np.expand_dims(image_data, axis=0)

        if not PRETRAINED_EXISTS:
            image_data = tf.keras.applications.inception_resnet_v2.preprocess_input(
                image_data
            )

        predict_class = model.predict(image_data, verbose=0)
        test_results_list.append(
            {"image_id": image_id, "label": int(np.argmax(predict_class, axis=1)[0])}
        )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 16
_TEST_DATAGEN = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)

_GLOBAL_SEED = 42


def build_tta_views_batched_fast(
    base_batch: np.ndarray,
    tta: int,
    view_seeds: np.ndarray,
    out_buf: np.ndarray,
):
    b = base_batch.shape[0]
    out_buf[:b] = base_batch  # view 0: identity

    if tta > 1:
        for v in range(1, tta):
            np.random.seed(int(view_seeds[v - 1]))
            start = v * b
            end = start + b
            for i in range(b):
                out_buf[start + i] = _TEST_DATAGEN.random_transform(base_batch[i])

    out_buf[: tta * b] = _TEST_DATAGEN.standardize(out_buf[: tta * b])

    if not PRETRAINED_EXISTS:
        out_buf[: tta * b] = tf.keras.applications.inception_resnet_v2.preprocess_input(
            out_buf[: tta * b]
        )

    return out_buf[: tta * b]




## === cell 17
test_image_ids = test_df["image_id"].values

TTA = 6
PRED_IMG_BATCH = 64

num_test = len(test_image_ids)
idx = np.arange(num_test, dtype=np.int64)
v = np.arange(1, TTA, dtype=np.int64)[:, None]  # (TTA-1, 1)
seed_matrix = ((_GLOBAL_SEED * 1000003) ^ (idx[None, :] * 9176) ^ (v * 6361)).astype(
    np.int64
)


def _tf_decode_resize(image_id):
    path = tf.strings.join([TEST_DIR, image_id])
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3", antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img, image_id


ds = tf.data.Dataset.from_tensor_slices(test_image_ids.astype(str))
ds = ds.map(_tf_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(PRED_IMG_BATCH, drop_remainder=False)
ds = ds.prefetch(tf.data.AUTOTUNE)

tta_buf = np.empty((TTA * PRED_IMG_BATCH, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

test_results = []
append_results = test_results.extend  # minor overhead reduction

for batch_index, (base_batch, batch_ids) in enumerate(ds.as_numpy_iterator()):
    if base_batch.dtype != np.float32:
        base_batch = base_batch.astype(np.float32, copy=False)
    image_ids_chunk = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in batch_ids
    ]

    b = base_batch.shape[0]
    view_seeds = (
        seed_matrix[:, batch_index * PRED_IMG_BATCH]
        if TTA > 1
        else np.array([], dtype=np.int64)
    )

    tta_batch = build_tta_views_batched_fast(base_batch, TTA, view_seeds, tta_buf)

    preds = model.predict_on_batch(tta_batch)  # (B*TTA, num_classes)
    preds = preds.reshape(TTA, b, -1).mean(axis=0)  # (B, num_classes)
    labels = np.argmax(preds, axis=1).astype(int)
    append_results(zip(image_ids_chunk, labels.tolist()))

test_results_df = pd.DataFrame(test_results, columns=["image_id", "label"])

submission = sample_sub[["image_id"]].merge(test_results_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head(3))
print("Wrote submission.csv with shape:", submission.shape)



## === cell 18
submission = pd.read_csv("submission.csv")
submission.head(3)

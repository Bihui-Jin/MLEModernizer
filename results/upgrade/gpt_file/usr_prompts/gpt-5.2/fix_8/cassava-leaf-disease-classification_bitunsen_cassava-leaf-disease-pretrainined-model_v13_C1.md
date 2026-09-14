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
import numpy as np
import pandas as pd
import os



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import json
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
train_csv_path = os.path.join(BASE_DIR, "train.csv")
_train_df_peek = pd.read_csv(train_csv_path, usecols=["image_id"])
print(f"Number of train images (from train.csv): {len(_train_df_peek)}")



## === cell 6
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class KerasAugmentWrapper:
    def __init__(
        self,
        datagen=None,
        do_center_crop=False,
        crop_height=None,
        crop_width=None,
        to_float=True,
    ):
        self.datagen = datagen
        self.do_center_crop = do_center_crop
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.to_float = to_float

    def __call__(self, image):
        img = image
        if self.do_center_crop:
            h, w = img.shape[:2]
            ch, cw = self.crop_height, self.crop_width
            if ch is None or cw is None:
                ch, cw = h, w
            ch = min(ch, h)
            cw = min(cw, w)
            y0 = max(0, (h - ch) // 2)
            x0 = max(0, (w - cw) // 2)
            img = img[y0 : y0 + ch, x0 : x0 + cw]

        img = img.astype(np.float32, copy=False)

        if self.datagen is not None:
            batch = np.expand_dims(img, axis=0)
            it = self.datagen.flow(batch, batch_size=1, shuffle=False)
            img = next(it)[0]

        if self.to_float and img.max() > 1.5:
            img = img / 255.0

        return {"image": img}


IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/unitedmodelv02/Cassava_Best_UnitedModel_V02.hdf5"

train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=[0.5, 1.5],
    horizontal_flip=True,
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    fill_mode="nearest",
)

AUGMENTATIONS_TRAIN = KerasAugmentWrapper(
    datagen=train_datagen,
    do_center_crop=True,
    crop_height=IMG_HEIGHT,
    crop_width=IMG_WIDTH,
    to_float=True,
)

AUGMENTATIONS_TEST = KerasAugmentWrapper(
    datagen=None,
    do_center_crop=False,
    to_float=True,
)



## === cell 7
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 8
test_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=45,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)



## === cell 9
from tensorflow.keras.utils import Sequence
import random

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_HEIGHT, IMG_WIDTH), RESAMPLE)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x = np.asarray(x_set)
        self.y = None if y_set is None else np.asarray(y_set)
        self.batch_size = int(batch_size)
        self.augment = augmentations
        self._n = int(len(self.x))

    def __len__(self):
        return int(np.ceil(self._n / float(self.batch_size)))

    def on_epoch_end(self):
        return

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self._n)
        batch_x = self.x[start:end]

        if self.mode == "TEST":
            batch_y = None
        else:
            batch_y = self.y[start:end]

        img_list = []
        append = img_list.append

        for x in batch_x:
            if self.data_type == "TRAIN_DATA":
                if random.uniform(0, 1) > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x).astype(
                        np.float32, copy=False
                    )
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x).astype(
                        np.float32, copy=False
                    )

            append(img_data)

        img_array = np.stack(img_list, axis=0).astype(np.float32, copy=False)

        if img_array.max() > 1.5:
            img_array = img_array / 255.0

        if batch_y is None:
            return img_array, np.array([])
        return img_array, np.asarray(batch_y)




## === cell 10
test_filenames = [e.name for e in os.scandir(TEST_DIR) if e.is_file()]
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
test_gen = AugmentedImageSequence(
    "TEST",
    "TEST_DATA",
    test_df["image_id"].values,
    None,
    batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 12
from tensorflow.keras.models import load_model



## === cell 13
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

num_classes = 5


def build_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Rescaling(1.0)(inputs)  # no-op; keeps original behavior
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if PRE_TRAINED_MODEL and os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
else:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_gen = AugmentedImageSequence(
        "TRAIN",
        "TRAIN_DATA",
        tr_df["image_id"].values,
        tr_df["label"].values,
        batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        "VALIDATE",
        "VALIDATE_DATA",
        va_df["image_id"].values,
        va_df["label"].values,
        batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=3, verbose=1)


@tf.function(reduce_retracing=True)
def _predict_batch(x):
    return model(x, training=False)


model.summary()




## === cell 14
def old_predict():
    test_results_list = []
    for image_id in test_df["image_id"]:
        image_path = os.path.join(TEST_DIR, image_id)
        image_data = Image.open(image_path).convert("RGB")
        image_data = image_data.resize((IMG_HEIGHT, IMG_WIDTH), RESAMPLE)
        image_data = np.array(image_data).astype(np.float32)
        if image_data.max() > 1.5:
            image_data = image_data / 255.0
        image_data = np.expand_dims(image_data, axis=0)

        predict_class = model.predict(image_data, verbose=0)
        test_results_dict = {
            "image_id": image_id,
            "label": int(np.argmax(predict_class)),
        }
        test_results_list.append(test_results_dict)

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 15
_TTA_DATAGEN = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)

from concurrent.futures import ThreadPoolExecutor

_TEST_IMAGE_CACHE = {}  # image_id -> float32 [0,1] array
_TEST_IMAGE_CACHE_ORDER = []
_TEST_IMAGE_CACHE_MAX = 768  # bounded; enough to amortize I/O without blowing RAM


def _cache_put(image_id: str, arr: np.ndarray):
    if image_id in _TEST_IMAGE_CACHE:
        return
    _TEST_IMAGE_CACHE[image_id] = arr
    _TEST_IMAGE_CACHE_ORDER.append(image_id)
    if len(_TEST_IMAGE_CACHE_ORDER) > _TEST_IMAGE_CACHE_MAX:
        old = _TEST_IMAGE_CACHE_ORDER.pop(0)
        _TEST_IMAGE_CACHE.pop(old, None)


def _load_test_image_as_float01(image_id: str) -> np.ndarray:
    cached = _TEST_IMAGE_CACHE.get(image_id)
    if cached is not None:
        return cached

    image_path = os.path.join(TEST_DIR, image_id)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_HEIGHT, IMG_WIDTH), RESAMPLE)
        arr = np.asarray(im, dtype=np.uint8)

    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.5:
        arr *= 1.0 / 255.0

    _cache_put(image_id, arr)
    return arr


def _preload_test_images(image_ids, max_workers=8):
    image_ids = list(image_ids)
    if not image_ids:
        return

    def _worker(iid):
        _load_test_image_as_float01(iid)
        return iid

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(_worker, image_ids, chunksize=64):
            pass


def predict_with_batched_tta(image_ids, predict_batch_size=64, tta_views=6, seed=12345):
    image_ids = list(image_ids)
    n = len(image_ids)
    results = np.empty((n,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    for start in range(0, n, predict_batch_size):
        end = min(start + predict_batch_size, n)
        bs = end - start

        base_images = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        for i in range(bs):
            base_images[i] = _load_test_image_as_float01(image_ids[start + i])

        transforms = [[None] * (tta_views - 1) for _ in range(bs)]
        for i in range(bs):
            for v in range(tta_views - 1):
                np.random.seed(int(rng.randint(0, 2**31 - 1)))
                transforms[i][v] = _TTA_DATAGEN.get_random_transform(
                    (IMG_HEIGHT, IMG_WIDTH, 3)
                )

        batch = np.empty((bs * tta_views, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        batch[0::tta_views] = base_images  # view 0

        k = 0
        for i in range(bs):
            base = base_images[i]
            off = i * tta_views
            for v in range(1, tta_views):
                batch[off + v] = _TTA_DATAGEN.apply_transform(
                    base, transforms[i][v - 1]
                )
                k += 1

        preds = _predict_batch(batch).numpy()  # (bs*tta_views, num_classes)
        preds = preds.reshape(bs, tta_views, -1).mean(axis=1)
        results[start:end] = np.argmax(preds, axis=1).astype(np.int64)

    return results




## === cell 16
image_ids = test_df["image_id"].values

_preload_test_images(image_ids, max_workers=min(8, (os.cpu_count() or 8)))

pred_labels = predict_with_batched_tta(
    image_ids, predict_batch_size=64, tta_views=6, seed=12345
)

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(image_ids.tolist(), pred_labels.astype(int).tolist()))
sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())



## === cell 17
submission = pd.read_csv("submission.csv")
submission.head(3)

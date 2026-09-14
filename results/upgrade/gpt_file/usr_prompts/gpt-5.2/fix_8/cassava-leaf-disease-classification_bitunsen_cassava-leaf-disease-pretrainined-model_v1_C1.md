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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import json
from PIL import Image

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)



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
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 32

PRE_TRAINED_MODEL_CANDIDATES = [
    "../input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
    "/kaggle/input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in PRE_TRAINED_MODEL_CANDIDATES if os.path.exists(p)), None
)
print("PRE_TRAINED_MODEL:", PRE_TRAINED_MODEL)



## === cell 7
import random
from tensorflow.keras.utils import Sequence

try:
    import cv2  # available in most Kaggle TF images
except Exception:
    cv2 = None


class SimpleAugment:
    def __init__(self, mode="train", img_h=IMG_HEIGHT, img_w=IMG_WIDTH):
        self.mode = mode
        self.img_h = img_h
        self.img_w = img_w

    def __call__(self, image):
        img = image

        if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
            if cv2 is not None:
                img = cv2.resize(
                    img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA
                )
            else:
                img = np.array(
                    Image.fromarray(img.astype(np.uint8)).resize(
                        (self.img_w, self.img_h), Image.BILINEAR
                    )
                )

        if self.mode == "train":
            if random.random() < 0.5:
                img = img[:, ::-1, :]

            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.2, 0.2)  # contrast
                beta = random.uniform(-0.2, 0.2) * 255.0  # brightness in pixel scale
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

            if random.random() < 0.2:
                angle = random.uniform(-15, 15)
                if cv2 is not None:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                    img = cv2.warpAffine(
                        img,
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                else:
                    pil = Image.fromarray(img.astype(np.uint8))
                    img = np.array(pil.rotate(angle, resample=Image.BILINEAR))

        img = img.astype(np.float32) / 255.0
        return {"image": img}


AUGMENTATIONS_TRAIN = SimpleAugment(mode="train", img_h=IMG_HEIGHT, img_w=IMG_WIDTH)
AUGMENTATIONS_TEST = SimpleAugment(mode="test", img_h=IMG_HEIGHT, img_w=IMG_WIDTH)



## === cell 8
from collections import OrderedDict

_IMAGE_CACHE = OrderedDict()
_IMAGE_CACHE_MAX = (
    256  # bounded to avoid memory blow-ups (~256 * 300*300*3 bytes ~= 69MB)
)


def _cache_get(path):
    img = _IMAGE_CACHE.get(path)
    if img is not None:
        _IMAGE_CACHE.move_to_end(path)
    return img


def _cache_put(path, img):
    _IMAGE_CACHE[path] = img
    _IMAGE_CACHE.move_to_end(path)
    if len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)


def _image_path(data_type, image_id):
    if data_type == "TEST_DATA":
        return os.path.join(TEST_DIR, image_id)
    return os.path.join(TRAIN_DIR, image_id)


def _load_resized_uint8(image_path):
    cached = _cache_get(image_path)
    if cached is not None:
        return cached

    if cv2 is not None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        img_data = Image.open(image_path).convert("RGB")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = np.array(img_data.resize((IMG_WIDTH, IMG_HEIGHT), resample))

    _cache_put(image_path, img)
    return img


def load_single_image(data_type, image_id_or_path):
    if isinstance(image_id_or_path, str) and (
        image_id_or_path.startswith(TRAIN_DIR) or image_id_or_path.startswith(TEST_DIR)
    ):
        return _load_resized_uint8(image_id_or_path)
    image_path = _image_path(data_type, image_id_or_path)
    return _load_resized_uint8(image_path)


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type

        if data_set_type == "TEST_DATA":
            base_dir = TEST_DIR
        else:
            base_dir = TRAIN_DIR
        x_list = list(x_set)
        self.x = np.asarray([os.path.join(base_dir, fn) for fn in x_list], dtype=object)

        self.y = None if y_set is None else np.asarray(list(y_set), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.augment = augmentations

        self._is_test = mode == "TEST"
        self._is_validate_data = data_set_type == "VALIDATE_DATA"
        self._augment_obj = augmentations  # local alias
        self._inv255 = np.float32(1.0 / 255.0)

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.x))
        batch_x = self.x[start:end]
        bs = end - start

        if self._is_test:
            tmp = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            for i, p in enumerate(batch_x):
                tmp[i] = load_single_image(self.data_type, p)
            out = tmp.astype(np.float32) * self._inv255
            return out, np.empty((0,), dtype=np.int64)

        augment = self._augment_obj
        out = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

        if self._is_validate_data:
            for i, p in enumerate(batch_x):
                out[i] = augment(image=load_single_image(self.data_type, p))["image"]
            return out, self.y[start:end]

        for i, p in enumerate(batch_x):
            if random.uniform(0, 1) > 0.5:
                out[i] = augment(image=load_single_image(self.data_type, p))["image"]
            else:
                out[i] = (
                    load_single_image(self.data_type, p).astype(np.float32)
                    * self._inv255
                )

        return out, self.y[start:end]




## === cell 9
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
test_samples



## === cell 10
test_gen = AugmentedImageSequence(
    mode="TEST",
    data_set_type="TEST_DATA",
    x_set=test_df["image_id"],
    y_set=None,
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 11
from keras.models import load_model


def build_fallback_model(img_h=IMG_HEIGHT, img_w=IMG_WIDTH, n_classes=5):
    base = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(img_h, img_w, 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)  # inputs are 0..1
    x = base(x, training=False)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

model = None
if PRE_TRAINED_MODEL is not None and os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
else:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_df))
    split = int(0.9 * len(train_df))
    tr_idx = perm[:split]
    va_idx = perm[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_gen = AugmentedImageSequence(
        mode="TRAIN",
        data_set_type="TRAIN_DATA",
        x_set=tr_df["image_id"],
        y_set=tr_df["label"],
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    valid_gen = AugmentedImageSequence(
        mode="VALIDATE",
        data_set_type="VALIDATE_DATA",
        x_set=va_df["image_id"],
        y_set=va_df["label"],
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    model = build_fallback_model()
    model.fit(train_gen, validation_data=valid_gen, epochs=1, verbose=1)

model.summary()



## === cell 12
workers = min(8, (os.cpu_count() or 2))
predict = model.predict(
    test_gen,
    steps=int(np.ceil(test_samples / batch_size)),
    verbose=1,
    workers=workers,
    use_multiprocessing=True,
    max_queue_size=64,
)
predict.shape



## === cell 13
test_pred_labels = np.argmax(predict, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_pred_labels}
)
submission.to_csv("submission.csv", index=False)
submission.head(3)



## === cell 14
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
check.head(3)

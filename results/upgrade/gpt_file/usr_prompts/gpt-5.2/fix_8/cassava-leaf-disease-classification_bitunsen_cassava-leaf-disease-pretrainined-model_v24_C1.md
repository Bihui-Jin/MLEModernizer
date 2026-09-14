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



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")



## === cell 2
with open(MAP_JSON) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 3
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 4
input_files = os.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv10/Cassava_Best_Xception_Model_V10.hdf5"



## === cell 6
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightness,
        RandomContrast,
        CenterCrop,
        ShiftScaleRotate,
        ToFloat,
    )

    HAS_ALBUMENTATIONS = True
except Exception as e:
    HAS_ALBUMENTATIONS = False
    print(
        "Albumentations not available/compatible, continuing without it. Error:",
        repr(e),
    )



## === cell 7
if HAS_ALBUMENTATIONS:
    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
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
else:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 9
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 10
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
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

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 11
from tensorflow.keras.utils import Sequence
from PIL import Image

_PIL_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), resample=_PIL_RESAMPLE)
        img_data = np.asarray(img)
    return img_data


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
            img_data = load_single_image(self.data_type, x)

            if self.augment is not None:
                if self.data_type == "TRAIN_DATA":
                    if random.uniform(0, 1) > 0.5:
                        img_data = self.augment(image=img_data)["image"]
                elif self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=img_data)["image"]

            img_list.append(img_data)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## === cell 12
test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 13
test_gen = None



## === cell 14
from tensorflow.keras import layers

num_classes = 5


def build_model():
    base = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    x = base.output
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=outputs)
    return model




## === cell 15
train_df = pd.read_csv(TRAIN_CSV)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_DIR, x)
)
train_df["label"] = train_df["label"].astype(str)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = valid_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)



## === cell 16
if os.path.exists(PRE_TRAINED_MODEL):
    model = keras.models.load_model(PRE_TRAINED_MODEL)
    print("Loaded pre-trained model:", PRE_TRAINED_MODEL)
else:
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.get_layer(index=1).trainable = False
    model.fit(train_gen, epochs=2, validation_data=val_gen, verbose=1)

    base_model = model.get_layer(index=1)
    base_model.trainable = True
    for l in base_model.layers[:-30]:
        l.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_gen, epochs=3, validation_data=val_gen, verbose=1)




## === cell 17
def old_predict():
    test_results_list = []
    for image_id in test_df["image_id"]:
        image_path = os.path.join(TEST_DIR, image_id)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((IMG_HEIGHT, IMG_WIDTH), resample=_PIL_RESAMPLE)
            image_data = np.asarray(image, dtype=np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        predict_class = model.predict(image_data, verbose=0)
        test_results_list.append(
            {"image_id": image_id, "label": int(np.argmax(predict_class, axis=1)[0])}
        )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 18
_TTA_DATAGEN = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)


def get_augmented_images(image_id):
    image_path = os.path.join(TEST_DIR, image_id)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((IMG_HEIGHT, IMG_WIDTH), resample=_PIL_RESAMPLE)
        image_data = np.asarray(image, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)

    it = _TTA_DATAGEN.flow(image_data, batch_size=1, shuffle=False)

    image_list = []
    image_list.append(image_data / 255.0)
    for _ in range(5):
        batch = next(it)
        image_list.append(batch)
    return image_list




## === cell 19
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_ids = sample_sub["image_id"].tolist()

AUTOTUNE = tf.data.AUTOTUNE
base_seed = tf.constant([42, 12345], dtype=tf.int32)

TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")
tfrec_files = []
if os.path.isdir(TEST_TFREC_DIR):
    tfrec_files = sorted(
        [
            os.path.join(TEST_TFREC_DIR, f)
            for f in os.listdir(TEST_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )


def _decode_resize_from_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="bicubic", antialias=True
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _parse_tfrecord_image_only(example):
    feat = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feat)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    image_id = tf.strings.join([ex["image_name"], tf.constant(".jpg")])
    return img, image_id


@tf.function(reduce_retracing=True)
def _apply_tta_vec(img, seed_pairs_5x2):
    img5 = tf.repeat(tf.expand_dims(img, 0), repeats=5, axis=0)

    s0 = seed_pairs_5x2[:, 0]
    s1 = seed_pairs_5x2[:, 1]

    r = tf.random.stateless_uniform([5], seed=tf.stack([s0, s1], axis=1))
    img5 = tf.where(r[:, None, None, None] < 0.5, img5, tf.image.flip_left_right(img5))

    r = tf.random.stateless_uniform([5], seed=tf.stack([s0 + 1, s1 + 1], axis=1))
    img5 = tf.where(r[:, None, None, None] < 0.5, img5, tf.image.flip_up_down(img5))

    tx = tf.random.stateless_uniform(
        [5], seed=tf.stack([s0 + 2, s1 + 2], axis=1), minval=-0.1, maxval=0.1
    ) * float(IMG_WIDTH)
    ty = tf.random.stateless_uniform(
        [5], seed=tf.stack([s0 + 3, s1 + 3], axis=1), minval=-0.1, maxval=0.1
    ) * float(IMG_HEIGHT)
    zx = tf.random.stateless_uniform(
        [5], seed=tf.stack([s0 + 4, s1 + 4], axis=1), minval=0.6, maxval=1.4
    )
    zy = tf.random.stateless_uniform(
        [5], seed=tf.stack([s0 + 5, s1 + 5], axis=1), minval=0.6, maxval=1.4
    )
    ang = tf.random.stateless_uniform(
        [5],
        seed=tf.stack([s0 + 6, s1 + 6], axis=1),
        minval=-45.0,
        maxval=45.0,
    ) * (np.pi / 180.0)
    sh = tf.random.stateless_uniform(
        [5], seed=tf.stack([s0 + 7, s1 + 7], axis=1), minval=-0.1, maxval=0.1
    )

    cos_a = tf.cos(ang)
    sin_a = tf.sin(ang)

    a0 = (cos_a + sh * sin_a) / zx
    a1 = (-sin_a + sh * cos_a) / zx
    b0 = (sin_a) / zy
    b1 = (cos_a) / zy

    cx = (IMG_WIDTH - 1.0) / 2.0
    cy = (IMG_HEIGHT - 1.0) / 2.0

    t0 = cx - a0 * cx - a1 * cy - tx
    t1 = cy - b0 * cx - b1 * cy - ty

    transforms = tf.stack(
        [a0, a1, t0, b0, b1, t1, tf.zeros([5]), tf.zeros([5])], axis=1
    )

    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img5,
        transforms=transforms,
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out


@tf.function(reduce_retracing=True)
def _make_6_variants_from_img(img, idx):
    k = tf.range(1, 6, dtype=tf.int32)  # 1..5
    seeds0 = idx * 31 + k * 997
    seeds1 = idx * 131 + k * 541
    seed_pairs = base_seed + tf.stack([seeds0, seeds1], axis=1)  # (5,2)
    tta_imgs = _apply_tta_vec(img, seed_pairs)  # (5,H,W,3)
    return tf.concat([tf.expand_dims(img, 0), tta_imgs], axis=0)  # (6,H,W,3)


options = tf.data.Options()
options.deterministic = True

if tfrec_files:
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds = ds.map(
        _parse_tfrecord_image_only, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    n = len(sample_ids)
    ds = ds.take(n)

    cache_path = "/kaggle/working/_decoded_resized_cache_tfrecord"
    try:
        ds = ds.cache(cache_path)
    except Exception:
        ds = ds.cache()

    ds = ds.enumerate()  # (idx, (img, image_id))
    ds = ds.map(
        lambda i, x: (x[0], i, x[1]), num_parallel_calls=AUTOTUNE, deterministic=True
    )
else:
    paths = [os.path.join(TEST_DIR, iid) for iid in sample_ids]
    paths_tf = tf.constant(paths)
    ids_tf = tf.constant(sample_ids)

    def _decode_resize(path):
        img_bytes = tf.io.read_file(path)
        return _decode_resize_from_jpeg_bytes(img_bytes)

    ds = tf.data.Dataset.from_tensor_slices((paths_tf, ids_tf)).with_options(options)
    ds = ds.map(
        lambda p, iid: (_decode_resize(p), iid),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    cache_path = "/kaggle/working/_decoded_resized_cache"
    try:
        ds = ds.cache(cache_path)
    except Exception:
        ds = ds.cache()

    ds = ds.enumerate()  # (idx, (img, image_id))
    ds = ds.map(
        lambda i, x: (x[0], i, x[1]), num_parallel_calls=AUTOTUNE, deterministic=True
    )

predict_bs = 32  # unchanged

ds = ds.batch(predict_bs, drop_remainder=False).prefetch(AUTOTUNE)


@tf.function(reduce_retracing=True)
def _make_6_variants_batch(imgs, idxs, image_ids):
    def _one(args):
        img, idx = args
        return _make_6_variants_from_img(img, idx)  # (6,H,W,3)

    batch6 = tf.map_fn(
        _one, (imgs, idxs), fn_output_signature=tf.float32, parallel_iterations=16
    )
    shp = tf.shape(batch6)
    b = shp[0]
    flat = tf.reshape(batch6, [b * 6, IMG_HEIGHT, IMG_WIDTH, 3])
    return flat, image_ids


ds_flat_with_ids = ds.map(
    _make_6_variants_batch, num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)

pred_list = []
id_list = []
for flat_imgs, batch_ids in ds_flat_with_ids:
    preds = model.predict(flat_imgs, verbose=0)
    pred_list.append(preds)
    id_list.append(batch_ids.numpy().astype(str))

preds_all = np.concatenate(pred_list, axis=0)
ids_all = np.concatenate(id_list, axis=0)

n = len(sample_ids)
preds_all = preds_all[: n * 6].reshape(n, 6, num_classes)
mean_preds = preds_all.mean(axis=1)
labels = mean_preds.argmax(axis=1).astype(int)

submission = pd.DataFrame({"image_id": ids_all, "label": labels})
submission.to_csv("submission.csv", index=False)
print(submission.head())



## === cell 20
submission_check = pd.read_csv("submission.csv")
print(submission_check.shape)
print(submission_check.columns.tolist())
print(submission_check.head(3))

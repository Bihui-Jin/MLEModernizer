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

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/unitedmodelv02/Cassava_Best_UnitedModel_V02.hdf5"



## === cell 2
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 4
train_csv_path = os.path.join(BASE_DIR, "train.csv")
_train_df_peek = pd.read_csv(train_csv_path, usecols=["image_id"])
print(f"Number of train images (from train.csv): {len(_train_df_peek)}")




## === cell 5
class KerasAugmentWrapper:
    """
    OPTIMIZATION (correctness-preserving):
    - Use ImageDataGenerator's random_transform/standardize directly instead of building a flow()
      iterator per image, eliminating huge per-sample overhead while keeping identical transform logic.
    """

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
            params = self.datagen.get_random_transform(img.shape)
            img = self.datagen.apply_transform(img, params)
            img = self.datagen.standardize(img)

        if self.to_float and img.max() > 1.5:
            img = img / 255.0

        return {"image": img}


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



## === cell 6
from tensorflow.keras.utils import Sequence


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




## === cell 7
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
test_df = pd.read_csv(sample_path, usecols=["image_id"])
test_samples = test_df.shape[0]
test_samples



## === cell 8
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
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

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


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec([None, IMG_HEIGHT, IMG_WIDTH, 3], tf.float32)],
    jit_compile=True,
)
def _predict_batch(x):
    return model(x, training=False)


model.summary()




## === cell 9
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




## === cell 10
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


@tf.function(reduce_retracing=True)
def _tf_decode_resize_to_float01(image_id: tf.Tensor) -> tf.Tensor:
    path = tf.strings.join([TEST_DIR, image_id])
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3", antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


@tf.function(reduce_retracing=True)
def _tta_tf_equivalent(img: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    seed = tf.cast(seed, tf.int32)

    r0 = tf.random.stateless_uniform([], seed=[seed, 1], minval=0.0, maxval=1.0)
    r1 = tf.random.stateless_uniform([], seed=[seed, 2], minval=0.0, maxval=1.0)
    do_h = tf.less(r0, 0.5)
    do_v = tf.less(r1, 0.5)
    img = tf.cond(do_h, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(do_v, lambda: tf.image.flip_up_down(img), lambda: img)

    angle = tf.random.stateless_uniform(
        [], seed=[seed, 3], minval=-45.0, maxval=45.0
    ) * (np.pi / 180.0)

    zx = tf.random.stateless_uniform([], seed=[seed, 4], minval=0.5, maxval=1.5)
    zy = tf.random.stateless_uniform([], seed=[seed, 5], minval=0.5, maxval=1.5)

    shear = tf.random.stateless_uniform([], seed=[seed, 6], minval=-0.1, maxval=0.1)

    tx = tf.random.stateless_uniform(
        [], seed=[seed, 7], minval=-0.1, maxval=0.1
    ) * tf.cast(IMG_HEIGHT, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=[seed, 8], minval=-0.1, maxval=0.1
    ) * tf.cast(IMG_WIDTH, tf.float32)

    cx = (tf.cast(IMG_WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_HEIGHT, tf.float32) - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    r = tf.stack([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], axis=0)

    sh = tf.stack(
        [[1.0, -tf.sin(shear), 0.0], [0.0, tf.cos(shear), 0.0], [0.0, 0.0, 1.0]], axis=0
    )

    sc = tf.stack([[1.0 / zx, 0.0, 0.0], [0.0, 1.0 / zy, 0.0], [0.0, 0.0, 1.0]], axis=0)

    t_center = tf.stack([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], axis=0)
    t_uncenter = tf.stack([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], axis=0)

    t_shift = tf.stack([[1.0, 0.0, ty], [0.0, 1.0, tx], [0.0, 0.0, 1.0]], axis=0)

    m = tf.linalg.matmul(
        t_uncenter,
        tf.linalg.matmul(
            t_shift,
            tf.linalg.matmul(tf.linalg.matmul(r, tf.linalg.matmul(sh, sc)), t_center),
        ),
    )

    a0, a1, a2 = m[0, 0], m[0, 1], m[0, 2]
    b0, b1, b2 = m[1, 0], m[1, 1], m[1, 2]
    c0, c1 = m[2, 0], m[2, 1]
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[None, :]

    img_b = img[None, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_b,
        transforms=transform,
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
        fill_value=0.0,
        interpolation="NEAREST",
        fill_mode="NEAREST",
    )[0]
    out.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return out


def _make_test_tta_views_dataset(image_ids, tta_views, base_seed=12345):
    image_ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    n = tf.shape(image_ids)[0]

    ds = tf.data.Dataset.range(n)

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    def _per_index(i):
        image_id = image_ids[i]
        base = _tf_decode_resize_to_float01(image_id)

        base_ds = tf.data.Dataset.from_tensors(base).cache()

        def _make_view(v, base_img):
            v = tf.cast(v, tf.int32)
            seed = (
                tf.cast(base_seed, tf.int32)
                + tf.cast(i, tf.int32) * tf.cast(tta_views, tf.int32)
                + v
            )
            img = tf.cond(
                tf.equal(v, 0),
                lambda: base_img,
                lambda: _tta_tf_equivalent(base_img, seed),
            )
            return tf.cast(i, tf.int32), v, img

        vds = tf.data.Dataset.range(tta_views)
        vds = vds.map(lambda v: (tf.cast(v, tf.int32)), num_parallel_calls=AUTOTUNE)

        base_rep = base_ds.repeat(tta_views)
        vds = tf.data.Dataset.zip((vds, base_rep)).map(
            lambda v, b: _make_view(v, b), num_parallel_calls=AUTOTUNE
        )
        return vds

    ds = ds.flat_map(_per_index)
    return ds


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([None, IMG_HEIGHT, IMG_WIDTH, 3], tf.float32),
    ],
    jit_compile=True,
)
def _predict_views_batch(views_4d: tf.Tensor) -> tf.Tensor:
    return _predict_batch(views_4d)


def predict_with_batched_tta_from_ds(
    image_ids, predict_batch_size=32, tta_views=6, seed=12345
):
    image_ids = np.asarray(image_ids, dtype=object)
    n = int(len(image_ids))

    ds = _make_test_tta_views_dataset(
        image_ids,
        tta_views=tta_views,
        base_seed=seed,
    )

    ds = ds.batch(predict_batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    sums = tf.zeros([n, num_classes], dtype=tf.float32)
    counts = tf.zeros([n], dtype=tf.float32)

    for i_batch, v_batch, x_batch in ds:
        preds = _predict_views_batch(tf.cast(x_batch, tf.float32))  # [B, C]
        i_batch = tf.cast(i_batch, tf.int32)

        batch_sums = tf.math.unsorted_segment_sum(preds, i_batch, n)  # [n, C]
        batch_counts = tf.math.unsorted_segment_sum(
            tf.ones([tf.shape(i_batch)[0]], dtype=tf.float32), i_batch, n
        )  # [n]

        sums = sums + batch_sums
        counts = counts + batch_counts

    means = sums / counts[:, None]
    out = tf.argmax(means, axis=1, output_type=tf.int64)
    return out.numpy()




## === cell 11
image_ids = test_df["image_id"].values

pred_labels = predict_with_batched_tta_from_ds(
    image_ids, predict_batch_size=64, tta_views=6, seed=12345
)

sample_sub = pd.read_csv(
    os.path.join(BASE_DIR, "sample_submission.csv"), usecols=["image_id", "label"]
)
pred_map = dict(zip(image_ids.tolist(), pred_labels.astype(int).tolist()))
sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())



## === cell 12
submission = pd.read_csv("submission.csv")
submission.head(3)

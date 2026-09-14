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

0.8862194016319129

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
train_csv_path = os.path.join(BASE_DIR, "train.csv")
_train_df_tmp = pd.read_csv(train_csv_path, usecols=["image_id"])
print(f"Number of train images (from train.csv): {len(_train_df_tmp)}")
del _train_df_tmp



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = "../input/resnet50v01/Cassava_Best_ResNet50_Model_V01.hdf5"




## === cell 7
class _NoOpAugment:
    def __call__(self, image):
        return {"image": image}


AUGMENTATIONS_TRAIN = _NoOpAugment()
AUGMENTATIONS_TEST = _NoOpAugment()



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img_data = Image.open(image_path).convert("RGB")
    img_data = np.array(img_data.resize((IMG_HEIGHT, IMG_WIDTH), _RESAMPLE))
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




## === cell 10
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_df = pd.read_csv(sample_sub_path)
test_df = sample_df[["image_id"]].copy()

test_samples = test_df.shape[0]
print("Test samples:", test_samples)



## === cell 11
import glob


def resolve_model_path(preferred_path: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path

    target_name = os.path.basename(preferred_path)
    matches = glob.glob(f"/kaggle/input/**/{target_name}", recursive=True)
    return matches[0] if matches else preferred_path


resolved_model_path = resolve_model_path(PRE_TRAINED_MODEL)
print("Resolved model path:", resolved_model_path)
print("Exists:", os.path.exists(resolved_model_path))



## === cell 12
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_DIR, x).replace("\\", "/")
)

print("Train rows:", len(train_df))
print(train_df.head(2))

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

print("Train/Val:", len(trn_df), len(val_df))



## === cell 13
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_flow = train_datagen.flow_from_dataframe(
    trn_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="sparse",
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)

val_flow = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="sparse",
    batch_size=batch_size,
    shuffle=False,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2956401227.py in <cell line: 0>()
     11 val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
     12 
---> 13 train_flow = train_datagen.flow_from_dataframe(
     14     trn_df,
     15     x_col="image_path",

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_dataframe(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
   1206             )
   1207 
-> 1208         return DataFrameIterator(
   1209             dataframe,
   1210             directory,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)
    749         self.dtype = dtype
    750         # check that inputs match the required class_mode
--> 751         self._check_params(df, x_col, y_col, weight_col, classes)
    752         if (
    753             validate_filenames

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in _check_params(self, df, x_col, y_col, weight_col, classes)
    817         if self.class_mode in {"binary", "sparse"}:
    818             if not all(df[y_col].apply(lambda x: isinstance(x, str))):
--> 819                 raise TypeError(
    820                     'If class_mode="{}", y_col="{}" column '
    821                     "values must be strings.".format(self.class_mode, y_col)

TypeError: If class_mode="sparse", y_col="label" column values must be strings.

## === cell 14
NUM_CLASSES = 5

if os.path.exists(resolved_model_path):
    model = load_model(resolved_model_path, compile=False)
    print("Loaded pretrained model.")
else:
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
    val_steps = int(np.ceil(len(val_df) / batch_size))
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=3,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
        workers=os.cpu_count() or 1,
        use_multiprocessing=True,
        max_queue_size=32,
    )

model.summary()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3080552351.py in <cell line: 0>()
     28     val_steps = int(np.ceil(len(val_df) / batch_size))
     29     model.fit(
---> 30         train_flow,
     31         validation_data=val_flow,
     32         epochs=3,

NameError: name 'train_flow' is not defined

## === cell 15
TEST_TTA_N = 3  # keep identical to original loop range(3)

_test_datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)

_NEED_RESCALE = not os.path.exists(resolved_model_path)

AUTOTUNE = tf.data.AUTOTUNE

_tta_aug_layer = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=42),
        keras.layers.RandomZoom(
            height_factor=(-0.4, 0.4),
            width_factor=(-0.4, 0.4),
            fill_mode="nearest",
            seed=42,
        ),
        keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=42
        ),
    ],
    name="tta_augment_fast",
)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3", antialias=True
    )
    img = tf.cast(img, tf.float32)
    if _NEED_RESCALE:
        img = img / 255.0
    return img


try:
    from tensorflow.keras.layers import RandomShear

    _HAS_RANDOM_SHEAR = True
except Exception:
    _HAS_RANDOM_SHEAR = False

if _HAS_RANDOM_SHEAR:
    _shear_layer = RandomShear(
        x_factor=(-0.1, 0.1), y_factor=0.0, fill_mode="nearest", seed=42
    )
else:
    _shear_layer = None


@tf.function(reduce_retracing=True)
def _apply_full_tta_batch(x_batch: tf.Tensor) -> tf.Tensor:
    views = [x_batch]
    for _ in range(TEST_TTA_N):
        x_aug = _tta_aug_layer(x_batch, training=True)
        if _shear_layer is not None:
            x_aug = _shear_layer(x_aug, training=True)
        views.append(x_aug)
    return tf.concat(views, axis=0)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2597893392.py in <cell line: 0>()
     59 
     60 if _HAS_RANDOM_SHEAR:
---> 61     _shear_layer = RandomShear(
     62         x_factor=(-0.1, 0.1), y_factor=0.0, fill_mode="nearest", seed=42
     63     )

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py in __init__(self, x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, **kwargs)
     82     ):
     83         super().__init__(data_format=data_format, **kwargs)
---> 84         self.x_factor = self._set_factor_with_name(x_factor, "x_factor")
     85         self.y_factor = self._set_factor_with_name(y_factor, "y_factor")
     86 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py in _set_factor_with_name(self, factor, factor_name)
    110                     + f"Received: {factor_name}={factor}"
    111                 )
--> 112             self._check_factor_range(factor[0])
    113             self._check_factor_range(factor[1])
    114             lower, upper = sorted(factor)

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py in _check_factor_range(self, input_number)
    126     def _check_factor_range(self, input_number):
    127         if input_number > 1.0 or input_number < 0.0:
--> 128             raise ValueError(
    129                 self._FACTOR_VALIDATION_ERROR
    130                 + f"Received: input_number={input_number}"

ValueError: The `factor` argument should be a number (or a list of two numbers) in the range [0, 1.0]. Received: input_number=-0.1

## === cell 16
test_image_ids = test_df["image_id"].values
n_test = len(test_image_ids)

PRED_BATCH = 32
pred_labels = np.empty((n_test,), dtype=np.int64)

paths = np.array(
    [os.path.join(TEST_DIR, img_id) for img_id in test_image_ids], dtype=object
)
idxs = np.arange(n_test, dtype=np.int32)

ds = tf.data.Dataset.from_tensor_slices((idxs, paths))
ds = ds.map(
    lambda i, p: (i, _decode_resize(p)),
    num_parallel_calls=AUTOTUNE,
)
ds = ds.cache()
ds = ds.batch(PRED_BATCH, drop_remainder=False).prefetch(AUTOTUNE)

for id_batch, x_batch in ds:
    x_tta = _apply_full_tta_batch(x_batch)  # (B*(1+TEST_TTA_N),H,W,3)
    preds_tta = model.predict(x_tta, verbose=0, batch_size=batch_size)

    b = int(id_batch.shape[0])
    preds_tta = preds_tta.reshape((b, 1 + TEST_TTA_N, NUM_CLASSES))
    probs = preds_tta.mean(axis=1)
    labels = np.argmax(probs, axis=1).astype(np.int64)

    pred_labels[id_batch.numpy()] = labels

test_results_df = pd.DataFrame(
    {"image_id": test_image_ids, "label": pred_labels.astype(int)}
)

submission = sample_df[["image_id"]].merge(test_results_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)
submission = submission[["image_id", "label"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head(3))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2121835079.py in <cell line: 0>()
     21 
     22 for id_batch, x_batch in ds:
---> 23     x_tta = _apply_full_tta_batch(x_batch)  # (B*(1+TEST_TTA_N),H,W,3)
     24     preds_tta = model.predict(x_tta, verbose=0, batch_size=batch_size)
     25 

NameError: name '_apply_full_tta_batch' is not defined

## === cell 17
submission_check = pd.read_csv("submission.csv")
print(submission_check.head(3))
print("Columns:", list(submission_check.columns))
print("Rows:", len(submission_check))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4263370173.py in <cell line: 0>()
----> 1 submission_check = pd.read_csv("submission.csv")
      2 print(submission_check.head(3))
      3 print("Columns:", list(submission_check.columns))
      4 print("Rows:", len(submission_check))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

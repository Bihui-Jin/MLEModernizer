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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf
from tensorflow import keras

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
batch_size = 16

_PRETRAINED_CANDIDATES = [
    "../input/unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5",
    "/kaggle/input/unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in _PRETRAINED_CANDIDATES if os.path.exists(p)), _PRETRAINED_CANDIDATES[0]
)

if hasattr(Image, "Resampling"):
    PIL_RESAMPLE = Image.Resampling.LANCZOS
else:
    PIL_RESAMPLE = Image.LANCZOS



## === cell 7
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
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
    print("Albumentations available: using AUGMENTATIONS_TEST/TRAIN.")
except Exception as e:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None
    print(
        f"Albumentations not usable ({e}); proceeding without albumentations augmentations."
    )



## === cell 8
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img_data = Image.open(image_path).convert("RGB")
    img_data = np.array(img_data.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE))
    return img_data


class AugmentedImageSequence(Sequence):
    def __init__(
        self, mode, data_set_type, x_set, y_set, batch_size, augmentations, seed=42
    ):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations
        self._seed = int(seed)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        rng = random.Random(self._seed + int(idx))

        img_list = []
        for x in batch_x:
            img = load_single_image(self.data_type, x)

            if self.data_type == "TRAIN_DATA":
                random_num = rng.uniform(0, 1)
                if random_num > 0.5 and self.augment is not None:
                    img = self.augment(image=img)["image"]
            else:
                if self.data_type == "VALIDATE_DATA" and self.augment is not None:
                    img = self.augment(image=img)["image"]

            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0

            img_list.append(img)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## === cell 10
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
test_df = pd.read_csv(sample_sub_path)[["image_id"]]
test_samples = test_df.shape[0]
test_samples



## === cell 11
test_gen = None
print(
    "Skipped building slow Python Sequence test_gen; using batched TTA prediction instead."
)



## === cell 12
from tensorflow.keras.models import load_model

model = None
if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    print(f"Loaded pretrained model from: {PRE_TRAINED_MODEL}")
else:
    print(
        f"WARNING: Pretrained model not found at {PRE_TRAINED_MODEL}. Using EfficientNetB0(ImageNet) fallback."
    )

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = keras.layers.Dropout(0.2)(base.output)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()



## === cell 13
TTA_N_AUG = 5  # keep exactly the same number of generated augmentations as original

_TTA_DATAGEN = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)


def _load_test_batch(image_ids):
    bsz = len(image_ids)
    out = np.empty((bsz, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)  # in [0,255]
    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(TEST_DIR, image_id)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE)
            out[i] = np.asarray(im, dtype=np.float32)
    return out


def _predict_np_batch(model, np_batch, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(np_batch).batch(
        batch_size, drop_remainder=False
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return model.predict(ds, verbose=0)


def _collect_n_from_flow(flow_iter, n_needed):
    out = []
    got = 0
    while got < n_needed:
        batch = next(flow_iter)
        out.append(batch)
        got += batch.shape[0]
    out = np.concatenate(out, axis=0)
    return out[:n_needed]


def predict_with_tta_batched(
    model, image_ids, tta_n_aug=5, predict_batch_size=128, seed=42
):
    n = len(image_ids)
    n_views = 1 + tta_n_aug  # original image + tta_n_aug generated
    preds_accum = np.zeros((n, 5), dtype=np.float32)
    inv255 = np.float32(1.0 / 255.0)

    internal_pred_bs = 256

    for start in range(0, n, predict_batch_size):
        end = min(n, start + predict_batch_size)
        chunk_ids = image_ids[start:end]
        bsz = end - start

        base_batch = _load_test_batch(chunk_ids)  # (bsz,H,W,3) in [0,255]

        views = np.empty((bsz * n_views, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        views[:bsz] = base_batch

        if tta_n_aug > 0:
            total_aug = bsz * tta_n_aug

            aug_iter = _TTA_DATAGEN.flow(
                base_batch,
                batch_size=bsz,
                shuffle=False,
                seed=int(seed) + int(start),
            )
            aug_all = _collect_n_from_flow(aug_iter, total_aug)  # (total_aug,H,W,3)

            aug_all = aug_all.reshape(
                (tta_n_aug, bsz, IMG_HEIGHT, IMG_WIDTH, 3)
            ).transpose(1, 0, 2, 3, 4)
            views[bsz:] = aug_all.reshape((total_aug, IMG_HEIGHT, IMG_WIDTH, 3))

        views *= inv255

        preds = _predict_np_batch(
            model, views, batch_size=internal_pred_bs
        )  # (bsz*n_views, 5)
        preds = preds.reshape((bsz, n_views, 5))
        preds_accum[start:end] = preds.mean(axis=1)

    return preds_accum


image_ids = test_df["image_id"].values
mean_preds = predict_with_tta_batched(
    model,
    image_ids,
    tta_n_aug=TTA_N_AUG,
    predict_batch_size=256,  # Speed: fewer outer-loop iterations; same TTA semantics.
    seed=42,
)

labels = np.argmax(mean_preds, axis=1).astype(int)
test_results_df = pd.DataFrame({"image_id": image_ids, "label": labels})

submission = pd.read_csv(sample_sub_path)[["image_id"]].merge(
    test_results_df, on="image_id", how="left"
)
assert submission["label"].isna().sum() == 0, "Some test images missing predictions."
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head(5)



## === cell 14
check = pd.read_csv("submission.csv")
print(check.columns.tolist(), check.shape)
print(check.head(3))

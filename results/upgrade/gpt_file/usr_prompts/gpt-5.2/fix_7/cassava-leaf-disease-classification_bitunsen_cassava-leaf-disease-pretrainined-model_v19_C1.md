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

0.8834995466908432

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'The timeout is dominated by Python-side per-image TTA generation (`ImageDataGenerator.flow` + `next` in nested loops) and repeated PIL decode/resize for each augmentation view. I keep the exact same model and TTA semantics (original + 5 augmented views, same `ImageDataGenerator` params), but reduce overhead by (1) decoding/resizing each test image once into a contiguous float32 array cache, (2) generating all TTA views for a chunk using a single `flow` call per image (instead of creating/advancing an iterator per augmentation), and (3) preallocating arrays and minimizing Python loop work. These changes are provably equivalent in terms of what gets predicted (same generator, same number of views, same averaging), with only negligible floating-point ordering differences.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
            img = load_single_image(self.data_type, x)

            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
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
test_gen = AugmentedImageSequence(
    "TEST",
    "TEST_DATA",
    test_df["image_id"].values,
    None,
    batch_size,
    augmentations=AUGMENTATIONS_TEST,
)

len(test_gen)



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

_base_test_cache = {}


def _get_base_test_image(image_id):
    arr = _base_test_cache.get(image_id)
    if arr is None:
        image_path = os.path.join(TEST_DIR, image_id)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE)
            arr = np.asarray(im, dtype=np.float32)  # (H,W,3) in [0,255]
        arr = np.ascontiguousarray(arr)
        _base_test_cache[image_id] = arr
    return arr


def _predict_np_batch(model, np_batch, batch_size):
    ds = (
        tf.data.Dataset.from_tensor_slices(np_batch)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return model.predict(ds, verbose=0)


def predict_with_tta_batched(
    model, image_ids, tta_n_aug=5, predict_batch_size=64, seed=42
):
    n = len(image_ids)
    n_views = 1 + tta_n_aug  # original image + tta_n_aug generated
    preds_accum = np.zeros((n, 5), dtype=np.float32)
    inv255 = np.float32(1.0 / 255.0)

    for start in range(0, n, predict_batch_size):
        end = min(n, start + predict_batch_size)
        chunk_ids = image_ids[start:end]
        bsz = end - start

        base_batch = np.empty((bsz, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        for i, image_id in enumerate(chunk_ids):
            base_batch[i] = _get_base_test_image(image_id)

        views = np.empty((bsz * n_views, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        views[:bsz] = base_batch

        if tta_n_aug > 0:
            total_aug = bsz * tta_n_aug
            aug_iter = _TTA_DATAGEN.flow(
                base_batch,
                batch_size=total_aug,
                shuffle=False,
                seed=seed + start,  # deterministic per chunk
            )
            aug_all = next(aug_iter)  # (total_aug, H, W, 3)
            aug_all = aug_all.reshape(
                (tta_n_aug, bsz, IMG_HEIGHT, IMG_WIDTH, 3)
            ).transpose(1, 0, 2, 3, 4)
            views[bsz:] = aug_all.reshape((total_aug, IMG_HEIGHT, IMG_WIDTH, 3))

        views *= inv255  # normalize to [0,1] like original

        preds = _predict_np_batch(
            model, views, batch_size=256
        )  # internal predict batching for throughput
        preds = preds.reshape((bsz, n_views, 5))
        preds_accum[start:end] = preds.mean(axis=1)

    return preds_accum




## === cell 14
image_ids = test_df["image_id"].values
mean_preds = predict_with_tta_batched(
    model,
    image_ids,
    tta_n_aug=TTA_N_AUG,
    predict_batch_size=64,  # constant-factor speedup; does not change algorithm
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/718510662.py in <cell line: 0>()
      1 image_ids = test_df["image_id"].values
----> 2 mean_preds = predict_with_tta_batched(
      3     model,
      4     image_ids,
      5     tta_n_aug=TTA_N_AUG,

/tmp/ipykernel_11/3076041256.py in predict_with_tta_batched(model, image_ids, tta_n_aug, predict_batch_size, seed)
     77             )
     78             aug_all = next(aug_iter)  # (total_aug, H, W, 3)
---> 79             aug_all = aug_all.reshape(
     80                 (tta_n_aug, bsz, IMG_HEIGHT, IMG_WIDTH, 3)
     81             ).transpose(1, 0, 2, 3, 4)

ValueError: cannot reshape array of size 17280000 into shape (5,64,300,300,3)

## === cell 15
check = pd.read_csv("submission.csv")
print(check.columns.tolist(), check.shape)
print(check.head(3))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3661227437.py in <cell line: 0>()
----> 1 check = pd.read_csv("submission.csv")
      2 print(check.columns.tolist(), check.shape)
      3 print(check.head(3))

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

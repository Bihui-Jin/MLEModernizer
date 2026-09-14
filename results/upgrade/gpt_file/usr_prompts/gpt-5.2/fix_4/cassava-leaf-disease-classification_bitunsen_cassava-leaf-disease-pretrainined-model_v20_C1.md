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

0.8818374131157449

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
if False:
    input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
    print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16
PRE_TRAINED_MODEL = "../input/unionmodelv06/Cassava_Best_UnitedModel_V06.hdf5"

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS



## === cell 7
AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr




## === cell 10
from tensorflow.keras.utils import Sequence
import random


class AugmentedImageSequence(Sequence):
    def __init__(
        self, mode, data_set_type, x_set, y_set, batch_size, augmentations=None
    ):
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
            img_list.append(img_data)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## === cell 11
test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 12
test_gen = AugmentedImageSequence(
    mode="TEST",
    data_set_type="TEST_DATA",
    x_set=test_df["image_id"].values,
    y_set=None,
    batch_size=batch_size,
    augmentations=None,
)



## === cell 13
from tensorflow.keras.models import load_model

model = None
if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model:", PRE_TRAINED_MODEL)
else:
    print("Pretrained model not found at:", PRE_TRAINED_MODEL)
    print(
        "Training a small fallback model from train_images/train.csv to generate submission."
    )



## === cell 14
if model is None:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.1,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)

    train_flow = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="raw",
        batch_size=batch_size,
        shuffle=True,
        subset="training",
        seed=42,
    )

    valid_flow = valid_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        subset="validation",
        seed=42,
    )

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_flow,
        validation_data=valid_flow,
        epochs=3,
        verbose=1,
        workers=max(1, (os.cpu_count() or 2) // 2),
        use_multiprocessing=True,
        max_queue_size=16,
    )



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421435677.py in <cell line: 0>()
     63     # Speed: Keras generators can use multiprocessing workers to overlap JPEG decode/resize with GPU/CPU compute.
     64     # Correctness: data order/shuffle behavior remains defined by seed/shuffle; workers just parallelize loading.
---> 65     model.fit(
     66         train_flow,
     67         validation_data=valid_flow,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 15
TTA_DATAGEN = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.05,
    height_shift_range=0.05,
    width_shift_range=0.05,
)


def _load_batch_images(image_ids):
    out = np.empty((len(image_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        out[i] = load_single_image("TEST_DATA", image_id)
    return out


def get_augmented_batch_from_images(images_batch, n_aug=5):
    b = images_batch.shape[0]
    it = TTA_DATAGEN.flow(images_batch, batch_size=b, shuffle=False)
    batches = [images_batch]
    for _ in range(n_aug):
        batches.append(next(it))
    aug = np.concatenate(batches, axis=0)
    return aug




## === cell 16
image_ids = test_df["image_id"].values
pred_labels = np.empty(len(image_ids), dtype=np.int64)

n_aug = 5
views = 1 + n_aug
bsize = batch_size  # keep consistent

for start in range(0, len(image_ids), bsize):
    end = min(start + bsize, len(image_ids))
    batch_ids = image_ids[start:end]

    base_imgs = _load_batch_images(batch_ids)  # (B,H,W,3)
    aug_batch = get_augmented_batch_from_images(
        base_imgs, n_aug=n_aug
    )  # (B*views,H,W,3)

    probs = model.predict(aug_batch, verbose=0, batch_size=bsize * views)  # (B*views,5)
    probs = probs.reshape((views, end - start, -1))  # (views,B,5)
    mean_probs = probs.mean(axis=0)  # (B,5)
    pred_labels[start:end] = np.argmax(mean_probs, axis=1).astype(np.int64)

test_results_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
test_results_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_results_df.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1254124246.py in <cell line: 0>()
     11     batch_ids = image_ids[start:end]
     12 
---> 13     base_imgs = _load_batch_images(batch_ids)  # (B,H,W,3)
     14     aug_batch = get_augmented_batch_from_images(
     15         base_imgs, n_aug=n_aug

/tmp/ipykernel_11/3804665257.py in _load_batch_images(image_ids)
     20     out = np.empty((len(image_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
     21     for i, image_id in enumerate(image_ids):
---> 22         out[i] = load_single_image("TEST_DATA", image_id)
     23     return out
     24 

/tmp/ipykernel_11/1904370484.py in load_single_image(data_type, image_id)
      7         image_path = os.path.join(TRAIN_DIR, image_id)
      8 
----> 9     with Image.open(image_path) as img:
     10         img = img.convert("RGB")
     11         img = img.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'

## === cell 17
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
submission = pd.read_csv("submission.csv")

print("Sample columns:", sample_sub.columns.tolist(), "rows:", sample_sub.shape[0])
print("Submission columns:", submission.columns.tolist(), "rows:", submission.shape[0])

missing = set(sample_sub["image_id"]) - set(submission["image_id"])
extra = set(submission["image_id"]) - set(sample_sub["image_id"])
print("Missing ids:", len(missing), "Extra ids:", len(extra))
submission.head(3)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1225753631.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
----> 2 submission = pd.read_csv("submission.csv")
      3 
      4 print("Sample columns:", sample_sub.columns.tolist(), "rows:", sample_sub.shape[0])
      5 print("Submission columns:", submission.columns.tolist(), "rows:", submission.shape[0])

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

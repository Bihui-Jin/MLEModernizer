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

0.8833484436385615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import matplotlib.pyplot as plt
from PIL import Image



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.load(file)
print(json.dumps(map_classes, indent=2))



## === cell 3
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 4
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = "../input/xceptionv2/Cassava_Model_V06.hdf5"  # may not exist



## === cell 6
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/941929707.py in <cell line: 0>()
      1 # Albumentations imports – removed unavailable Flip augmentation
----> 2 from albumentations import (
      3     Compose,
      4     HorizontalFlip,
      5     CLAHE,

ImportError: cannot import name 'RandomBrightness' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 7
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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4276477694.py in <cell line: 0>()
      2     [
      3         HorizontalFlip(p=0.5),
----> 4         RandomContrast(limit=0.2, p=0.5),
      5         RandomBrightness(limit=0.2, p=0.5),
      6         CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),

NameError: name 'RandomContrast' is not defined

## === cell 8
import random
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
    return np.array(img)


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
                if random.random() > 0.5:
                    img = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img = load_single_image(self.data_type, x)
            else:
                if self.data_type == "VALIDATE_DATA":
                    img = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img = load_single_image(self.data_type, x)
            img_list.append(img)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({"image_id": test_filenames})
print("Test samples:", test_df.shape[0])



## === cell 10
test_gen = AugmentedImageSequence(
    mode="TEST",
    data_set_type="TEST_DATA",
    x_set=test_df["image_id"].values,
    y_set=None,
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/853911306.py in <cell line: 0>()
      5     y_set=None,
      6     batch_size=batch_size,
----> 7     augmentations=AUGMENTATIONS_TEST,
      8 )
      9 

NameError: name 'AUGMENTATIONS_TEST' is not defined

## === cell 11
from keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam




## === cell 12
def build_fresh_model():
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    )
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(5, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


try:
    model = load_model(PRE_TRAINED_MODEL)
    print("Pre‑trained model loaded.")
except Exception as e:
    print(f"Could not load pre‑trained model ({e}), building a new one.")
    model = build_fresh_model()

    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_gen = AugmentedImageSequence(
        mode="TRAIN",
        data_set_type="TRAIN_DATA",
        x_set=train_df["image_id"].values,
        y_set=train_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    model.fit(train_gen, epochs=2, verbose=1)

model.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2128056608.py in <cell line: 0>()
     18 try:
---> 19     model = load_model(PRE_TRAINED_MODEL)
     20     print("Pre‑trained model loaded.")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/xceptionv2/Cassava_Model_V06.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2128056608.py in <cell line: 0>()
     31         y_set=train_df["label"].values,
     32         batch_size=batch_size,
---> 33         augmentations=AUGMENTATIONS_TRAIN,
     34     )
     35     # Train for a few epochs (adjust as needed for time constraints)

NameError: name 'AUGMENTATIONS_TRAIN' is not defined

## === cell 13
test_results_list = []
for image_id in test_df["image_id"]:
    image_path = os.path.join(TEST_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
    img_array = np.expand_dims(np.array(img), axis=0)  # shape (1, H, W, 3)
    preds = model.predict(img_array, verbose=0)
    pred_label = int(np.argmax(preds, axis=1)[0])
    test_results_list.append({"image_id": image_id, "label": pred_label})

test_results_df = pd.DataFrame(test_results_list)
submission_path = "submission.csv"
test_results_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(test_results_df)}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/557788016.py in <cell line: 0>()
      2 for image_id in test_df["image_id"]:
      3     image_path = os.path.join(TEST_DIR, image_id)
----> 4     img = Image.open(image_path).convert("RGB")
      5     img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
      6     img_array = np.expand_dims(np.array(img), axis=0)  # shape (1, H, W, 3)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'

## === cell 14
submission = pd.read_csv(submission_path)
print(submission.head(3))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/828181052.py in <cell line: 0>()
----> 1 submission = pd.read_csv(submission_path)
      2 print(submission.head(3))

NameError: name 'submission_path' is not defined

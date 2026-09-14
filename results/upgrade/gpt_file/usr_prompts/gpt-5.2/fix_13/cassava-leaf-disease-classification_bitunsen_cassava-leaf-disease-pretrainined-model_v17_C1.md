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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.exists(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
assert os.path.exists(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"



## === cell 2
import matplotlib.pyplot as plt
import cv2
from PIL import Image

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    PIL_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    PIL_RESAMPLE = Image.LANCZOS



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

PRE_TRAINED_MODEL_CANDIDATES = [
    "/kaggle/input/unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5",
    "/kaggle/input/cassava-best-unitedmodel-v04/Cassava_Best_UnitedModel_V04.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in PRE_TRAINED_MODEL_CANDIDATES if os.path.exists(p)),
    PRE_TRAINED_MODEL_CANDIDATES[0],
)
PRE_TRAINED_MODEL



## === cell 7
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    RandomBrightnessContrast,
    RandomGamma,
    ToFloat,
    ShiftScaleRotate,
)



## === cell 8
AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.2),
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
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 10
from tensorflow.keras.utils import Sequence


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return img_rgb  # uint8 RGB


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x = list(x_set)
        self.y = None if y_set is None else np.asarray(list(y_set), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_x = self.x[start:end]

        if self.mode == "TEST":
            batch_y = None
        else:
            batch_y = self.y[start:end]

        bs = len(batch_x)
        img_array = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

        data_type = self.data_type
        augment = self.augment
        rand_uniform = random.uniform
        _load = load_single_image

        if data_type == "TRAIN_DATA":
            for i, x in enumerate(batch_x):
                base_img = _load(data_type, x)
                if rand_uniform(0, 1) > 0.5:
                    img_data = augment(image=base_img)["image"]
                else:
                    img_data = base_img
                img_array[i] = np.asarray(img_data, dtype=np.float32)
        elif data_type == "VALIDATE_DATA":
            for i, x in enumerate(batch_x):
                base_img = _load(data_type, x)
                img_data = augment(image=base_img)["image"]
                img_array[i] = np.asarray(img_data, dtype=np.float32)
        else:
            for i, x in enumerate(batch_x):
                base_img = _load(data_type, x)
                img_array[i] = np.asarray(base_img, dtype=np.float32)

        if batch_y is None:
            return img_array
        return img_array, np.asarray(batch_y)




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
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 13
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

num_classes = train_df["label"].nunique()
print("Train rows:", len(train_df), "Num classes:", num_classes)



## === cell 14
from tensorflow.keras.models import load_model

model = None
if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model:", PRE_TRAINED_MODEL)
else:
    print("Pretrained model not found, will train a fallback model:", PRE_TRAINED_MODEL)



## === cell 15
if model is None:
    from sklearn.model_selection import train_test_split

    tr_df, va_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )

    train_gen = AugmentedImageSequence(
        mode="TRAIN",
        data_set_type="TRAIN_DATA",
        x_set=tr_df["image_id"].values,
        y_set=tr_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        mode="VALIDATE",
        data_set_type="VALIDATE_DATA",
        x_set=va_df["image_id"].values,
        y_set=va_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Rescaling(1.0)(inputs)  # keep identical behavior
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
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

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1,
    )



## === cell 16
model.summary()



## === cell 17
pred_proba = model.predict(
    test_gen,
    verbose=1,
)
pred_labels = np.argmax(pred_proba, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    submission["label"] = submission["label"].fillna(0).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head(3))
print(
    "Wrote submission.csv with rows:",
    len(submission),
    "cols:",
    list(submission.columns),
)

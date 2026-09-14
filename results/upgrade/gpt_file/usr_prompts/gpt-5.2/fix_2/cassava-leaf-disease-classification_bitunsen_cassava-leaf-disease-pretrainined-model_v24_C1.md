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



## === cell 10
from tensorflow.keras.utils import Sequence
from PIL import Image


def _pil_resample():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), resample=_pil_resample())
    img_data = np.array(img)
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




## === cell 11
test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 12
test_gen = AugmentedImageSequence(
    "TEST",
    "TEST_DATA",
    test_df["image_id"].values,
    None,
    batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 13
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



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
train_df["label"] = train_df["label"].astype(
    str
)  # flow_from_dataframe expects string for categorical

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

    model.get_layer(index=1).trainable = (
        False  # base model is layer 1 in this graph (after Input)
    )
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
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMG_HEIGHT, IMG_WIDTH), resample=_pil_resample())
        image_data = np.array(image, dtype=np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        predict_class = model.predict(image_data, verbose=0)
        test_results_list.append(
            {"image_id": image_id, "label": int(np.argmax(predict_class, axis=1)[0])}
        )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 18
def get_augmented_images(image_id):
    image_path = os.path.join(TEST_DIR, image_id)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH), resample=_pil_resample())
    image_data = np.array(image, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)

    tta_datagen = ImageDataGenerator(
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
    it = tta_datagen.flow(image_data, batch_size=1, shuffle=False)

    image_list = []
    image_list.append(image_data / 255.0)
    for _ in range(5):
        batch = next(it)
        image_list.append(batch)
    return image_list




## === cell 19
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_ids = sample_sub["image_id"].tolist()
sample_set = set(sample_ids)

test_results = {}
for image_id in sample_ids:
    if image_id not in sample_set:
        continue
    image_list = get_augmented_images(image_id)
    preds = []
    for image_data in image_list:
        p = model.predict(image_data, verbose=0)[0]
        preds.append(p)
    mean_p = np.mean(np.stack(preds, axis=0), axis=0)
    test_results[image_id] = int(np.argmax(mean_p))

submission = pd.DataFrame(
    {"image_id": sample_ids, "label": [test_results[iid] for iid in sample_ids]}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())



## === cell 20
submission_check = pd.read_csv("submission.csv")
print(submission_check.shape)
print(submission_check.columns.tolist())
print(submission_check.head(3))

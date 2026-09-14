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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from PIL import Image

warnings.simplefilter("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification"

print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("Train images:", len(os.listdir(os.path.join(WORK_DIR, "train_images"))))
print("Test images:", len(os.listdir(os.path.join(WORK_DIR, "test_images"))))

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()



## === cell 2
BATCH_SIZE = 8
STEPS_PER_EPOCH = len(train_labels) * 0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels) * 0.2 / BATCH_SIZE
EPOCHS = 20
TARGET_SIZE = 350



## === cell 3
base_path = Path("../input/cassava-leaf-disease-classification")
train_img_dir = base_path / "train_images"
test_img_dir = base_path / "test_images"



## === cell 4
train_df = pd.read_csv(base_path / "train.csv")
diseaseMapping = pd.read_json(base_path / "label_num_to_disease_map.json", typ="series")

train_images = os.listdir(base_path / "train_images/")
test_images = os.listdir(base_path / "test_images/")



## === cell 5
diseaseMapping



## === cell 6
mappingDict = diseaseMapping.to_dict()



## === cell 7
train_df.head()



## === cell 8
train_df = train_df.replace(mappingDict)



## === cell 9
labelCounts = train_df["label"].value_counts().reset_index()
labelCounts.columns = ["Label", "Number of Observations"]
labelCounts.head()



## === cell 10
uniqueIds = train_df["image_id"].nunique()
if uniqueIds == len(train_df):
    print("There are no repeating Image IDs in the dataset")
else:
    print(f"There are {len(train_df) - uniqueIds} repeating Image IDs")



## === cell 11
print(len(train_images))



## === cell 12
healthyImages = train_df[train_df["label"] == "Healthy"]["image_id"].to_list()
cbbImages = train_df[train_df["label"] == "Cassava Bacterial Blight (CBB)"][
    "image_id"
].to_list()
cbsdImages = train_df[train_df["label"] == "Cassava Brown Streak Disease (CBSD)"][
    "image_id"
].to_list()
cgmImages = train_df[train_df["label"] == "Cassava Green Mottle (CGM)"][
    "image_id"
].to_list()
cmdImages = train_df[train_df["label"] == "Cassava Mosaic Disease (CMD)"][
    "image_id"
].to_list()



## === cell 13
import matplotlib.pyplot as plt


def showImages(images):
    random_images = [np.random.choice(images) for _ in range(9)]
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(train_img_dir / random_images[i])
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()




## === cell 14
def showHistogram(sample_img, title):
    f = plt.figure(figsize=(16, 8))
    f.add_subplot(1, 2, 1)

    raw_image = plt.imread(train_img_dir / sample_img)
    plt.imshow(raw_image, cmap="gray")
    plt.colorbar()
    plt.title(title)
    print(f"Image dimensions:  {raw_image.shape[0], raw_image.shape[1]}")
    print(
        f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}"
    )
    print(
        f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}"
    )

    f.add_subplot(1, 2, 2)

    _ = plt.hist(raw_image[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 1].ravel(), bins=256, color="Green", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 2].ravel(), bins=256, color="Blue", alpha=0.5)
    _ = plt.xlabel("Intensity Value")
    _ = plt.ylabel("Count")
    _ = plt.legend(["Red_Channel", "Green_Channel", "Blue_Channel"])
    plt.show()




## === cell 15
def load_image_pil_rgb(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size))
    return np.asarray(img)




## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
pass



## === cell 35
pass



## === cell 36
channelIntensityDf = pd.DataFrame(
    {
        "Leaf Type": ["Healthy", "CBB", "CBSD", "CGM", "CMD"],
        "Red Channel Mean": [108, 102, 106, 113, 110],
        "Green Channel Mean": [126, 117, 123, 128, 128],
        "Blue Channel Mean": [80, 66, 72, 85, 80],
    }
)
channelIntensityDf



## === cell 37
train_labels.label = train_labels.label.astype("str")



## === cell 38
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=None,
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    rescale=1.0 / 255.0,  # keep consistent scaling for both train and inference
)



## === cell 39
train_generator = train_datagen.flow_from_dataframe(
    train_labels,
    directory=os.path.join(WORK_DIR, "train_images"),
    subset="training",
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=SEED,
    shuffle=True,
)



## === cell 40
validation_datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0 / 255.0)



## === cell 41
validation_generator = validation_datagen.flow_from_dataframe(
    train_labels,
    directory=os.path.join(WORK_DIR, "train_images"),
    subset="validation",
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=SEED,
    shuffle=False,
)




## === cell 42
def create_model():
    conv_base = EfficientNetB3(
        include_top=False, weights=None, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5, activation="softmax")(x)
    model = models.Model(conv_base.input, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model




## === cell 43
model = create_model()
model.summary()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=int(np.ceil(STEPS_PER_EPOCH)),
    validation_data=validation_generator,
    validation_steps=int(np.ceil(VALIDATION_STEPS)),
    verbose=2,
)



## === cell 44
model.save("./EffNetB0_512_8.h5")



## === cell 45
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
ss.head()



## === cell 46
preds = []



## === cell 47
for image_id in ss.image_id:
    img_path = os.path.join(WORK_DIR, "test_images", image_id)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((TARGET_SIZE, TARGET_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    p = model.predict(arr, verbose=0)
    preds.append(int(np.argmax(p, axis=1)[0]))



## === cell 48
y_predict = p
y_predict



## === cell 49
print("Number of predictions:", len(preds))
print("First 10 predictions:", preds[:10])



## === cell 50
assert len(preds) == len(ss), f"Pred length {len(preds)} != submission length {len(ss)}"



## === cell 51
ss["label"] = pd.Series(preds, dtype="int64")
ss.head()



## === cell 52
ss.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", ss.shape)



## === cell 53
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image_id", "label"]
print(check.head())

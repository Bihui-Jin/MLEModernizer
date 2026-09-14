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
import os, json, warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification"

print("Files in WORK_DIR:", os.listdir(WORK_DIR)[:10])
print("Train images:", len(os.listdir(os.path.join(WORK_DIR, "train_images"))))
print("Test images:", len(os.listdir(os.path.join(WORK_DIR, "test_images"))))

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()



## === cell 2
BATCH_SIZE = 8
EPOCHS = 20
TARGET_SIZE = 350

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
STEPS_PER_EPOCH = int(np.ceil(len(train_labels) * TRAIN_SPLIT / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train_labels) * VAL_SPLIT / BATCH_SIZE))

print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)



## === cell 3
from pathlib import Path

base_path = Path(WORK_DIR)
train_img_dir = base_path / "train_images"
test_img_dir = base_path / "test_images"



## === cell 4
train_df = pd.read_csv(base_path / "train.csv")
diseaseMapping = pd.read_json(base_path / "label_num_to_disease_map.json", typ="series")

train_images = os.listdir(base_path / "train_images")
test_images = os.listdir(base_path / "test_images")

print(train_df.shape, len(train_images), len(test_images))



## === cell 5
diseaseMapping



## === cell 6
mappingDict = diseaseMapping.to_dict()
mappingDict



## === cell 7
train_df.head()



## === cell 8
train_df_explore = train_df.replace(mappingDict)
train_df_explore.head()



## === cell 9
labelCounts = train_df_explore["label"].value_counts().reset_index()
labelCounts.columns = ["Label", "Number of Observations"]
labelCounts



## === cell 10
uniqueIds = train_df["image_id"].nunique()
if uniqueIds == len(train_df):
    print("There are no repeating Image IDs in the dataset")
else:
    print(f"There are {len(train_df) - uniqueIds} repeating Image IDs")



## === cell 11
print("Number of train image files:", len(train_images))



## === cell 12
healthyImages = train_df_explore[train_df_explore["label"] == "Healthy"][
    "image_id"
].to_list()
cbbImages = train_df_explore[
    train_df_explore["label"] == "Cassava Bacterial Blight (CBB)"
]["image_id"].to_list()
cbsdImages = train_df_explore[
    train_df_explore["label"] == "Cassava Brown Streak Disease (CBSD)"
]["image_id"].to_list()
cgmImages = train_df_explore[train_df_explore["label"] == "Cassava Green Mottle (CGM)"][
    "image_id"
].to_list()
cmdImages = train_df_explore[
    train_df_explore["label"] == "Cassava Mosaic Disease (CMD)"
]["image_id"].to_list()

[len(healthyImages), len(cbbImages), len(cbsdImages), len(cgmImages), len(cmdImages)]




## === cell 13
def showImages(images):
    if plt is None:
        print("matplotlib not available; skipping showImages.")
        return
    if len(images) == 0:
        print("No images provided.")
        return

    random_images = [np.random.choice(images) for _ in range(9)]
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(str(train_img_dir / random_images[i]))
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()




## === cell 14
def showHistogram(sample_img, title):
    if plt is None:
        print("matplotlib not available; skipping showHistogram.")
        return

    raw_image = plt.imread(str(train_img_dir / sample_img))
    f = plt.figure(figsize=(16, 8))
    f.add_subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.colorbar()
    plt.title(title)
    print(f"Image dimensions: {(raw_image.shape[0], raw_image.shape[1])}")
    print(
        f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}"
    )
    print(
        f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}"
    )

    f.add_subplot(1, 2, 2)
    _ = plt.hist(raw_image[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
    _ = plt.xlabel("Intensity Value")
    _ = plt.ylabel("Count")
    _ = plt.legend(["Red_Channel", "Green_Channel", "Blue_Channel"])
    plt.show()




## === cell 15
import cv2


def load_image(image_id):
    image = cv2.imread(str(train_img_dir / image_id))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def showChannelDistribution(images, leafType):
    print(
        "Channel distribution visualization skipped (plotly not used in this runtime-stable version)."
    )
    return None




## === cell 16
def showBoxPlot(histData, leafType):
    print(
        "Box plot visualization skipped (plotly not used in this runtime-stable version)."
    )




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
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype(str)
train_labels.head()



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
    shuffle=True,
    seed=42,
)



## === cell 40
validation_datagen = ImageDataGenerator(validation_split=0.2)



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

ckpt_path = "EffNetB3_TARGET350_BS8_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_acc", save_best_only=True, mode="max", verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

model = tf.keras.models.load_model(ckpt_path)



## === cell 44
model.save("./EffNetB0_512_8.h5")



## === cell 45
print("Our EfficientNet CNN has %d layers" % len(model.layers))



## === cell 46
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
ss.head()



## === cell 47
test_datagen = ImageDataGenerator(preprocessing_function=None)

test_generator = test_datagen.flow_from_dataframe(
    ss,
    directory=os.path.join(WORK_DIR, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)



## === cell 48
pred_probs = model.predict(test_generator, verbose=1)
preds = np.argmax(pred_probs, axis=1).astype(int).tolist()

print("Predictions:", len(preds), "Expected:", len(ss))



## === cell 49
pred_probs[0]



## === cell 50
print("First 20 predicted labels:", preds[:20])



## === cell 51
submission = ss.copy()
submission["label"] = preds
submission.head()



## === cell 52
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)



## === cell 53
assert list(submission.columns) == ["image_id", "label"]
assert submission["image_id"].nunique() == len(submission)
assert len(submission) == len(ss)
print("Submission format looks valid.")

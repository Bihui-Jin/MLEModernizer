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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("../input/cassava-leaf-disease-classification"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
import warnings

warnings.simplefilter("ignore")
import json
from PIL import Image



## === cell 2
WORK_DIR = "../input/cassava-leaf-disease-classification"
os.listdir(WORK_DIR)
print("Train images: %d" % len(os.listdir(os.path.join(WORK_DIR, "train_images"))))

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()



## === cell 3
BATCH_SIZE = 8
STEPS_PER_EPOCH = int(len(train_labels) * 0.8 // BATCH_SIZE)
VALIDATION_STEPS = int(len(train_labels) * 0.2 // BATCH_SIZE)
EPOCHS = 1  # short training just to have a fitted model
TARGET_SIZE = 350



## === cell 4
base_path = Path("../input/cassava-leaf-disease-classification")
train_img_dir = base_path / "train_images"
test_img_dir = base_path / "test_images"



## === cell 5
train_df = pd.read_csv(base_path / "train.csv")
diseaseMapping = pd.read_json(base_path / "label_num_to_disease_map.json", typ="series")

train_images = os.listdir(base_path / "train_images/")
test_images = os.listdir(base_path / "test_images/")



## === cell 6
diseaseMapping



## === cell 7
mappingDict = diseaseMapping.to_dict()



## === cell 8
train_df.head()



## === cell 9
train_df = train_df.replace(mappingDict)



## === cell 10
labelCounts = train_df["label"].value_counts().reset_index()
labelCounts.columns = ["Label", "Number of Observations"]

fig = px.pie(
    labelCounts,
    names="Label",
    values="Number of Observations",
    labels=mappingDict,
    color_discrete_sequence=px.colors.sequential.YlOrBr,
)


fig.show()



## === cell 11
uniqueIds = train_df["image_id"].nunique()
if uniqueIds == len(train_df):
    print("There are no repeating Image IDs in the dataset")
else:
    print(f"There are {len(train_df) - uniqueIds} repeating Image IDs")



## === cell 12
print(len(train_images))



## === cell 13
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




## === cell 14
def showImages(images):

    random_images = [np.random.choice(images) for i in range(9)]

    plt.figure(figsize=(10, 8))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(train_img_dir / random_images[i])
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()




## === cell 15
def showHistogram(sample_img, title):
    f = plt.figure(figsize=(16, 8))
    f.add_subplot(1, 2, 1)

    raw_image = plt.imread(train_img_dir / sample_img)
    plt.imshow(raw_image, cmap="gray")
    plt.colorbar()
    plt.title(title)
    print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
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




## === cell 16
def load_image(image_id):
    image = cv2.imread(str(train_img_dir / image_id))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def showChannelDistribution(images, leafType):
    imageArray = [load_image(image_id) for image_id in images]

    red_values = [np.mean(imageArray[idx][:, :, 0]) for idx in range(len(imageArray))]
    green_values = [np.mean(imageArray[idx][:, :, 1]) for idx in range(len(imageArray))]
    blue_values = [np.mean(imageArray[idx][:, :, 2]) for idx in range(len(imageArray))]
    values = [np.mean(imageArray[idx]) for idx in range(len(imageArray))]

    hist_data = [red_values, green_values, blue_values, values]
    group_labels = ["Red", "Green", "Blue", "All"]

    fig = ff.create_distplot(
        hist_data, group_labels, colors=["red", "green", "blue", "grey"]
    )
    fig.update_layout(
        template="plotly_white", title_text=f"Channel Distribution - {leafType}"
    )
    fig.show()
    return hist_data




## === cell 17
def showBoxPlot(histData, leafType):
    figData = []
    for i, name in zip(range(3), ["Red", "Green", "Blue"]):
        trace = go.Box(y=histData[i], name=name, boxpoints="all", marker_color=name)
        figData.append(trace)

    fig = go.Figure(figData)
    fig.update_layout(
        title_text=f"Pixel Intensity Distribution - {leafType}", template="plotly_white"
    )
    fig.show()




## === cell 18
showImages(healthyImages)



## === cell 19
showHistogram(healthyImages[0], "Healthy Image")



## === cell 20
data = showChannelDistribution(healthyImages, "Healthy")



## === cell 21
showBoxPlot(data, "Healthy Leaves")



## === cell 22
showImages(cbbImages)



## === cell 23
showHistogram(cbbImages[0], "CBB Image")



## === cell 24
data = showChannelDistribution(cbbImages, "CBB Images")



## === cell 25
showBoxPlot(data, "CBB Images")



## === cell 26
showImages(cbsdImages)



## === cell 27
showHistogram(cbsdImages[0], "CBSD Image")



## === cell 28
data = showChannelDistribution(cbsdImages, "CBSD Images")



## === cell 29
showImages(cgmImages)



## === cell 30
showHistogram(cgmImages[0], "CGM Image")



## === cell 31
data = showChannelDistribution(cgmImages, "CGM Images")



## === cell 32
showBoxPlot(data, "CGM Images")



## === cell 33
showImages(cmdImages)



## === cell 34
showHistogram(cmdImages[0], "CMD Image")



## === cell 35
data = showChannelDistribution(cmdImages[:2000], "CMD Images")



## === cell 36
showBoxPlot(data, "CMD Images")



## === cell 37
channelIntensityDf = pd.DataFrame(
    {
        "Leaf Type": ["Healthy", "CBB", "CBSD", "CGM", "CMD"],
        "Red Channel Mean": [108, 102, 106, 113, 110],
        "Green Channel Mean": [126, 117, 123, 128, 128],
        "Blue Channel Mean": [80, 66, 72, 85, 80],
    }
)

channelIntensityDf.style.background_gradient(cmap="Greens", axis=0)



## === cell 38
train_labels.label = train_labels.label.astype("str")



## === cell 39
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



## === cell 40
train_generator = train_datagen.flow_from_dataframe(
    train_labels,
    directory=os.path.join(WORK_DIR, "train_images"),
    subset="training",
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
)



## === cell 41
validation_datagen = ImageDataGenerator(validation_split=0.2)



## === cell 42
validation_generator = validation_datagen.flow_from_dataframe(
    train_labels,
    directory=os.path.join(WORK_DIR, "train_images"),
    subset="validation",
    x_col="image_id",
    y_col="label",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
)




## === cell 43
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




## === cell 44
model = create_model()
model.summary()
model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    verbose=0,
)



## === cell 45
model.save("./EffNetB3_350_8.h5")



## === cell 46
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
ss.head()



## === cell 47
preds = []



## === cell 48
for image_id in ss.image_id:
    img = Image.open(os.path.join(WORK_DIR, "test_images", image_id)).convert("RGB")
    img = img.resize((TARGET_SIZE, TARGET_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    pred = np.argmax(model.predict(img_array, verbose=0), axis=1)[0]
    preds.append(int(pred))



## === cell 49
print("prediction is :", preds[:10], "...", f"total {len(preds)} predictions")



## === cell 50
ss["label"] = preds
ss.to_csv("submission.csv", index=False)

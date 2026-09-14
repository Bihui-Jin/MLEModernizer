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

albumentations==2.0.8
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.0018

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import warnings

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import albumentations as aug

warnings.simplefilter("ignore")

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

base_walk = "/kaggle/input"
for dirname, _, filenames in os.walk(base_walk):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    break




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
general_path = "../input/cassava-leaf-disease-classification/"
print("Exists:", os.path.exists(general_path))
print(os.listdir(general_path)[:10])




## === cell 2
with open(os.path.join(general_path, "label_num_to_disease_map.json"), "r") as file:
    map_classes = json.loads(file.read())
map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))




## === cell 3
input_files = os.listdir(os.path.join(general_path, "train_images"))
print(f"Number of train images: {len(input_files)}")




## === cell 4
img_shapes = {}
print("Skipping image shape scan for performance (not used downstream).")




## === cell 5
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()




## === cell 6
print("Skipping class distribution plot for performance (not used downstream).")




## === cell 7
def visualize_batch(image_ids, labels, class_names):
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label, class_name) in enumerate(
        zip(image_ids, labels, class_names)
    ):
        if ind >= 9:
            break
        plt.subplot(3, 3, ind + 1)
        img = cv2.imread(os.path.join(general_path, "train_images", image_id))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class {label}:  {class_name}", fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 8
print("Skipping class-0 sample visualization for performance.")




## === cell 9
print("Skipping class-1 sample visualization for performance.")




## === cell 10
print("Skipping class-2 sample visualization for performance.")




## === cell 11
print("Skipping class-3 sample visualization for performance.")




## === cell 12
print("Skipping class-4 sample visualization for performance.")




## === cell 13
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(general_path, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("original")

    for i in range(2, 5):
        plt.subplot(2, 2, i)
        x = transform(image=img)["image"]
        plt.imshow(x)
        plt.axis("off")
        plt.title(f"Augmentation -{i-1}")

    plt.show()




## === cell 14
transform_shift_scale_rotate = aug.ShiftScaleRotate(
    p=1.0,
    shift_limit=(-0.3, 0.3),
    scale_limit=(-0.1, 0.1),
    rotate_limit=(-180, 180),
    interpolation=0,
    border_mode=4,
)
print("Skipping augmentation demo (ShiftScaleRotate).")




## === cell 15
transform_coarse_dropout = aug.CoarseDropout(
    p=1.0,
    max_holes=100,
    max_height=50,
    max_width=50,
    min_holes=30,
    min_height=20,
    min_width=20,
)
print("Skipping augmentation demo (CoarseDropout).")




## === cell 16
transform_hsv = aug.HueSaturationValue(
    hue_shift_limit=0,
    sat_shift_limit=(40, 80),
    val_shift_limit=(40, 80),
    p=1.0,
)
print("Skipping augmentation demo (HueSaturationValue).")




## === cell 17
transform_clahe = aug.CLAHE(
    p=1.0,
    clip_limit=(10, 30),
    tile_grid_size=(10, 10),
)
print("Skipping augmentation demo (CLAHE).")




## === cell 18
transform_fog = aug.RandomFog(p=1.0)
print("Skipping augmentation demo (RandomFog).")




## === cell 19
transform_sunflare = aug.RandomSunFlare(p=1.0)
print("Skipping augmentation demo (RandomSunFlare).")




## === cell 20
transform_brightness_contrast = aug.RandomBrightnessContrast(p=1.0)
print("Skipping augmentation demo (RandomBrightnessContrast).")




## === cell 21
transform_randomcrop = aug.RandomCrop(p=1.0, height=512, width=512)
print("Skipping augmentation demo (RandomCrop).")




## === cell 22
transform_rgbshift = aug.RGBShift(p=1.0)
print("Skipping augmentation demo (RGBShift).")




## === cell 23
transform_snow = aug.RandomSnow(p=1.0)
print("Skipping augmentation demo (RandomSnow).")




## === cell 24
transform_hflip = aug.HorizontalFlip(p=1.0)
print("Skipping augmentation demo (HorizontalFlip).")




## === cell 25
transform_vflip = aug.VerticalFlip(p=1.0)
print("Skipping augmentation demo (VerticalFlip).")




## === cell 26
transform_transpose = aug.Transpose(p=1.0)
print("Skipping augmentation demo (Transpose).")




## === cell 27
img_width, img_height = 224, 224

train = pd.read_csv(os.path.join(general_path, "train.csv"))
train["label"] = train["label"].astype("string")
train.head()




## === cell 28
datagen = ImageDataGenerator(
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

train_datagen_flow = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(general_path, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    subset="training",
    class_mode="categorical",  # critical: one-hot for categorical_crossentropy
    shuffle=True,
    seed=42,
)




## === cell 29
valid_datagen_flow = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(general_path, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    subset="validation",
    class_mode="categorical",
    shuffle=False,
    seed=42,
)




## === cell 30
x, y = next(iter(train_datagen_flow))
print("Batch X shape:", x.shape, "Batch y shape:", y.shape)




## === cell 31
from tensorflow.keras.applications import EfficientNetB0

backbone = EfficientNetB0(
    weights="imagenet",
    input_shape=(img_width, img_height, 3),
    pooling="avg",
    include_top=False,
)
backbone.summary()




## === cell 32
opt = Adam(learning_rate=5e-4)

model = Sequential()
model.add(backbone)
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()




## === cell 33
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)




## === cell 34
history = model.fit(
    train_datagen_flow,
    validation_data=valid_datagen_flow,
    epochs=40,
    verbose=1,
    callbacks=[early_stop],
    workers=min(4, os.cpu_count() or 1),
    use_multiprocessing=True,
    max_queue_size=32,
)




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/741197081.py in <cell line: 0>()
      2 # This does NOT change the training loop/epochs/early-stopping/model; it only improves input throughput.
      3 # Keep workers bounded to avoid process-spawn overhead; enable multiprocessing for PIL decode/augment.
----> 4 history = model.fit(
      5     train_datagen_flow,
      6     validation_data=valid_datagen_flow,

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

## === cell 35
model.save("submission.h5")




## === cell 36
print("Skipping training curves plot for performance.")




## === cell 37
ss = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))

test_datagen = ImageDataGenerator()
test_flow = test_datagen.flow_from_dataframe(
    dataframe=ss,
    directory=os.path.join(general_path, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode=None,
    shuffle=False,
)

pred_proba = model.predict(
    test_flow,
    verbose=1,
    workers=min(4, os.cpu_count() or 1),
    use_multiprocessing=True,
    max_queue_size=32,
)
preds = np.argmax(pred_proba, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1910091030.py in <cell line: 0>()
     14 
     15 # Timeout fix: enable multi-worker loading for prediction as well; identical predictions.
---> 16 pred_proba = model.predict(
     17     test_flow,
     18     verbose=1,

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

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

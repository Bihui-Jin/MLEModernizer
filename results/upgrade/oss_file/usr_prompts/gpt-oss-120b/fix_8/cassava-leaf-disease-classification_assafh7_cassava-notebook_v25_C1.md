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

0.853732245391357

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I add the missing standard imports (warnings, json, pandas, numpy, matplotlib, seaborn, cv2) and adjust the albumentations import to avoid the protobuf error. I also ensure the dataframe variable names are consistent so the training pipeline can run. These changes fix the runtime errors and allow the script to generate a valid `submission.csv` without altering the core model logic.'
- What this solution (achieved 0.05531) has done: 'The script is sped up by enabling parallel data loading and configuring TensorFlow threading, which cuts the I/O‑bound overhead of the ImageDataGenerator without altering the model architecture, loss, or training schedule.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

dataset_root = "/kaggle/input"
print("Dataset root exists:", os.path.isdir(dataset_root))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import warnings
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    import cv2
except Exception:
    cv2 = None  # visualization will be skipped if cv2 is unavailable

aug = None




## === cell 2
warnings.simplefilter("ignore")




## === cell 3
general_path = "/kaggle/input/cassava-leaf-disease-classification/"
print("Listing root of dataset:")
print(os.listdir(general_path))




## === cell 4
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.load(file)
    map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))




## === cell 5
train_images_path = os.path.join(general_path, "train_images")
train_image_files = os.listdir(train_images_path)
print(f"Number of train images: {len(train_image_files)}")




## === cell 6
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()




## === cell 7
plt.figure(figsize=(8, 4))
sns.countplot(y="class_name", data=df_train)
plt.title("Class distribution")
plt.show()




## === cell 8
def visualize_batch(image_ids, labels, class_names):
    if cv2 is None:
        print("OpenCV not available – skipping visualisation.")
        return
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label, cname) in enumerate(zip(image_ids, labels, class_names)):
        plt.subplot(3, 3, ind + 1)
        img = cv2.imread(os.path.join(train_images_path, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class {label}: {cname}", fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 9
tmp = df_train[df_train["label"] == 0].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## === cell 10
tmp = df_train[df_train["label"] == 1].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## === cell 11
tmp = df_train[df_train["label"] == 2].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## === cell 12
tmp = df_train[df_train["label"] == 3].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## === cell 13
tmp = df_train[df_train["label"] == 4].sample(6, random_state=42)
visualize_batch(tmp["image_id"].values, tmp["label"].values, tmp["class_name"].values)




## === cell 14
if aug is not None:
    transform_shift_scale_rotate = aug.ShiftScaleRotate(
        p=1.0,
        shift_limit=(-0.3, 0.3),
        scale_limit=(-0.1, 0.1),
        rotate_limit=(-180, 180),
        interpolation=0,
        border_mode=4,
    )

    def plot_augmentation(image_id, transform):
        plt.figure(figsize=(12, 12))
        img = cv2.imread(os.path.join(train_images_path, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("original")
        for i in range(2, 5):
            aug_img = transform(image=img)["image"]
            plt.subplot(2, 2, i)
            plt.imshow(aug_img)
            plt.axis("off")
            plt.title(f"aug {i-1}")
        plt.show()

    plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)




## === cell 15
img_width, img_height = 260, 260




## === cell 16
train_df = pd.read_csv(os.path.join(general_path, "train.csv"))
train_df["label"] = train_df["label"].astype(str)




## === cell 17
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255,
)

train_flow = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_path,
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

valid_flow = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_path,
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)




## === cell 18
x_batch, y_batch = next(train_flow)
print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")




## === cell 19
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(5, activation="softmax")(x)  # 5 classes
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 20
history = model.fit(
    train_flow,
    epochs=8,  # increased epochs for better learning
    validation_data=valid_flow,
    verbose=1,
    workers=4,
    use_multiprocessing=True,
    max_queue_size=10,
)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/361577286.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_flow,
      3     epochs=8,  # increased epochs for better learning
      4     validation_data=valid_flow,
      5     verbose=1,

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

## === cell 21
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_flow,
    epochs=8,
    validation_data=valid_flow,
    verbose=1,
    workers=4,
    use_multiprocessing=True,
    max_queue_size=10,
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3531816821.py in <cell line: 0>()
      8 )
      9 
---> 10 model.fit(
     11     train_flow,
     12     epochs=8,

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

## === cell 22
sample_sub = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
sample_sub.head()




## === cell 23
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_flow = test_datagen.flow_from_dataframe(
    dataframe=sample_sub,
    directory=os.path.join(general_path, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode=None,
    shuffle=False,
)




## === cell 24
test_predictions = model.predict(
    test_flow,
    steps=len(test_flow),
    verbose=1,
)
test_labels = np.argmax(test_predictions, axis=1)




## === cell 25
submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": test_labels})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
print(submission.head())

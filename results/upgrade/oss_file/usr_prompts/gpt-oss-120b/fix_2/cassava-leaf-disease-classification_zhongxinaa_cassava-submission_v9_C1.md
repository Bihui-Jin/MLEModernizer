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

3.12

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

0.8401329706860079

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, json, random, shutil, datetime
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"




## === cell 2
def build_model(input_shape=(224, 224, 3), num_classes=5):
    base = applications.efficientnet_v2.EfficientNetV2B0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False  # freeze for quick training
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 3
train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_df, val_df = tf.keras.utils.split_dataset(
    train_df.sample(frac=1, random_state=42), 0.2
)

preprocess = applications.efficientnet_v2.preprocess_input
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=os.path.join(WORK_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="raw",
    shuffle=True,
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=os.path.join(WORK_DIR, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="raw",
    shuffle=False,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1963617364.py in <cell line: 0>()
      4 train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
      5 # split into train/validation
----> 6 train_df, val_df = tf.keras.utils.split_dataset(
      7     train_df.sample(frac=1, random_state=42), 0.2
      8 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in split_dataset(dataset, left_size, right_size, shuffle, seed)
     53 
     54     if dataset_type_spec is None:
---> 55         raise TypeError(
     56             "The `dataset` argument must be either"
     57             "a `tf.data.Dataset`, a `torch.utils.data.Dataset`"

TypeError: The `dataset` argument must be eithera `tf.data.Dataset`, a `torch.utils.data.Dataset`object, or a list/tuple of arrays. Received: dataset=             image_id  label
13639  3939876859.jpg      0
18545   964482896.jpg      3
16511   584853244.jpg      4
8866    306133807.jpg      3
17986   851527428.jpg      3
...               ...    ...
11284  3494906160.jpg      3
11964  3623486148.jpg      3
5390    220364890.jpg      3
860    1147602192.jpg      3
15795   457931044.jpg      3

[18721 rows x 2 columns] of type <class 'pandas.core.frame.DataFrame'>

## === cell 4
model = build_model()
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, mode="max"
)

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, reduce_lr_cb],
    verbose=2,
)

model.load_weights("best_model.h5")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1131329464.py in <cell line: 0>()
     11 
     12 model.fit(
---> 13     train_generator,
     14     epochs=5,
     15     validation_data=val_generator,

NameError: name 'train_generator' is not defined

## === cell 5
test_dir = os.path.join(WORK_DIR, "test_images")
test_images = sorted(os.listdir(test_dir))

submission = pd.DataFrame(columns=["image_id", "label"])

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = applications.efficientnet_v2.preprocess_input(img_array)
    preds = model.predict(img_array, verbose=0)
    label = int(np.argmax(preds, axis=-1)[0])
    submission.loc[len(submission)] = [img_name, label]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2079829156.py in <cell line: 0>()
      9 for img_name in test_images:
     10     img_path = os.path.join(test_dir, img_name)
---> 11     img = load_img(img_path, target_size=(224, 224))
     12     img_array = img_to_array(img)
     13     img_array = np.expand_dims(img_array, axis=0)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/cassava-leaf-disease-classification/test_images/test_images'

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

3.13

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

0.8540344514959202

# 6. Current score

0.23169

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.23169) has done: 'The fix removes the failing TF‑Hub and TFSMLayer code, replaces it with a standard EfficientNetB0 classifier built directly on the training images, and adds a simple inference step that writes a correctly‑formatted `submission.csv`. This resolves the import and missing‑model errors while keeping the overall approach (image preprocessing, EfficientNet backbone, and a final classifier) unchanged, allowing the script to run end‑to‑end and produce a valid submission file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

train_df = pd.read_csv(TRAIN_CSV_PATH)
label_map = pd.read_json(LABEL_MAP_PATH, typ="series")
train_df["label_str"] = train_df["label"].astype(str)

train_df["img_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label_str"])

train_split, val_split = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label_enc"], random_state=42
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

batch_size = 32
img_size = (224, 224)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_split,
    x_col="img_path",
    y_col="label_enc",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_split,
    x_col="img_path",
    y_col="label_enc",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4227764147.py in <cell line: 0>()
     17 img_size = (224, 224)
     18 
---> 19 train_generator = train_datagen.flow_from_dataframe(
     20     dataframe=train_split,
     21     x_col="img_path",

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_dataframe(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
   1206             )
   1207 
-> 1208         return DataFrameIterator(
   1209             dataframe,
   1210             directory,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)
    749         self.dtype = dtype
    750         # check that inputs match the required class_mode
--> 751         self._check_params(df, x_col, y_col, weight_col, classes)
    752         if (
    753             validate_filenames

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in _check_params(self, df, x_col, y_col, weight_col, classes)
    839             types = (str, list, tuple)
    840             if not all(df[y_col].apply(lambda x: isinstance(x, types))):
--> 841                 raise TypeError(
    842                     'If class_mode="{}", y_col="{}" column '
    843                     "values must be type string, list or tuple.".format(

TypeError: If class_mode="categorical", y_col="label_enc" column values must be type string, list or tuple.

## === cell 2
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation="softmax")(x)  # 5 classes
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=2,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600707936.py in <cell line: 0>()
     20 # Train (use a small number of epochs for speed; adjust as needed)
     21 model.fit(
---> 22     train_generator,
     23     epochs=5,
     24     validation_data=val_generator,

NameError: name 'train_generator' is not defined

## === cell 3
test_df = pd.DataFrame(
    {
        "image_id": [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")],
    }
)
test_df["img_path"] = test_df["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="img_path",
    y_col=None,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)

pred_probs = model.predict(test_generator, verbose=0)
pred_indices = np.argmax(pred_probs, axis=1)
pred_label_str = le.inverse_transform(
    pred_indices
)  # maps back to original string labels

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_label_str}
)
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file created at: {submission_path}")
print(submission.head())

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

0.6128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Implemented path corrections and ensured the submission is saved to the proper working directory. The script now points to the Kaggle input folder (`/kaggle/input/...`) for training and test data, and writes `submission.csv` into `/kaggle/working`. With these fixes the data loading, model training, and prediction steps execute, producing a valid Kaggle‑format CSV.'
- What this solution (achieved 0.05531) has done: 'I move the protobuf environment setting to the very top, convert the label column to strings so `flow_from_dataframe` works with `class_mode="categorical"`, and raise the training epochs modestly to improve accuracy. These fixes resolve the import error, the datatype error, and allow the model to train, producing a valid CSV submission.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = GetPrototype
except Exception:
    pass

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

batch_size = 64
img_height = 256
img_width = 256
num_classes = 5

base_input = "/kaggle/input/cassava-leaf-disease-classification"
base_dir = "/kaggle/working"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2785799583.py in <cell line: 0>()
     21 # Enable mixed‑precision for faster computation on GPU/TPU (no change to model logic)
     22 if tf.config.list_physical_devices("GPU"):
---> 23     from tensorflow.keras.mixed_precision import experimental as mixed_precision
     24 
     25     policy = mixed_precision.Policy("mixed_float16")

ImportError: cannot import name 'experimental' from 'tensorflow.keras.mixed_precision' (/usr/local/lib/python3.11/dist-packages/keras/_tf_keras/keras/mixed_precision/__init__.py)

## === cell 1
train_dir = os.path.join(base_input, "train_images")
test_dir = os.path.join(base_input, "test_images")
train_csv_path = os.path.join(base_input, "train.csv")
sample_sub_path = os.path.join(base_input, "sample_submission.csv")

labels = pd.read_csv(train_csv_path)
labels["label"] = labels["label"].astype(str)

sample_sub = pd.read_csv(sample_sub_path)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_ds = train_datagen.flow_from_dataframe(
    dataframe=labels,
    directory=train_dir,
    x_col="image_id",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
    workers=4,
    use_multiprocessing=True,
)

val_ds = val_datagen.flow_from_dataframe(
    dataframe=labels,
    directory=train_dir,
    x_col="image_id",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
    workers=4,
    use_multiprocessing=True,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2750537980.py in <cell line: 0>()
----> 1 train_dir = os.path.join(base_input, "train_images")
      2 test_dir = os.path.join(base_input, "test_images")
      3 train_csv_path = os.path.join(base_input, "train.csv")
      4 sample_sub_path = os.path.join(base_input, "sample_submission.csv")
      5 

NameError: name 'base_input' is not defined

## === cell 2
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    weights="imagenet",
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2327192822.py in <cell line: 0>()
      1 base_model = tf.keras.applications.EfficientNetB0(
      2     include_top=False,
----> 3     input_shape=(img_height, img_width, 3),
      4     weights="imagenet",
      5 )

NameError: name 'img_height' is not defined

## === cell 3
epochs = 20  # increased epochs for better learning
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
    workers=4,
    use_multiprocessing=True,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3767466277.py in <cell line: 0>()
      1 epochs = 20  # increased epochs for better learning
----> 2 model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=epochs,

NameError: name 'model' is not defined

## === cell 4
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_ds = test_datagen.flow_from_dataframe(
    dataframe=sample_sub,
    directory=test_dir,
    x_col="image_id",
    y_col=None,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
    workers=4,
    use_multiprocessing=True,
)

pred_probs = model.predict(test_ds, verbose=1)
pred_labels = pred_probs.argmax(axis=1)

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = os.path.join(base_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1359301744.py in <cell line: 0>()
      2 
      3 test_ds = test_datagen.flow_from_dataframe(
----> 4     dataframe=sample_sub,
      5     directory=test_dir,
      6     x_col="image_id",

NameError: name 'sample_sub' is not defined

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

0.9079782411604714

# 6. Current score

0.08969

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented faster data loading and batch inference. Added deterministic seeds, enabled multiprocessing workers for model training, and replaced the per‑image prediction loop with a single batched `model.predict` call, dramatically reducing Python overhead while preserving the exact model architecture and training logic.'
- What this solution (achieved 0.51457) has done: 'I removed the problematic protobuf environment setting, fixed the use of TensorFlow’s `fit` and `predict` by dropping the unsupported `workers` and `use_multiprocessing` arguments, and aligned the label handling so the model trains directly on the numeric `label` field. This eliminates the generator‑related errors and ensures that predictions are written as the correct integer class IDs, producing a valid `submission.csv` file.'
- What this solution (achieved 0.09529) has done: 'I fixed the protobuf import error by setting the appropriate environment variable before loading TensorFlow, changed the data generators to use `class_mode='sparse'` so numeric labels are accepted, switched the loss to `sparse_categorical_crossentropy`, and increased the training epochs with a more forgiving early‑stopping patience to boost validation accuracy. The script now runs end‑to‑end and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.3793) has done: 'Implement faster data loading and larger batches while keeping the exact model and training logic.  
- Increase `batch_size` to 128 to halve the number of steps per epoch.  
- Enable multiprocessing in `model.fit` with several workers to parallelize image decoding and augmentation.  
- These changes speed up I/O and training without altering the architecture, loss, or evaluation semantics.'
- What this solution (achieved 0.18348) has done: 'I fix the TensorFlow `.fit` call by removing unsupported `workers`/`use_multiprocessing` arguments (eliminating the runtime error) and modestly adjust training settings to improve validation accuracy: use integer labels, increase the maximum epochs to 50 with a longer early‑stopping patience, and keep the EfficientNet‑B0 architecture unchanged. These changes keep the core model logic intact while allowing the network to train longer and reach a score closer to the target.'
- What this solution (achieved 0.21338) has done: 'I speed up the data pipeline and model training by (1) limiting TensorFlow thread usage to a reasonable number, (2) enabling multiprocessing for the ImageDataGenerator workers, and (3) adding those worker settings to `model.fit`. These changes keep the same model, epochs, and augmentations, so the predictions remain identical while reducing CPU‑bound image‑loading overhead.'
- What this solution (achieved 0.08969) has done: 'The fix sets the protobuf implementation to the pure‑Python version (avoiding the C‑extension import error) and ensures all required imports are available after TensorFlow loads successfully. No other logic is changed, preserving the model architecture, training routine, and submission format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)
train_csv["label"] = train_csv["label"].astype(int)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

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

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="label",
    target_size=(224, 224),
    batch_size=128,
    class_mode="sparse",  # integer labels
    shuffle=True,
    seed=42,
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="path",
    y_col="label",
    target_size=(224, 224),
    batch_size=128,
    class_mode="sparse",
    shuffle=False,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1424799688.py in <cell line: 0>()
     29 valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
     30 
---> 31 train_generator = train_datagen.flow_from_dataframe(
     32     dataframe=train_df,
     33     x_col="path",

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
    817         if self.class_mode in {"binary", "sparse"}:
    818             if not all(df[y_col].apply(lambda x: isinstance(x, str))):
--> 819                 raise TypeError(
    820                     'If class_mode="{}", y_col="{}" column '
    821                     "values must be strings.".format(self.class_mode, y_col)

TypeError: If class_mode="sparse", y_col="label" column values must be strings.

## === cell 2
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation="softmax")(x)  # 5 classes (0‑4)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
)
lr_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
)




## === cell 3
model.fit(
    train_generator,
    epochs=50,
    validation_data=valid_generator,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4232503357.py in <cell line: 0>()
      1 model.fit(
----> 2     train_generator,
      3     epochs=50,
      4     validation_data=valid_generator,
      5     callbacks=[early_stop, lr_reduce],

NameError: name 'train_generator' is not defined

## === cell 4
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]

test_df = pd.DataFrame(
    {"path": [os.path.join(test_dir, fname) for fname in test_filenames]}
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col=None,
    target_size=(224, 224),
    batch_size=128,
    class_mode=None,
    shuffle=False,
)

probs = model.predict(test_generator, verbose=0)
pred_labels = np.argmax(probs, axis=1).astype(int)  # numeric class IDs (0‑4)

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())

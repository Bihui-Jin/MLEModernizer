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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
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

0.8851616802659413

# 6. Current score

0.53886

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.70441) has done: 'I fix two critical problems: 1) the TensorFlow import crashes because of a protobuf incompatibility, so I set the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable **before** importing TensorFlow. 2) the submission dataframe was built from two mismatched arrays; I now derive the list of actually‑processed test filenames from the generator and align it with the predictions, truncating if needed. These minimal changes unblock execution and guarantee a correctly‑sized `submission.csv` while preserving the original model and training logic.'
- What this solution (achieved 0.09118) has done: 'The changes speed up data loading and move image augmentation onto the GPU by using a Keras preprocessing layer, which eliminates the costly CPU‑side augmentations from `ImageDataGenerator`. The training loops now use multiple worker processes for the generators, further reducing I/O overhead. These adjustments keep the model architecture, loss, optimizer, and training epochs unchanged, preserving result accuracy while fitting comfortably within the 600‑second limit.'
- What this solution (achieved 0.53886) has done: 'We eliminate the on‑the‑fly TensorFlow install (which dominates start‑up time) and accelerate data loading by using multiple worker processes for the `ImageDataGenerator` iterators and for `model.fit`. These changes keep the model architecture, training epochs, and loss unchanged while substantially lowering I/O overhead, allowing the script to finish well within the 600 s limit.'

# 9. Code solution

## === cell 0
import os, sys, json, warnings, numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras import Sequential  # for data‑augmentation pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    map_classes = {int(k): v for k, v in json.load(f).items()}
print("Label map:", map_classes)




## === cell 2
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["label"] = df_train["label"].astype(int)
df_train["image_path"] = df_train["image_id"].apply(
    lambda x: os.path.join(BASE_DIR, "train_images", x)
)
print(df_train.head())




## === cell 3
val_frac = 0.1
df_val = df_train.sample(frac=val_frac, random_state=42)
df_train = df_train.drop(df_val.index)




## === cell 4
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col="image_path",
    y_col="label",
    target_size=(128, 128),
    batch_size=32,
    class_mode="raw",  # integer labels as‑is
    shuffle=True,
    seed=42,
    workers=4,
    use_multiprocessing=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col="image_path",
    y_col="label",
    target_size=(128, 128),
    batch_size=32,
    class_mode="raw",
    shuffle=False,
    workers=4,
    use_multiprocessing=True,
)




## === cell 5
data_augmentation = Sequential(
    [
        layers.RandomRotation(0.1111),  # ~20 degrees
        layers.RandomWidth(0.1),
        layers.RandomHeight(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip("horizontal"),
    ]
)

base_model = applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze backbone initially

model = models.Sequential(
    [
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 6
EPOCHS_FROZEN = 5
history_frozen = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FROZEN,
    verbose=2,
    workers=4,
    use_multiprocessing=True,
)

base_model.trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_UNFROZEN = 7
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_UNFROZEN,
    verbose=2,
    workers=4,
    use_multiprocessing=True,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/502333781.py in <cell line: 0>()
      1 EPOCHS_FROZEN = 5
----> 2 history_frozen = model.fit(
      3     train_generator,
      4     validation_data=val_generator,
      5     epochs=EPOCHS_FROZEN,

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

## === cell 7
TEST_DIR = os.path.join(BASE_DIR, "test_images")
test_images = sorted(os.listdir(TEST_DIR))
df_test = pd.DataFrame(
    {
        "image_id": test_images,
        "image_path": [os.path.join(TEST_DIR, f) for f in test_images],
    }
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col="image_path",
    y_col=None,
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,
    shuffle=False,
    workers=4,
    use_multiprocessing=True,
)




## === cell 8
pred_probs = model.predict(test_generator, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

processed_filenames = test_generator.filenames
processed_image_ids = [os.path.basename(fp) for fp in processed_filenames]

min_len = min(len(processed_image_ids), len(pred_labels))
submission = pd.DataFrame(
    {
        "image_id": processed_image_ids[:min_len],
        "label": pred_labels[:min_len].astype(int),
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")

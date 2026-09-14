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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.8425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14083) has done: 'I fix the import conflict causing the TensorFlow error, replace the deprecated fit_generator call with model.fit, correct the image rescaling factor and generator class mode, adjust the test directory path, and rewrite the prediction loop so it builds a proper DataFrame and writes a single valid sampleSubmission.csv with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

candidates = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
    "/kaggle/working/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
]


def find_valid_base(paths):
    for p in paths:
        if os.path.isdir(p):
            train_dir = os.path.join(p, "train")
            if os.path.isdir(train_dir) and any(
                f.lower().endswith(".jpg") for f in os.listdir(train_dir)
            ):
                return p
    raise FileNotFoundError(
        "Could not locate a valid aerial-cactus-identification data folder."
    )


BASE_PATH = find_valid_base(candidates)

TRAINING_DIR = os.path.join(BASE_PATH, "train")
TRAINING_LABEL_PATH = os.path.join(BASE_PATH, "train.csv")
TESTING_DIR = os.path.join(BASE_PATH, "test")
if not os.path.isdir(TESTING_DIR):
    raise FileNotFoundError("Could not locate the test folder for the dataset.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels_df = pd.read_csv(TRAINING_LABEL_PATH)
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df["has_cactus"],
    random_state=42,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2280356335.py in <cell line: 0>()
----> 1 labels_df = pd.read_csv(TRAINING_LABEL_PATH)
      2 train_df, val_df = train_test_split(
      3     labels_df,
      4     test_size=0.2,
      5     stratify=labels_df["has_cactus"],

NameError: name 'TRAINING_LABEL_PATH' is not defined

## === cell 2
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAINING_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw",
    shuffle=True,
    validate_filenames=False,
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAINING_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw",
    shuffle=False,
    validate_filenames=False,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3501827624.py in <cell line: 0>()
     12 
     13 train_generator = train_datagen.flow_from_dataframe(
---> 14     dataframe=train_df,
     15     directory=TRAINING_DIR,
     16     x_col="id",

NameError: name 'train_df' is not defined

## === cell 3
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
base_model.trainable = False  # freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)



## === cell 4
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1221525741.py in <cell line: 0>()
      1 # Train the model (slightly more epochs to improve AUC while keeping the same architecture)
      2 history = model.fit(
----> 3     train_generator,
      4     epochs=15,
      5     validation_data=validation_generator,

NameError: name 'train_generator' is not defined

## === cell 5
test_files = [f for f in os.listdir(TESTING_DIR) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame({"id": test_files})

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TESTING_DIR,
    x_col="id",
    y_col=None,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False,
    validate_filenames=False,
)

preds = model.predict(test_generator, verbose=1).ravel()
submission = pd.DataFrame({"id": test_df["id"], "has_cactus": preds})
submission_path = os.path.join(".", "sample_submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894324480.py in <cell line: 0>()
----> 1 test_files = [f for f in os.listdir(TESTING_DIR) if f.lower().endswith(".jpg")]
      2 test_df = pd.DataFrame({"id": test_files})
      3 
      4 test_datagen = ImageDataGenerator(rescale=1.0 / 255)
      5 test_generator = test_datagen.flow_from_dataframe(

NameError: name 'TESTING_DIR' is not defined

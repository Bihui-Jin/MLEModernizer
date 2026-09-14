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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

22.28318

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    concatenate,
    Dropout,
    MaxPooling2D,
    Conv2D,
    Flatten,
    Input,
)
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

keras.utils.set_random_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
comp_path = "../input/petfinder-pawpularity-score"
train_csv_path = "../input/petfinder-pawpularity-score/train.csv"
test_csv_path = "../input/petfinder-pawpularity-score/test.csv"



## === cell 2
train_meta = pd.read_csv(train_csv_path)
train_meta.head()




## === cell 3
def train_img_path(id_):
    return os.path.join(comp_path, "train", id_ + ".jpg")


train_meta["Id2"] = train_meta["Id"].apply(train_img_path)
train_meta.head()



## === cell 4
META_COLS = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


def gen_flow_for_two_inputs(datagen, batch, x_train, shuffle=True):
    """
    Yields:
        ((batch_images, batch_meta_features), batch_targets)
    Keras 2.18+ requires generators to yield tuples (not lists).
    """
    x_train_2 = x_train.set_index("Id")
    flow = datagen.flow_from_dataframe(
        x_train,
        batch_size=batch,
        shuffle=shuffle,
        x_col="Id2",
        y_col="Id",  # use Ids as "labels" to recover rows
        class_mode="raw",
        target_size=(224, 224),
    )

    while True:
        batch_image, batch_index = next(flow)
        batch_meta = x_train_2.loc[batch_index, META_COLS].values.astype(np.float32)
        batch_y = x_train_2.loc[batch_index, "Pawpularity"].values.astype(np.float32)
        yield (batch_image, batch_meta), batch_y




## === cell 5
input1 = Input(shape=(224, 224, 3))
input2 = Input(shape=(12,))

x1 = Conv2D(16, kernel_size=3, activation="relu")(input1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(32, kernel_size=3, activation="relu")(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(64, kernel_size=3, activation="relu")(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Dropout(0.5)(x1)
x1 = Flatten()(x1)
x1 = Model(inputs=input1, outputs=x1)

x2 = Model(inputs=input2, outputs=input2)

combined = concatenate([x1.output, x2.output])

z = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(combined)
z = Dropout(0.25)(z)
z = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.001))(z)
z = Dense(1)(z)

model = Model(inputs=[x1.input, x2.input], outputs=z)
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.summary()



## === cell 6
train_datagen = image.ImageDataGenerator(rescale=1 / 255)

EPOCH = 10
BATCH = 32

train_output_signature = (
    (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 12), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: gen_flow_for_two_inputs(train_datagen, BATCH, train_meta, shuffle=True),
    output_signature=train_output_signature,
)

log = model.fit(
    train_ds,
    steps_per_epoch=2,
    epochs=EPOCH,
    verbose=1,
)



## === cell 7
plt.plot(log.history["loss"], label="loss")
plt.legend(frameon=False)
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()



## === cell 8
test_datagen = image.ImageDataGenerator(rescale=1 / 255)
test_meta = pd.read_csv(test_csv_path)


def test_img_path(id_):
    return os.path.join(comp_path, "test", id_ + ".jpg")


test_meta["Id2"] = test_meta["Id"].apply(test_img_path)


def gen_flow_for_two_inputs_2(datagen, batch, x_test, shuffle=False):
    """
    Yields:
        (batch_images, batch_meta_features)
    Keras 2.18+ requires tuples.
    """
    x_test_2 = x_test.set_index("Id")
    flow = datagen.flow_from_dataframe(
        x_test,
        batch_size=batch,
        shuffle=shuffle,
        x_col="Id2",
        y_col="Id",
        class_mode="raw",
        target_size=(224, 224),
    )

    while True:
        batch_image, batch_index = next(flow)
        batch_meta = x_test_2.loc[batch_index, META_COLS].values.astype(np.float32)
        yield (batch_image, batch_meta)


test_output_signature = (
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 12), dtype=tf.float32),
)

test_ds = tf.data.Dataset.from_generator(
    lambda: gen_flow_for_two_inputs_2(test_datagen, BATCH, test_meta, shuffle=False),
    output_signature=test_output_signature,
)

steps = int(np.ceil(test_meta.shape[0] / BATCH))
pred = model.predict(
    test_ds,
    steps=steps,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1253457799.py in <cell line: 0>()
     44 
     45 steps = int(np.ceil(test_meta.shape[0] / BATCH))
---> 46 pred = model.predict(
     47     test_ds,
     48     steps=steps,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    158     inputs = tree.flatten(inputs)
    159     if len(inputs) != len(input_spec):
--> 160         raise ValueError(
    161             f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
    162             f" but it received {len(inputs)} input tensors. "

ValueError: Layer "functional_2" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 224, 224, 3) dtype=float32>]

## === cell 9
test_meta["Pawpularity"] = np.clip(pred.reshape(-1)[: len(test_meta)], 0, 100)

submission_df = test_meta[["Id", "Pawpularity"]].copy()
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1341203078.py in <cell line: 0>()
      1 # Ensure correct shape and valid submission range (Pawpularity is 0..100).
----> 2 test_meta["Pawpularity"] = np.clip(pred.reshape(-1)[: len(test_meta)], 0, 100)
      3 
      4 submission_df = test_meta[["Id", "Pawpularity"]].copy()
      5 submission_df.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined

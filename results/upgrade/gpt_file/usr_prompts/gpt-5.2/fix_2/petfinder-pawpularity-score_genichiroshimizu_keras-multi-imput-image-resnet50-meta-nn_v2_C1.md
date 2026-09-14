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
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

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
def gen_flow_for_two_inputs(datagen, batch, x_train, shuffle=True):
    """
    Yields:
        ([batch_images, batch_meta_features], batch_targets)
    """
    x_train_2 = x_train.set_index("Id")
    flow = datagen.flow_from_dataframe(
        x_train,
        batch_size=batch,
        shuffle=shuffle,
        x_col="Id2",
        y_col="Id",  # we just use Ids as "labels" to recover rows
        class_mode="raw",
        target_size=(224, 224),
    )
    meta_cols = [
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

    while True:
        batch_image, batch_index = next(flow)
        batch_meta = x_train_2.loc[batch_index, meta_cols].values
        batch_y = x_train_2.loc[batch_index, "Pawpularity"].values
        yield [batch_image, batch_meta], batch_y




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

log = model.fit(
    gen_flow_for_two_inputs(train_datagen, BATCH, train_meta, shuffle=True),
    steps_per_epoch=2,
    epochs=EPOCH,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3780969888.py in <cell line: 0>()
      6 # FIX: Keras 2.18 removed fit_generator; use fit with the same generator
      7 # NOTE: steps_per_epoch kept as in original code to preserve core behavior.
----> 8 log = model.fit(
      9     gen_flow_for_two_inputs(train_datagen, BATCH, train_meta, shuffle=True),
     10     steps_per_epoch=2,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py in _from_generator(generator, output_types, output_shapes, args, output_signature, name)
    122     for spec in nest.flatten(output_signature):
    123       if not isinstance(spec, type_spec.TypeSpec):
--> 124         raise TypeError(f"`output_signature` must contain objects that are "
    125                         f"subclass of `tf.TypeSpec` but found {type(spec)} "
    126                         f"which is not.")

TypeError: `output_signature` must contain objects that are subclass of `tf.TypeSpec` but found <class 'list'> which is not.

## === cell 7
plt.plot(log.history["loss"], label="loss")
plt.legend(frameon=False)
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3499836895.py in <cell line: 0>()
----> 1 plt.plot(log.history["loss"], label="loss")
      2 plt.legend(frameon=False)
      3 plt.xlabel("epochs")
      4 plt.ylabel("mse")
      5 plt.show()

NameError: name 'log' is not defined

## === cell 8
test_datagen = image.ImageDataGenerator(rescale=1 / 255)

test_meta = pd.read_csv(test_csv_path)


def test_img_path(id_):
    return os.path.join(comp_path, "test", id_ + ".jpg")


test_meta["Id2"] = test_meta["Id"].apply(test_img_path)


def gen_flow_for_two_inputs_2(datagen, batch, x_test, shuffle=False):
    """
    Yields:
        [batch_images, batch_meta_features]
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
    meta_cols = [
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

    while True:
        batch_image, batch_index = next(flow)
        batch_meta = x_test_2.loc[batch_index, meta_cols].values
        yield [batch_image, batch_meta]


steps = int(np.ceil(test_meta.shape[0] / BATCH))
pred = model.predict(
    gen_flow_for_two_inputs_2(test_datagen, BATCH, test_meta, shuffle=False),
    steps=steps,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/425626424.py in <cell line: 0>()
     49 # FIX: Keras 2.18 removed predict_generator; use predict with same generator
     50 steps = int(np.ceil(test_meta.shape[0] / BATCH))
---> 51 pred = model.predict(
     52     gen_flow_for_two_inputs_2(test_datagen, BATCH, test_meta, shuffle=False),
     53     steps=steps,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py in __init__(self, generator)
     15         self._output_signature = None
     16         if not isinstance(first_batches[0], tuple):
---> 17             raise ValueError(
     18                 "When passing a Python generator to a Keras model, "
     19                 "the generator must return a tuple, either "

ValueError: When passing a Python generator to a Keras model, the generator must return a tuple, either (input,) or (inputs, targets) or (inputs, targets, sample_weights). Received: [array([[[[0.07450981, 0.03921569, 0.00392157],
         [0.07843138, 0.03921569, 0.        ],
         [0.1137255 , 0.05882353, 0.00784314],
         ...,
         [0.10588236, 0.32156864, 0.9215687 ],
         [0.01960784, 0.24705884, 0.8117648 ],
         [0.16078432, 0.35686275, 0.8313726 ]],

        [[0.05882353, 0.03137255, 0.        ],
         [0.07450981, 0.03921569, 0.00392157],
         [0.08627451, 0.04705883, 0.00784314],
         ...,
         [0.06666667, 0.28627452, 0.8941177 ],
         [0.04705883, 0.25490198, 0.7568628 ],
         [0.3019608 , 0.49411768, 0.9058824 ]],

        [[0.02352941, 0.01960784, 0.00392157],
         [0.03137255, 0.01960784, 0.        ],
         [0.05490196, 0.02745098, 0.        ],
         ...,
         [0.0627451 , 0.29411766, 0.8980393 ],
         [0.07058824, 0.25490198, 0.7725491 ],
         [0.12941177, 0.32156864, 0.78823537]],

        ...,

        [[0.82745105, 0.7019608 , 0.6509804 ],
         [0.81568635, 0.6901961 , 0.6392157 ],
         [0.8078432 , 0.68235296, 0.6313726 ],
         ...,
         [0.89019614, 0.69411767, 0.59607846],
         [0.8235295 , 0.63529414, 0.54509807],
         [0.854902  , 0.6666667 , 0.5764706 ]],

        [[0.8235295 , 0.69803923, 0.64705884],
         [0.7803922 , 0.654902  , 0.6039216 ],
         [0.8000001 , 0.6745098 , 0.62352943],
         ...,
         [0.90196085, 0.7058824 , 0.60784316],
         [0.8431373 , 0.654902  , 0.5647059 ],
         [0.8235295 , 0.63529414, 0.54509807]],

        [[0.8196079 , 0.69411767, 0.6431373 ],
         [0.82745105, 0.7019608 , 0.6509804 ],
         [0.79215693, 0.6666667 , 0.6156863 ],
         ...,
         [0.8313726 , 0.63529414, 0.5372549 ],
         [0.8431373 , 0.654902  , 0.5647059 ],
         [0.8235295 , 0.63529414, 0.54509807]]],


       [[[0.21176472, 0.09019608, 0.00784314],
         [0.227451  , 0.10196079, 0.01960784],
         [0.2392157 , 0.09411766, 0.01960784],
         ...,
         [0.1254902 , 0.0509804 , 0.03529412],
         [0.2392157 , 0.17254902, 0.10196079],
         [0.48235297, 0.4156863 , 0.3372549 ]],

        [[0.21960786, 0.09803922, 0.01568628],
         [0.227451  , 0.10196079, 0.01960784],
         [0.24705884, 0.10196079, 0.02745098],
         ...,
         [0.34901962, 0.27450982, 0.2509804 ],
         [0.4901961 , 0.42352945, 0.36078432],
         [0.53333336, 0.46274513, 0.4156863 ]],

        [[0.21960786, 0.09803922, 0.01568628],
         [0.227451  , 0.10196079, 0.01960784],
         [0.24705884, 0.10196079, 0.02745098],
         ...,
         [0.5529412 , 0.4784314 , 0.45098042],
         [0.5294118 , 0.45882356, 0.4039216 ],
         [0.53333336, 0.45882356, 0.43529415]],

        ...,

        [[0.05882353, 0.04313726, 0.03137255],
         [0.05882353, 0.04313726, 0.03137255],
         [0.05882353, 0.04313726, 0.03137255],
         ...,
         [0.25882354, 0.14901961, 0.05490196],
         [0.25490198, 0.14509805, 0.0627451 ],
         [0.25490198, 0.14117648, 0.07058824]],

        [[0.05882353, 0.04313726, 0.03137255],
         [0.05882353, 0.04313726, 0.03137255],
         [0.05882353, 0.04313726, 0.03137255],
         ...,
         [0.24705884, 0.14117648, 0.05882353],
         [0.23529413, 0.12941177, 0.05490196],
         [0.227451  , 0.11764707, 0.0627451 ]],

        [[0.05882353, 0.04313726, 0.03137255],
         [0.05882353, 0.04313726, 0.03137255],
         [0.0509804 , 0.03529412, 0.02352941],
         ...,
         [0.21960786, 0.13333334, 0.0509804 ],
         [0.20392159, 0.1137255 , 0.0509804 ],
         [0.20000002, 0.10980393, 0.05490196]]],


       [[[0.01176471, 0.00392157, 0.00784314],
         [0.01568628, 0.00784314, 0.01176471],
         [0.02352941, 0.01568628, 0.01960784],
         ...,
         [0.53333336, 0.4784314 , 0.427451  ],
         [0.5372549 , 0.47450984, 0.42352945],
         [0.53333336, 0.47058827, 0.41960788]],

        [[0.02745098, 0.01960784, 0.02352941],
         [0.03137255, 0.02352941, 0.02745098],
         [0.03137255, 0.02352941, 0.02745098],
         ...,
         [0.5137255 , 0.4784314 , 0.41960788],
         [0.52156866, 0.47450984, 0.41960788],
         [0.52156866, 0.47450984, 0.41960788]],

        [[0.02745098, 0.01960784, 0.02352941],
         [0.02352941, 0.01568628, 0.01960784],
         [0.01960784, 0.01176471, 0.01568628],
         ...,
         [0.50980395, 0.48627454, 0.42352945],
         [0.5137255 , 0.5019608 , 0.43529415],
         [0.5137255 , 0.5019608 , 0.43529415]],

        ...,

        [[0.37647063, 0.36862746, 0.3803922 ],
         [0.4156863 , 0.40784317, 0.41960788],
         [0.43529415, 0.427451  , 0.43921572],
         ...,
         [0.36862746, 0.3254902 , 0.31764707],
         [0.454902  , 0.41176474, 0.4039216 ],
         [0.37647063, 0.33333334, 0.3254902 ]],

        [[0.44705886, 0.43921572, 0.45098042],
         [0.37647063, 0.36862746, 0.3803922 ],
         [0.30980393, 0.3019608 , 0.3137255 ],
         ...,
         [0.48627454, 0.4431373 , 0.43529415],
         [0.5568628 , 0.5137255 , 0.5058824 ],
         [0.38431376, 0.34117648, 0.33333334]],

        [[0.43529415, 0.427451  , 0.43921572],
         [0.36862746, 0.36078432, 0.37254903],
         [0.3921569 , 0.38431376, 0.39607847],
         ...,
         [0.50980395, 0.4666667 , 0.45882356],
         [0.48627454, 0.4431373 , 0.43529415],
         [0.4431373 , 0.40000004, 0.3921569 ]]],


       ...,


       [[[0.7019608 , 0.6156863 , 0.41960788],
         [0.6862745 , 0.6       , 0.4039216 ],
         [0.6862745 , 0.6       , 0.4039216 ],
         ...,
         [0.7137255 , 0.6666667 , 0.5176471 ],
         [0.74509805, 0.69803923, 0.54901963],
         [0.72156864, 0.6745098 , 0.5254902 ]],

        [[0.6862745 , 0.6       , 0.4039216 ],
         [0.7058824 , 0.61960787, 0.42352945],
         [0.6862745 , 0.6       , 0.4039216 ],
         ...,
         [0.7372549 , 0.6901961 , 0.5411765 ],
         [0.7176471 , 0.67058825, 0.52156866],
         [0.73333335, 0.6862745 , 0.5372549 ]],

        [[0.6745098 , 0.5882353 , 0.3921569 ],
         [0.7019608 , 0.6156863 , 0.41960788],
         [0.7019608 , 0.6156863 , 0.41960788],
         ...,
         [0.7294118 , 0.68235296, 0.53333336],
         [0.7254902 , 0.6784314 , 0.5294118 ],
         [0.7294118 , 0.68235296, 0.53333336]],

        ...,

        [[0.6784314 , 0.61960787, 0.45098042],
         [0.70980394, 0.6509804 , 0.48235297],
         [0.69803923, 0.6392157 , 0.47058827],
         ...,
         [0.54901963, 0.49411768, 0.31764707],
         [0.54509807, 0.4901961 , 0.3137255 ],
         [0.57254905, 0.5176471 , 0.34117648]],

        [[0.6862745 , 0.627451  , 0.45882356],
         [0.6509804 , 0.5921569 , 0.42352945],
         [0.6862745 , 0.627451  , 0.45882356],
         ...,
         [0.5411765 , 0.48627454, 0.30980393],
         [0.56078434, 0.5058824 , 0.32941177],
         [0.5137255 , 0.45882356, 0.28235295]],

        [[0.65882355, 0.6       , 0.43137258],
         [0.6509804 , 0.5921569 , 0.42352945],
         [0.62352943, 0.5647059 , 0.39607847],
         ...,
         [0.5372549 , 0.48235297, 0.30588236],
         [0.5921569 , 0.5372549 , 0.36078432],
         [0.5529412 , 0.49803925, 0.32156864]]],


       [[[0.9843138 , 0.98823535, 1.        ],
         [0.9960785 , 1.        , 1.        ],
         [0.9960785 , 1.        , 1.        ],
         ...,
         [0.98823535, 1.        , 1.        ],
         [1.        , 0.9960785 , 1.        ],
         [0.9803922 , 1.        , 0.98823535]],

        [[0.98823535, 1.        , 1.        ],
         [0.654902  , 0.70980394, 0.7137255 ],
         [0.6       , 0.68235296, 0.6862745 ],
         ...,
         [0.7803922 , 0.8196079 , 0.8235295 ],
         [0.97647065, 0.97647065, 0.97647065],
         [0.9960785 , 1.        , 0.9921569 ]],

        [[1.        , 0.9960785 , 1.        ],
         [0.67058825, 0.7294118 , 0.7411765 ],
         [0.60784316, 0.7294118 , 0.7490196 ],
         ...,
         [0.91372555, 0.9686275 , 0.9686275 ],
         [1.        , 1.        , 1.        ],
         [1.        , 0.9960785 , 0.9921569 ]],

        ...,

        [[0.9921569 , 0.98823535, 0.9725491 ],
         [0.5529412 , 0.5882353 , 0.62352943],
         [0.56078434, 0.6431373 , 0.7176471 ],
         ...,
         [0.1764706 , 0.25882354, 0.09019608],
         [0.23529413, 0.2784314 , 0.16078432],
         [0.9803922 , 0.9803922 , 0.9333334 ]],

        [[0.9960785 , 1.        , 0.9921569 ],
         [0.5764706 , 0.64705884, 0.64705884],
         [0.43529415, 0.5568628 , 0.5647059 ],
         ...,
         [0.5019608 , 0.5529412 , 0.37254903],
         [0.7137255 , 0.74509805, 0.654902  ],
         [1.        , 0.9960785 , 1.        ]],

        [[0.97647065, 0.9803922 , 0.98823535],
         [1.        , 1.        , 0.9921569 ],
         [1.        , 0.9960785 , 0.9843138 ],
         ...,
         [1.        , 0.9960785 , 1.        ],
         [1.        , 1.        , 1.        ],
         [1.        , 1.        , 0.9843138 ]]],


       [[[0.82745105, 0.86666673, 0.9058824 ],
         [0.7725491 , 0.8000001 , 0.86274517],
         [0.8431373 , 0.86274517, 0.87843144],
         ...,
         [0.3529412 , 0.427451  , 0.48627454],
         [0.36862746, 0.45098042, 0.5176471 ],
         [0.3803922 , 0.46274513, 0.5294118 ]],

        [[0.76470596, 0.80392164, 0.8431373 ],
         [0.7411765 , 0.7725491 , 0.81568635],
         [0.9176471 , 0.93725497, 0.9490197 ],
         ...,
         [0.36078432, 0.43529415, 0.49411768],
         [0.37254903, 0.454902  , 0.52156866],
         [0.40784317, 0.4901961 , 0.5568628 ]],

        [[0.7294118 , 0.7686275 , 0.8078432 ],
         [0.90196085, 0.94117653, 0.9490197 ],
         [0.9843138 , 1.        , 1.        ],
         ...,
         [0.3803922 , 0.454902  , 0.5137255 ],
         [0.40784317, 0.4901961 , 0.5568628 ],
         [0.43137258, 0.5137255 , 0.5803922 ]],

        ...,

        [[0.15686275, 0.15294118, 0.14509805],
         [0.15686275, 0.15294118, 0.14509805],
         [0.14117648, 0.13725491, 0.12941177],
         ...,
         [0.17254902, 0.21960786, 0.26666668],
         [0.227451  , 0.28235295, 0.3254902 ],
         [0.25882354, 0.3137255 , 0.35686275]],

        [[0.15294118, 0.14901961, 0.14117648],
         [0.15294118, 0.14901961, 0.14117648],
         [0.13333334, 0.12941177, 0.12156864],
         ...,
         [0.18431373, 0.23137257, 0.2784314 ],
         [0.19215688, 0.24705884, 0.2901961 ],
         [0.27450982, 0.32941177, 0.37254903]],

        [[0.15294118, 0.14901961, 0.14117648],
         [0.13725491, 0.13333334, 0.1254902 ],
         [0.13725491, 0.13333334, 0.1254902 ],
         ...,
         [0.16470589, 0.21176472, 0.25882354],
         [0.16862746, 0.22352943, 0.26666668],
         [0.26666668, 0.32156864, 0.3647059 ]]]], dtype=float32), array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
       [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])]

## === cell 9
test_meta["Pawpularity"] = pred.reshape(-1)

submission_df = test_meta[["Id", "Pawpularity"]].copy()
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3038084062.py in <cell line: 0>()
      1 # Ensure correct shape for assignment (pred is (N,1))
----> 2 test_meta["Pawpularity"] = pred.reshape(-1)
      3 
      4 submission_df = test_meta[["Id", "Pawpularity"]].copy()
      5 submission_df.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined

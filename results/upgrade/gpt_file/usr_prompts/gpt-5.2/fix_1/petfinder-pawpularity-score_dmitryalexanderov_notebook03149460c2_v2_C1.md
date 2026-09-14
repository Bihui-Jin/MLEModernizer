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

No external packages required in the script and installed.

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

18.86876823169344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, roc_curve, auc, mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (8, 8)
sns.set_style('whitegrid')

import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, KFold


import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
image_size = 224
batch_size = 128

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()


test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
test["file_path"] = test["Id"].apply(lambda identifier: "../input/petfinder-pawpularity-score/test/" + identifier + ".jpg")
test.head()

## === cell 4
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

from tensorflow.keras.applications import EfficientNetB0

## === cell 5
tabular_columns = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']

def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.sqrt(tf.reduce_mean((y_true -  y_pred) ** 2))

def get_tabular_prediciton_model(inputs):
    width = 32
    depth = 3
    activation = "relu"
    kernel_regularizer = keras.regularizers.l2()
    x = keras.layers.Dense(
            width, 
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )(inputs)
    for i in range(depth):
        if i == 0:
            x = inputs
        x = keras.layers.Dense(
            width, 
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )(x)
        if (i + 1) % 3 == 0:
            x = keras.layers.Concatenate()([x, inputs])
    return x

def get_model():
    image_inputs = layers.Input(shape=(image_size, image_size, 3))
    image_x = img_augmentation(image_inputs)
    model = EfficientNetB0(include_top=False, 
                           input_tensor=image_x, 
                           weights="../input/b0-weights/efficientnetb0_notop.h5", 
                           input_shape = (image_size, image_size, 3))
    model.trainable = False
    image_x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    top_dropout_rate = 0.2
    image_x = layers.Dropout(top_dropout_rate, name="top_dropout")(image_x)
    
    tabular_inputs = tf.keras.Input(len(tabular_columns))
    tabular_x = get_tabular_prediciton_model(tabular_inputs)
    
    x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    outputs = layers.Dense(1)(x)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = tf.keras.Model(inputs=[image_inputs, tabular_inputs], outputs=[outputs], name="EfficientNet")
    model.compile(
        optimizer=optimizer, loss=rmse, metrics=["mae", "mape"]
    )
    return model


def preprocess_test_data(image_url, tabular):
    print(image_url)
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (image_size, image_size))
    return (image, tabular), 0


model_up = get_model()
model_up.load_weights("../input/b0-weights/pet_trained_weights_v6.h5")

ds_try = tf.data.Dataset.from_tensor_slices((test["file_path"], test[tabular_columns])).map(preprocess_test_data).batch(batch_size).cache().prefetch(2)
res_0 = model_up.predict(ds_try).reshape(-1)
res_0

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/244697554.py in <cell line: 0>()
     65 
     66 
---> 67 model_up = get_model()
     68 model_up.load_weights("../input/b0-weights/pet_trained_weights_v6.h5")
     69 # model_up = tf.keras.models.load_model('../input/b0-weights/pet_model_v3.h5', custom_objects = {"rmse": rmse})

/tmp/ipykernel_11/244697554.py in get_model()
     30     image_inputs = layers.Input(shape=(image_size, image_size, 3))
     31     image_x = img_augmentation(image_inputs)
---> 32     model = EfficientNetB0(include_top=False, 
     33                            input_tensor=image_x,
     34                            weights="../input/b0-weights/efficientnetb0_notop.h5",

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    273 
    274     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
--> 275         raise ValueError(
    276             "The `weights` argument should be either "
    277             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.

## === cell 6
train_prev = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test_prev = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
train_x = train_prev.drop(['Id'],axis=1)
test_x = test_prev.drop(['Id'],axis=1)
model_prev = RandomForestRegressor(criterion='mse', max_depth = 4, max_features = 6, n_estimators = 1000)
model_prev.fit(train_x.drop(['Pawpularity'],axis=1), train_x.loc[:,['Pawpularity']].values.ravel())
res_1 = model_prev.predict(test_x)

w = 0.96
predictions = w*res_0 + (1.0-w)*res_1

test['Pawpularity'] = predictions
test[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)
display(test[['Id', 'Pawpularity']].head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/4084027098.py in <cell line: 0>()
      4 test_x = test_prev.drop(['Id'],axis=1)
      5 model_prev = RandomForestRegressor(criterion='mse', max_depth = 4, max_features = 6, n_estimators = 1000)
----> 6 model_prev.fit(train_x.drop(['Pawpularity'],axis=1), train_x.loc[:,['Pawpularity']].values.ravel())
      7 res_1 = model_prev.predict(test_x)
      8 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    338             Fitted estimator.
    339         """
--> 340         self._validate_params()
    341 
    342         # Validate or convert input data

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'criterion' parameter of RandomForestRegressor must be a str among {'friedman_mse', 'squared_error', 'absolute_error', 'poisson'}. Got 'mse' instead.

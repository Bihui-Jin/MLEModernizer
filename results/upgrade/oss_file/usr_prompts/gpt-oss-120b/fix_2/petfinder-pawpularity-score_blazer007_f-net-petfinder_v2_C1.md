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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

20.51796

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        break




## === cell 1
os.listdir("/kaggle/input/petfinder-pawpularity-score")




## === cell 2
IMG_PATH = "/kaggle/input/petfinder-pawpularity-score/train/"




## === cell 3
df_train = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")




## === cell 4
df_train.shape




## === cell 5
df_train.head()




## === cell 6
import io
import shutil, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
from PIL import Image
from pathlib import Path





## === cell 8
weight_decay = 0.0001
batch_size = 64
num_epochs = 5
dropout_rate = 0.2
image_size = 224  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
embedding_dim = 256  # Number of hidden units.
num_blocks = 4  # Number of blocks.
num_classes = 1




## === cell 9
image_data = []
labels = []

for idx in range(df_train.shape[0]):
    img_path = IMG_PATH + str(df_train["Id"][idx]) + ".jpg"
    try:
        img = tf.keras.preprocessing.image.load_img(
            img_path, color_mode="rgb", target_size=(image_size, image_size)
        )
        img = np.array(img)
        image_data.append(img)
        labels.append(df_train["Pawpularity"][idx])
    except:
        ...
    else:
        ...




## === cell 10
print(len(image_data), len(labels))




## === cell 11
combined = list(zip(image_data, labels))
random.shuffle(combined)




## === cell 12
X_train = np.array(image_data)
Y_train = np.array(labels)
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_train = Y_train.astype("float32")
print(X_train.shape)
print(Y_train.shape)




## === cell 13
x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)




## === cell 14
y_train.dtype




## === cell 15
"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)




## === cell 16
"""
## Build a regression model
We implement a method that builds a regression model given the processing blocks.
"""


def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size, num_patches)(augmented)
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dim
        )(positions)
        x = x + position_embedding
    x = blocks(x)
    representation = layers.GlobalAveragePooling1D()(x)
    representation = layers.Dropout(rate=dropout_rate)(representation)
    logits = layers.Dense(num_classes, activation="linear")(representation)
    return keras.Model(inputs=inputs, outputs=logits)




## === cell 17
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




## === cell 18
"""
## Define an experiment
We implement a utility function to compile, train, and evaluate a given model.
"""


def run_experiment(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=root_mean_squared_error,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
    )

    print(model.evaluate(x_test, y_test))

    return history




## === cell 19
"""
## Implement patch extraction as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches




## === cell 20
"""
### Implement the FNet module
"""


class FNetLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(FNetLayer, self).__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim),
                layers.GELU(),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = tf.cast(
            tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)),
            dtype=tf.dtypes.float32,
        )
        x = x + inputs
        x = self.normalize1(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        return self.normalize2(x)




## === cell 21
"""
### Build, train, and evaluate the FNet model
"""

fnet_blocks = keras.Sequential(
    [FNetLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.001
fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
history = run_experiment(fnet_classifier)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/3170819962.py in <cell line: 0>()
      4 
      5 fnet_blocks = keras.Sequential(
----> 6     [FNetLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
      7 )
      8 learning_rate = 0.001

/tmp/ipykernel_56/3170819962.py in <listcomp>(.0)
      4 
      5 fnet_blocks = keras.Sequential(
----> 6     [FNetLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
      7 )
      8 learning_rate = 0.001

/tmp/ipykernel_56/1783325227.py in __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs)
     11             [
     12                 layers.Dense(units=embedding_dim),
---> 13                 layers.GELU(),
     14                 layers.Dropout(rate=dropout_rate),
     15                 layers.Dense(units=embedding_dim),

AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'GELU'

## === cell 22
df_test = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")




## === cell 23
df_test.head()




## === cell 24
sample_sub = pd.read_csv(
    "/kaggle/input/petfinder-pawpularity-score/sample_submission.csv"
)




## === cell 25
sample_sub.head()




## === cell 26
TEST_IMG_PATH = "/kaggle/input/petfinder-pawpularity-score/test/"




## === cell 27
output = {"Id": [], "Pawpularity": []}

for idx in range(df_test.shape[0]):
    img_path = TEST_IMG_PATH + str(df_test["Id"][idx]) + ".jpg"
    try:
        image = tf.keras.preprocessing.image.load_img(
            img_path, color_mode="rgb", target_size=(image_size, image_size)
        )
        image = np.array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        preds = fnet_classifier.predict(image, verbose=0)
        output["Id"].append(df_test["Id"][idx])
        output["Pawpularity"].append(preds[0][0])
    except:
        ...
    else:
        ...




## === cell 28
sub = pd.DataFrame(output)




## === cell 29
sub.shape




## === cell 30
sub.head()




## === cell 31
sub.to_csv("submission.csv", index=False)

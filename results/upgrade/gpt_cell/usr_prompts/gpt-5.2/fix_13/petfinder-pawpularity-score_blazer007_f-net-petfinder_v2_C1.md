# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        break



## === cell 1
os.listdir('/kaggle/input/petfinder-pawpularity-score')


## === cell 2
IMG_PATH = '/kaggle/input/petfinder-pawpularity-score/train/'


## === cell 3
df_train = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')


## === cell 4
df_train.shape


## === cell 5
df_train.head()


## === cell 6
import io
import shutil, random
import os
import sys
import subprocess
import importlib

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

for m in list(sys.modules):
    if m == "google" or m.startswith("google."):
        sys.modules.pop(m, None)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

tfa = None


## === cell 7
from PIL import Image
from pathlib import Path
from tensorflow import keras
from keras import backend as K

try:
    from keras.utils import np_utils  # legacy (older Keras)
except ImportError:
    from keras.utils import to_categorical

    class _NPUtils:
        to_categorical = staticmethod(to_categorical)

    np_utils = _NPUtils()

from sklearn.utils import shuffle
from tensorflow.keras import layers
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    img_path = IMG_PATH + str(df_train['Id'][idx]) + '.jpg'
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb', target_size= (image_size, image_size))
        img = np.array(img)
        image_data.append(img)
        labels.append(df_train['Pawpularity'][idx])
    except:
        ...
    else:
        ...


## === cell 10
print(len(image_data),len(labels))


## === cell 11
combined = list(zip(image_data,labels))
random.shuffle(combined)


## === cell 12
X_train = np.array(image_data)
Y_train = np.array(labels)
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_train = Y_train.astype("float32")
print(X_train.shape)
print(Y_train.shape)


## === cell 13
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


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
    logits = layers.Dense(num_classes, activation = 'linear')(representation)
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
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    
    model.compile(
        optimizer = optimizer,
        loss = root_mean_squared_error,
        metrics=[
            keras.metrics.RootMeanSquaredError()
        ],
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
## The FNet model
The FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer
in the Transformer block with a parameter-free 2D Fourier transformation layer:
1. One 1D Fourier Transform is applied along the patches.
2. One 1D Fourier Transform is applied along the channels.
"""

"""
### Implement the FNet module
"""

class FNetLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(FNetLayer, self).__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim),
                tfa.layers.GELU(),
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
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/965510735.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m fnet_blocks = keras.Sequential(
[0;32m----> 6[0;31m     [0;34m[[0m[0mFNetLayer[0m[0;34m([0m[0mnum_patches[0m[0;34m,[0m [0membedding_dim[0m[0;34m,[0m [0mdropout_rate[0m[0;34m)[0m [0;32mfor[0m [0m_[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mnum_blocks[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m )
[1;32m      8[0m [0mlearning_rate[0m [0;34m=[0m [0;36m0.001[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/965510735.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m fnet_blocks = keras.Sequential(
[0;32m----> 6[0;31m     [0;34m[[0m[0mFNetLayer[0m[0;34m([0m[0mnum_patches[0m[0;34m,[0m [0membedding_dim[0m[0;34m,[0m [0mdropout_rate[0m[0;34m)[0m [0;32mfor[0m [0m_[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mnum_blocks[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m )
[1;32m      8[0m [0mlearning_rate[0m [0;34m=[0m [0;36m0.001[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3549786356.py[0m in [0;36m__init__[0;34m(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs)[0m
[1;32m     18[0m             [
[1;32m     19[0m                 [0mlayers[0m[0;34m.[0m[0mDense[0m[0;34m([0m[0munits[0m[0;34m=[0m[0membedding_dim[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m                 [0mtfa[0m[0;34m.[0m[0mlayers[0m[0;34m.[0m[0mGELU[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m                 [0mlayers[0m[0;34m.[0m[0mDropout[0m[0;34m([0m[0mrate[0m[0;34m=[0m[0mdropout_rate[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m                 [0mlayers[0m[0;34m.[0m[0mDense[0m[0;34m([0m[0munits[0m[0;34m=[0m[0membedding_dim[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'NoneType' object has no attribute 'layers'

## === cell 22
df_test = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')

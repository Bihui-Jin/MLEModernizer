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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import warnings
warnings.filterwarnings("ignore")


## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            from google.protobuf import reflection as _reflection

            return _reflection.GeneratedProtocolMessageType(
                descriptor.name, (object,), {"DESCRIPTOR": descriptor}
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import keras.backend as K

from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.utils import to_categorical as _to_categorical


class _NPUtilsShim:
    to_categorical = staticmethod(_to_categorical)


np_utils = _NPUtilsShim()

from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, Latex


## === cell 2
plt.rc('font', size=15)
plt.rc('axes', titlesize=18)  
plt.rc('xtick', labelsize=10)  
plt.rc('ytick', labelsize=10)

sns.set(font_scale = 1.2)
sns.set_style("whitegrid")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


## === cell 3
class Cfg:
    RANDOM_STATE = 2021
    TRAIN_DATA = '../input/petfinder-pawpularity-score/train.csv'
    TEST_DATA = '../input/petfinder-pawpularity-score/test.csv'
    SUBMISSION = '../input/petfinder-pawpularity-score/sample_submission.csv'    
    IMG_FOLDER = '../input/petfinder-pawpularity-score/train'
    IMG_TEST_FOLDER = '../input/petfinder-pawpularity-score/test'
    IMG_RESIZE_FOLDER = './resized'
    SUBMISSION_FILE = './submission.csv'
    
    SAMPLE_FRAC = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.3
    BATCH_SIZE = 64
    IMG_SIZE = 128
    
    INDEX = 'Id'
    TARGET = 'Pawpularity'
    FEATURES = [
        'Subject Focus', 
        'Eyes', 
        'Face', 
        'Near', 
        'Action', 
        'Accessory', 
        'Group', 
        'Collage', 
        'Human', 
        'Occlusion', 
        'Info', 
        'Blur'
    ]


## === cell 4
if not os.path.isdir(Cfg.IMG_RESIZE_FOLDER):
    os.makedirs(Cfg.IMG_RESIZE_FOLDER)


## === cell 5
def read_data(
    train_file:str=Cfg.TRAIN_DATA, 
    test_file:str=Cfg.TEST_DATA
) -> (pd.DataFrame, pd.DataFrame):
    """Reads the csv files `train.csv` and `test.csv` and returns 
       them as pandas data frames.
    """
    train_df = pd.read_csv(Cfg.TRAIN_DATA, index_col=Cfg.INDEX)
    test_df = pd.read_csv(Cfg.TEST_DATA, index_col=Cfg.INDEX)

    return train_df, test_df


train_df, test_df = read_data()


## === cell 6
train_df


## === cell 7
test_df


## === cell 8
train_df.describe().drop('count')


## === cell 9
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.histplot(
    data=train_df,
    x=Cfg.TARGET,
    bins=30,
    legend=True,
    kde=True,
    ax=ax[0])

ax[0].set_title('Target Distribution')

ax[0].set_xlabel('Pawpularity')
ax[0].set_ylabel('Count')

sns.boxplot(
    data=train_df,
    y=Cfg.TARGET,
    ax=ax[1]
)

plt.show()


## === cell 10
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

for f, ax in zip(Cfg.FEATURES, axes.flatten()):
    sns.histplot(
        data=train_df,
        x=Cfg.TARGET,
        bins=30,
        hue=f,
        legend=True,
        kde=True,
        ax=ax,
        alpha=0.3)

fig.tight_layout()    
plt.show()


## === cell 11
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

for f, ax in zip(Cfg.FEATURES, axes.flatten()):
    sns.countplot(
        data=train_df,
        x=f,
        alpha=0.8,
        ax=ax)

fig.tight_layout()
plt.show()


## === cell 12
corr_df = train_df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

mask = np.triu(np.ones_like(corr_df, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_df, 
    mask=mask, 
    cmap=cmap, 
    vmin = -0.75, 
    vmax = 0.75,
    center=0,
    square=True,
    annot = True,
    fmt="0.0",
    linewidths=.5)

fig.tight_layout()
plt.show()


## === cell 13
def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
    resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, '{}.jpg'.format(image_id))
    if os.path.isfile(resized_path):
        img = Image.open(resized_path)
        return img
    
    img_path = os.path.join(image_folger, '{}.jpg'.format(image_id))
    
    img = Image.open(img_path)
    img = img.resize((Cfg.IMG_SIZE, Cfg.IMG_SIZE))
    img.save(resized_path)
        
    return img


## === cell 14
def plot_images(data, nrows=5, ncols=5, figsize=(15, 15)):
    """
    """
    indices = data.sample(nrows * ncols).index
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for index, ax in zip(indices, axes.flatten()):
        img = get_image(index)
        ax.imshow(img)

        ax.set_title(train_df.loc[index][Cfg.TARGET], fontsize=18)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()


## === cell 15
pawpularity_range = [0, 20, 40, 60, 80, 100]
query = lambda i: '{} < Pawpularity and Pawpularity <= {}'.format(pawpularity_range[i], pawpularity_range[i+1])

for i in range(0, 5):
    display(Markdown('### Pawpularity `{}` - `{}`'
        .format(pawpularity_range[i], pawpularity_range[i+1])))
    
    df = train_df.query(query(i))
    plot_images(df, nrows=1, ncols=5, figsize=(15, 5))


## === cell 16
class DataGenerator(Sequence):
    """
    """
    def __init__(
        self, 
        data, 
        target=None, 
        img_folder=Cfg.IMG_FOLDER, 
        batch_size=Cfg.BATCH_SIZE
    ):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.img_folder = img_folder
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        ids = self.data[start_idx : end_idx].index.values
        
        images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
        meta = np.array(self.data[start_idx : end_idx][Cfg.FEATURES]).astype(np.float32)
        
        if self.target is None or not self.target.any():
            return [images, meta]
        
        target = np.array(self.target[start_idx : end_idx]).astype(np.float32)
        return [images, meta], target


## === cell 17
def get_image_model(img_size=Cfg.IMG_SIZE, n_channel=3):
    """
    """
    inputs = layers.Input((img_size, img_size, n_channel))
    x = inputs
    
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = x
    model = keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name='image_cnn_model')

    return model    


## === cell 18
image_model = get_image_model()
image_model.summary()


## === cell 19
def get_meta_model(n_meta_features=12):
    """
    """
    inputs = layers.Input(shape=((n_meta_features, )))
    x = inputs
    
    x = layers.Dense(12, activation='relu')(x) 
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(12, activation='relu')(x) 
    
    outputs = x
    model = keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name='meta_nn_model')

    return model


## === cell 20
meta_model = get_meta_model()
meta_model.summary()


## === cell 21
def get_model(image_model, meta_model):
    """
    """
    x = layers.Concatenate(axis=1)([image_model.output, meta_model.output])
    x = layers.Dense(1, activation='linear')(x)
    output = x

    model = keras.Model(inputs=[image_model.input, meta_model.input], outputs=output)
    return model


## === cell 22
model = get_model(image_model, meta_model)
model.summary()


## === cell 23
data = train_df.sample(frac=Cfg.SAMPLE_FRAC)

X_train, X_val, y_train, y_val = train_test_split(
    data[Cfg.FEATURES],
    data[Cfg.TARGET],
    test_size=Cfg.TEST_SIZE, 
    random_state=Cfg.RANDOM_STATE
)

train_generator = DataGenerator(X_train, y_train)
val_generator = DataGenerator(X_val, y_val)


## === cell 24
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=Cfg.LEARNING_RATE,
    ), 
    loss = keras.losses.MeanSquaredError(),
    metrics=[ 
        keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)
    
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=3)
]


## === cell 25
class DataGenerator(Sequence):
    """ """

    def __init__(
        self, data, target=None, img_folder=Cfg.IMG_FOLDER, batch_size=Cfg.BATCH_SIZE
    ):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.img_folder = img_folder

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        ids = self.data[start_idx:end_idx].index.values

        images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
        meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)

        if self.target is None or not self.target.any():
            return (images, meta)

        target = np.array(self.target[start_idx:end_idx]).astype(np.float32)
        return (images, meta), target


## === cell 26
class _TupleAdapter(Sequence):
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        batch = self.seq[idx]
        if isinstance(batch, tuple) and len(batch) == 2:
            x, y = batch
            if isinstance(x, list):
                x = tuple(x)
            return x, y
        if isinstance(batch, list):
            return tuple(batch)
        return batch


y_pred = model.predict(_TupleAdapter(val_generator)).reshape(-1)

df = pd.DataFrame({"y_pred": y_pred, "y_val": y_val})
df["error"] = np.abs(y_pred - y_val)


## === cell 27
fig, ax = plt.subplots(1, 4, figsize=(23, 5))

ax[0].plot(result.history['rmse'])
ax[0].plot(result.history['val_rmse'])

ax[0].set_title('Model RMSE')
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('Epoch')
ax[0].legend(['train', 'val'], loc='upper right')

sns.regplot(
    data=df,
    x='y_val',
    y='y_pred',
    ax=ax[1],
    x_estimator=np.mean, 
    x_bins=30
)

ax[1].set_title('Regression True vs. Pred')
ax[1].set_ylabel('Pred (val)')
ax[1].set_xlabel('True (val)')

sns.histplot(
    data=df,
    x='y_pred',
    bins=30,
    legend=True,
    kde=True,
    ax=ax[2])

ax[2].set_title('Pred Target Distribution')

ax[2].set_xlabel('Pred Pawpularity')
ax[2].set_ylabel('Count')

sns.scatterplot(data=df, x='y_pred', y='error', ax=ax[3])
ax[3].set_title('Residuals')

ax[3].set_xlabel('Prediction')
ax[3].set_ylabel('Absolute error')

plt.tight_layout()
plt.show()


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2560870829.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;31m# plot model rmse[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0max[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mresult[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'rmse'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0max[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mresult[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'val_rmse'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;31mNameError[0m: name 'result' is not defined

## === cell 28
test_generator = DataGenerator(test_df, img_folder=Cfg.IMG_TEST_FOLDER)
y_pred_submission = model.predict(test_generator).reshape(-1)

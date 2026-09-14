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

18.196449398872364

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from IPython.display import FileLink

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df_path = '../input/petfinder-pawpularity-score/train.csv'
test_df_path = '../input/petfinder-pawpularity-score/test.csv'
train_data_path = '../input/petfinder-pawpularity-score/train'
test_data_path = '../input/petfinder-pawpularity-score/test'

checkpoint_filepath = 'save_models2'
checkpoint_filepath2 = 'save_models'


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 456
TARGET = 'Pawpularity'
SEED = 88
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
TEST_SIZE = .15
EPOCHS = 15
EPOCHS_2 = 10
DATA_SHAPE = 12

LEARNING_RATE = 1e-3
DECAY_STEPS = 100
DECAY_RATE = 0.96

## === cell 2
df_train = pd.read_csv(train_df_path)
df_test = pd.read_csv(test_df_path)

print("The shape of train dataset: ", df_train.shape)
print("The shape of test dataset: ", df_test.shape)
print("######## Train ########")
display(df_train[:2])

## === cell 3
def join_path_train(Id):
    """Add path and .jpg to foto Id"""
    return os.path.join(train_data_path + "/" + Id + ".jpg")

def join_path_test(Id):
    """Add path and .jpg to foto Id"""
    return os.path.join(test_data_path + "/" + Id + ".jpg")

def join_jpg(Id):
    """Add .jpg to foto Id."""
    return os.path.join(Id + ".jpg") 


df_train["Path"] = df_train["Id"].apply(join_path_train)
df_train["Filename"] = df_train["Id"].apply(join_jpg)

df_test["Path"] = df_test["Id"].apply(join_path_test)
df_test["Filename"] = df_test["Id"].apply(join_jpg)


## === cell 4
display(df_train.dtypes)

## === cell 6
def get_image(path, resize_method = 'mitchellcubic'):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], 
                            method = resize_method)
    image = tf.cast(image, dtype = tf.int32)
    return tf.keras.applications.efficientnet.preprocess_input(image)

def get_pair_of_dataset(path, label):
    return get_image(path), label

def get_dataset(x, label = None):
    if label is not None:
        return_set = tf.data.Dataset.from_tensor_slices((x, label))
        return return_set.map(get_pair_of_dataset)    \
                         .batch(BATCH_SIZE)           \
                         .prefetch(buffer_size = AUTOTUNE)
    else:
        return_set = tf.data.Dataset.from_tensor_slices(x)
        return return_set.map(get_image)              \
                         .batch(BATCH_SIZE)           \
                         .prefetch(buffer_size = AUTOTUNE)
    
    
def creat_dataset_metadata(df, drop_colums, label = None):

    if label:
        input_dataset_data = tf.data.Dataset.from_tensor_slices(df.drop(['Id', 'Pawpularity', "Path", "Filename"], axis=1))
        input_dataset_img = tf.data.Dataset.from_tensor_slices(df["Path"]).map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
        output_dataset = tf.data.Dataset.from_tensor_slices(df[label])
        dataset = tf.data.Dataset.zip(((input_dataset_img, input_dataset_data), output_dataset))   \
                                    .batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
        return dataset

    else:
        input_dataset_data = tf.data.Dataset.from_tensor_slices(df.drop(['Id', "Path", "Filename"], axis=1))
        input_dataset_img = tf.data.Dataset.from_tensor_slices(df["Path"]).map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip(((input_dataset_img, input_dataset_data), ))   \
                                    .batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

        return dataset




## === cell 7
train_set, test_set = train_test_split( df_train,
                                       test_size=TEST_SIZE,
                                       shuffle=True,
                                       random_state=SEED)


train_set = creat_dataset_metadata(train_set, drop_colums = ['Id', 'Pawpularity', "Path", "Filename"], label = 'Pawpularity')
test_set = creat_dataset_metadata(test_set, drop_colums = ['Id', 'Pawpularity', "Path", "Filename"], label = 'Pawpularity')

## === cell 8
X = df_train.drop(['Id', "Pawpularity"], axis=1)

## === cell 9
'''
model_EffNet = tf.keras.models.load_model("../input/keras-applications-models/EfficientNetB5.h5")

model_EffNet.trainable = False

augementation = tf.keras.Sequential(
        [
        tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomWidth(factor=(0.2, 0.3)), 
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.2, 0.3)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
        tf.keras.layers.experimental.preprocessing.RandomHeight(0.2)
        ])


def get_model():
        # image model
        img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name = "image input")
        x = augementation(img_input)
        x = model_EffNet(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        img_output = tf.keras.layers.Dense(32, activation ='relu')(x)
        img_model = tf.keras.Model(img_input, img_output)
        
        
        # metadata model
        data_input = tf.keras.layers.Input(shape=DATA_SHAPE, name = "data input")
        x = tf.keras.layers.Dense(64, activation='relu')(data_input)
        #x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        data_output = tf.keras.layers.Dense(32, activation='relu')(x)
        data_model = tf.keras.Model(data_input, data_output)

        # concatinating Model layers
        concat_layer = tf.keras.layers.Concatenate(name = 'concat_layer')([img_model.output, data_model.output])
        combined_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(concat_layer)
        combined_dence = tf.keras.layers.Dense(32, activation='relu')(combined_dropout)
        combined_batch = tf.keras.layers.BatchNormalization()(combined_dence)
        final_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(combined_batch)
        output_layer = tf.keras.layers.Dense(1, activation='relu')(final_dropout)

        model = tf.keras.Model(inputs = [img_model.input, data_model.input], outputs=output_layer)
    
        return model

model = get_model()
'''

## === cell 10
model = tf.keras.models.load_model("../input/effnetb5-data-pretrain/pawpul_EffNetB5data.h5")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1392620750.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("../input/effnetb5-data-pretrain/pawpul_EffNetB5data.h5")

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/effnetb5-data-pretrain/pawpul_EffNetB5data.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 11
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3536846749.py in <cell line: 0>()
      4     staircase=True)
      5 
----> 6 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
      7                     loss=tf.keras.losses.MeanSquaredError(),
      8                     metrics=[tf.keras.metrics.RootMeanSquaredError()])

NameError: name 'model' is not defined

## === cell 12
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2727731566.py in <cell line: 0>()
      1 from tensorflow.keras.utils import plot_model
----> 2 plot_model(model, show_shapes=True)

NameError: name 'model' is not defined

## === cell 13



model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_root_mean_squared_error',
    mode='max',
    verbose = 1,
    save_best_only=True)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1677493858.py in <cell line: 0>()
     14 
     15 
---> 16 model_checkpoint = ModelCheckpoint(
     17     filepath=checkpoint_filepath,
     18     save_weights_only=True,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=save_models2

## === cell 14
test_data = creat_dataset_metadata(df_test, drop_colums = ['Id', "Path", "Filename"])


## === cell 15
prediction =  model.predict(test_data)
df_test["Pawpularity"] = prediction
df_test[["Id","Pawpularity"]].to_csv('submission.csv', index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/428949758.py in <cell line: 0>()
----> 1 prediction =  model.predict(test_data)
      2 df_test["Pawpularity"] = prediction
      3 df_test[["Id","Pawpularity"]].to_csv('submission.csv', index=False)

NameError: name 'model' is not defined

## === cell 16
print(prediction)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2641910672.py in <cell line: 0>()
----> 1 print(prediction)

NameError: name 'prediction' is not defined

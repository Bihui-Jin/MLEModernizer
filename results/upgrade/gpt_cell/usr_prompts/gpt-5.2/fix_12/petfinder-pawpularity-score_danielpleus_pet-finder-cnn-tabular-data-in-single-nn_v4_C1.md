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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image
import math


## === cell 1
batch_size = 32
validation_split = 0.2
random_seed = 42
img_size = (64,64)


## === cell 2
training_dir = "../input/petfinder-pawpularity-score/train"
test_dir = "../input/petfinder-pawpularity-score/test"


## === cell 3
training_df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")[0:8000]
validation_df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")[8001:]
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")


## === cell 4
training_df


## === cell 5
target = "Pawpularity"
features = training_df.drop(columns = {"Id",target}).columns.values


## === cell 6
def get_image(image_id, image_folder, img_size, resized_folder):
    resized_path = os.path.join(resized_folder,'{}.jpg'.format(image_id))
    
    if os.path.isfile(resized_path):
        img = Image.open(resized_path)
        return img
    
    img_path = os.path.join(image_folder, '{}.jpg'.format(image_id))
    
    img = Image.open(img_path)
    img = img.resize(img_size)
    img.save(resized_path)
        
    return img

class DataGenerator(utils.Sequence):
    def __init__(
        self, 
        data_type,
        data, 
        target, 
        img_folder, 
        img_size,
        batch_size,
        features
    ):
        
    
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.img_folder = img_folder
        self.img_size = img_size
        self.data_type = data_type
        
        self.resized_folder = os.path.join("/resized", data_type)
        
        if os.path.isdir(self.resized_folder) == False:
            os.makedirs(self.resized_folder)
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        ids = self.data[start_idx : end_idx]["Id"]
        
        images = np.array([np.array(get_image(id, self.img_folder, self.img_size, self.resized_folder)) for id in ids])
        meta = np.array(self.data[start_idx : end_idx][features]).astype(np.float32)
        
        images = tf.cast(images, tf.float32)
        if self.target is None or not self.target.any():
            return [images, meta]
        else:
            target = np.array(self.target[start_idx : end_idx]).astype(np.float32)
            return [images, meta], target
      


## === cell 7
training_ds = DataGenerator("training",training_df, training_df[target], training_dir, img_size, batch_size, features)
validation_ds = DataGenerator("validation",validation_df, validation_df[target], training_dir, img_size, batch_size, features)
test_ds = DataGenerator("test",test_df, None, test_dir, img_size, batch_size, features)


## === cell 8
len(features)


## === cell 9
inputs = keras.Input(shape=(12,))
img_inputs = keras.Input(shape=(64, 64, 3))

img = layers.RandomRotation(0.2)(img_inputs)
img = layers.RandomFlip('horizontal')(img)
img1 = layers.Conv2D(4, (4, 4), activation='relu')(img)
img2 = layers.MaxPooling2D((4, 4))(img1)
img3 = layers.Conv2D(8, (4, 4), activation='relu')(img2)
img6 = layers.Flatten()(img3)
img7 = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(img6)
img8 = layers.Dropout(0.3)(img7)
img9 = layers.Dense(64)(img8)
img10 = layers.Dense(8)(img9)
img = keras.Model(inputs=img_inputs, outputs=img10)

tab1 = layers.Dense(8,input_shape = (12,))(inputs)
tab = keras.Model(inputs=inputs, outputs=tab1)

x1 = layers.Concatenate(axis=1)([tab.output, img.output])
x2 = layers.Dense(32)(x1)
x3 = layers.Dense(16)(x2)
outputs = layers.Dense(1)(x3)

model = keras.Model(inputs=[img.input, tab.input], outputs=outputs, name="model")


## === cell 10
model.summary()


## === cell 11
model.compile(optimizer='adam',
              loss=tf.keras.losses.MSE,
              metrics=['mse'])


## === cell 12
import math


def _train_gen():
    for i in range(len(training_ds)):
        (imgs, meta), y = training_ds[i]
        yield (imgs, meta), y


def _val_gen():
    for i in range(len(validation_ds)):
        (imgs, meta), y = validation_ds[i]
        yield (imgs, meta), y


output_signature = (
    (
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32),
)

train_dataset = tf.data.Dataset.from_generator(
    _train_gen, output_signature=output_signature
)
val_dataset = tf.data.Dataset.from_generator(
    _val_gen, output_signature=output_signature
)

history = model.fit(train_dataset, epochs=20, validation_data=val_dataset)


## === cell 13
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')



## === cell 14
def _test_gen():
    for i in range(len(test_ds)):
        batch = test_ds[i]
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, meta = batch
        else:
            imgs = batch
            meta = np.zeros((imgs.shape[0], len(features)), dtype=np.float32)
        yield (tf.cast(imgs, tf.float32), tf.cast(meta, tf.float32))


test_output_signature = (
    (
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
    ),
)

test_dataset = tf.data.Dataset.from_generator(
    _test_gen, output_signature=test_output_signature
)

prediction = model.predict(test_dataset)

submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
submission["Pawpularity"] = np.asarray(prediction).reshape(-1)
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidArgumentError[0m                      Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/763721114.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     24[0m )
[1;32m     25[0m [0;34m[0m[0m
[0;32m---> 26[0;31m [0mprediction[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mtest_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m [0;34m[0m[0m
[1;32m     28[0m [0msubmission[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m"../input/petfinder-pawpularity-score/sample_submission.csv"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36mraise_from_not_ok_status[0;34m(e, name)[0m
[1;32m   6000[0m [0;32mdef[0m [0mraise_from_not_ok_status[0m[0;34m([0m[0me[0m[0;34m,[0m [0mname[0m[0;34m)[0m [0;34m->[0m [0mNoReturn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6001[0m   [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m([0m[0;34m" name: "[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;34m""[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6002[0;31m   [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m  [0;31m# pylint: disable=protected-access[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6003[0m [0;34m[0m[0m
[1;32m   6004[0m [0;34m[0m[0m

[0;31mInvalidArgumentError[0m: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} TypeError: `generator` yielded an element that did not match the expected structure. The expected structure was ((tf.float32, tf.float32),), but the yielded element was (<tf.Tensor: shape=(32, 64, 64, 3), dtype=float32, numpy=
array([[[[ 16.,   9.,   2.],
         [ 26.,  15.,   4.],
         [ 26.,  13.,   5.],
         ...,
         [  5.,  66., 226.],
         [ 17.,  72., 221.],
         [ 34.,  85., 216.]],

        [[  2.,   2.,   0.],
         [  7.,   4.,   1.],
         [ 11.,   7.,   1.],
         ...,
         [  2.,  62., 225.],
         [ 16.,  71., 217.],
         [ 30.,  84., 216.]],

        [[  6.,   4.,   1.],
         [  4.,   4.,   1.],
         [  9.,   7.,   2.],
         ...,
         [  1.,  61., 225.],
         [ 19.,  74., 217.],
         [ 19.,  75., 215.]],

        ...,

        [[210., 180., 167.],
         [208., 179., 163.],
         [210., 177., 162.],
         ...,
         [213., 163., 141.],
         [212., 164., 142.],
         [211., 163., 140.]],

        [[207., 176., 163.],
         [207., 178., 163.],
         [213., 179., 164.],
         ...,
         [219., 170., 146.],
         [215., 167., 144.],
         [215., 165., 140.]],

        [[206., 174., 161.],
         [207., 174., 161.],
         [214., 179., 167.],
         ...,
         [218., 168., 145.],
         [218., 168., 144.],
         [217., 168., 143.]]],


       [[[ 59.,  26.,   5.],
         [ 64.,  27.,   5.],
         [ 68.,  31.,   5.],
         ...,
         [ 52.,  41.,  37.],
         [ 68.,  53.,  47.],
         [113.,  95.,  84.]],

        [[ 62.,  26.,   6.],
         [ 65.,  30.,   4.],
         [ 71.,  31.,   7.],
         ...,
         [ 72.,  58.,  55.],
         [121., 103.,  90.],
         [137., 115.,  96.]],

        [[ 64.,  28.,   5.],
         [ 70.,  32.,   6.],
         [ 79.,  34.,   7.],
         ...,
         [ 49.,  38.,  37.],
         [119.,  96.,  74.],
         [123.,  95.,  72.]],

        ...,

        [[ 15.,  11.,   7.],
         [ 12.,   8.,   5.],
         [  4.,   3.,   1.],
         ...,
         [124.,  80.,  33.],
         [107.,  66.,  27.],
         [ 89.,  53.,  21.]],

        [[ 15.,  10.,   7.],
         [ 15.,  10.,   7.],
         [ 11.,   7.,   5.],
         ...,
         [105.,  62.,  24.],
         [ 91.,  54.,  20.],
         [ 77.,  45.,  17.]],

        [[ 15.,  10.,   7.],
         [ 14.,   9.,   6.],
         [ 13.,   9.,   6.],
         ...,
         [ 86.,  51.,  18.],
         [ 74.,  42.,  16.],
         [ 62.,  34.,  14.]]],


       [[[  6.,   4.,   5.],
         [  6.,   4.,   5.],
         [  6.,   4.,   4.],
         ...,
         [132., 123., 108.],
         [133., 124., 108.],
         [133., 124., 108.]],

        [[  6.,   5.,   4.],
         [  6.,   5.,   4.],
         [  6.,   5.,   5.],
         ...,
         [137., 129., 115.],
         [137., 129., 115.],
         [139., 131., 116.]],

        [[ 13.,  12.,  10.],
         [ 13.,  12.,  10.],
         [ 10.,   9.,   8.],
         ...,
         [142., 134., 122.],
         [144., 136., 123.],
         [146., 136., 125.]],

        ...,

        [[106., 104., 107.],
         [107., 106., 109.],
         [102., 101., 103.],
         ...,
         [117., 107., 106.],
         [116., 106., 105.],
         [113., 104., 102.]],

        [[102.,  99., 102.],
         [ 99.,  97., 100.],
         [ 99.,  97., 100.],
         ...,
         [118., 108., 107.],
         [116., 106., 105.],
         [116., 106., 105.]],

        [[ 97.,  95.,  98.],
         [ 96.,  94.,  97.],
         [ 97.,  95.,  98.],
         ...,
         [118., 108., 107.],
         [118., 108., 107.],
         [121., 110., 108.]]],


       ...,


       [[[178., 155., 106.],
         [179., 155., 108.],
         [180., 158., 111.],
         ...,
         [184., 172., 130.],
         [183., 173., 132.],
         [184., 174., 135.]],

        [[180., 156., 107.],
         [180., 156., 109.],
         [180., 157., 110.],
         ...,
         [187., 177., 136.],
         [187., 177., 137.],
         [187., 178., 139.]],

        [[182., 158., 110.],
         [181., 158., 111.],
         [181., 158., 111.],
         ...,
         [187., 177., 138.],
         [189., 180., 141.],
         [191., 181., 144.]],

        ...,

        [[172., 159., 120.],
         [171., 157., 118.],
         [167., 152., 111.],
         ...,
         [145., 131.,  86.],
         [143., 129.,  84.],
         [143., 130.,  85.]],

        [[172., 158., 118.],
         [174., 158., 119.],
         [172., 157., 115.],
         ...,
         [148., 134.,  89.],
         [146., 132.,  87.],
         [143., 129.,  84.]],

        [[172., 157., 115.],
         [173., 158., 116.],
         [170., 155., 112.],
         ...,
         [144., 130.,  85.],
         [144., 130.,  86.],
         [145., 131.,  86.]]],


       [[[207., 221., 222.],
         [153., 169., 167.],
         [ 93.,  92.,  93.],
         ...,
         [244., 251., 252.],
         [195., 224., 237.],
         [226., 236., 241.]],

        [[186., 206., 206.],
         [131., 157., 158.],
         [ 23.,  22.,  22.],
         ...,
         [224., 239., 244.],
         [160., 208., 229.],
         [220., 234., 240.]],

        [[185., 203., 203.],
         [149., 175., 176.],
         [ 45.,  44.,  42.],
         ...,
         [218., 233., 235.],
         [146., 201., 224.],
         [225., 240., 245.]],

        ...,

        [[173., 186., 195.],
         [136., 158., 172.],
         [138., 161., 175.],
         ...,
         [103., 113.,  44.],
         [107., 104.,  57.],
         [162., 155., 128.]],

        [[168., 184., 193.],
         [129., 153., 167.],
         [139., 163., 177.],
         ...,
         [ 97., 111.,  49.],
         [129., 118.,  70.],
         [147., 138., 109.]],

        [[195., 207., 212.],
         [170., 188., 196.],
         [187., 203., 212.],
         ...,
         [128., 143.,  96.],
         [123., 133.,  94.],
         [182., 190., 160.]]],


       [[[223., 230., 235.],
         [209., 221., 231.],
         [172., 190., 207.],
         ...,
         [ 71.,  89., 105.],
         [ 78.,  98., 113.],
         [ 97., 117., 133.]],

        [[199., 211., 220.],
         [188., 203., 216.],
         [177., 194., 210.],
         ...,
         [ 77.,  98., 113.],
         [ 96., 117., 133.],
         [114., 135., 153.]],

        [[180., 196., 212.],
         [184., 200., 217.],
         [182., 199., 215.],
         ...,
         [ 97., 118., 134.],
         [112., 132., 150.],
         [133., 154., 173.]],

        ...,

        [[ 36.,  35.,  33.],
         [ 32.,  32.,  30.],
         [ 38.,  38.,  36.],
         ...,
         [ 47.,  57.,  69.],
         [ 40.,  50.,  62.],
         [ 50.,  61.,  72.]],

        [[ 33.,  33.,  31.],
         [ 35.,  35.,  33.],
         [ 37.,  36.,  34.],
         ...,
         [ 44.,  54.,  66.],
         [ 40.,  50.,  62.],
         [ 53.,  65.,  77.]],

        [[ 37.,  36.,  34.],
         [ 35.,  34.,  32.],
         [ 33.,  32.,  30.],
         ...,
         [ 38.,  49.,  60.],
         [ 41.,  52.,  64.],
         [ 56.,  68.,  80.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 12), dtype=float32, numpy=
array([[0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
       [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
       [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>).
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 204, in generator_py_func
    flattened_values = nest.flatten_up_to(output_types, values)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/nest.py", line 237, in flatten_up_to
    return nest_util.flatten_up_to(
           ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1541, in flatten_up_to
    return _tf_data_flatten_up_to(shallow_tree, input_tree)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1570, in _tf_data_flatten_up_to
    _tf_data_assert_shallow_structure(shallow_tree, input_tree)

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1427, in _tf_data_assert_shallow_structure
    raise ValueError(

ValueError: The two structures don't have the same sequence length. Input structure has length 2, while shallow structure has length 1.


The above exception was the direct cause of the following exception:


Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 206, in generator_py_func
    raise TypeError(

TypeError: `generator` yielded an element that did not match the expected structure. The expected structure was ((tf.float32, tf.float32),), but the yielded element was (<tf.Tensor: shape=(32, 64, 64, 3), dtype=float32, numpy=
array([[[[ 16.,   9.,   2.],
         [ 26.,  15.,   4.],
         [ 26.,  13.,   5.],
         ...,
         [  5.,  66., 226.],
         [ 17.,  72., 221.],
         [ 34.,  85., 216.]],

        [[  2.,   2.,   0.],
         [  7.,   4.,   1.],
         [ 11.,   7.,   1.],
         ...,
         [  2.,  62., 225.],
         [ 16.,  71., 217.],
         [ 30.,  84., 216.]],

        [[  6.,   4.,   1.],
         [  4.,   4.,   1.],
         [  9.,   7.,   2.],
         ...,
         [  1.,  61., 225.],
         [ 19.,  74., 217.],
         [ 19.,  75., 215.]],

        ...,

        [[210., 180., 167.],
         [208., 179., 163.],
         [210., 177., 162.],
         ...,
         [213., 163., 141.],
         [212., 164., 142.],
         [211., 163., 140.]],

        [[207., 176., 163.],
         [207., 178., 163.],
         [213., 179., 164.],
         ...,
         [219., 170., 146.],
         [215., 167., 144.],
         [215., 165., 140.]],

        [[206., 174., 161.],
         [207., 174., 161.],
         [214., 179., 167.],
         ...,
         [218., 168., 145.],
         [218., 168., 144.],
         [217., 168., 143.]]],


       [[[ 59.,  26.,   5.],
         [ 64.,  27.,   5.],
         [ 68.,  31.,   5.],
         ...,
         [ 52.,  41.,  37.],
         [ 68.,  53.,  47.],
         [113.,  95.,  84.]],

        [[ 62.,  26.,   6.],
         [ 65.,  30.,   4.],
         [ 71.,  31.,   7.],
         ...,
         [ 72.,  58.,  55.],
         [121., 103.,  90.],
         [137., 115.,  96.]],

        [[ 64.,  28.,   5.],
         [ 70.,  32.,   6.],
         [ 79.,  34.,   7.],
         ...,
         [ 49.,  38.,  37.],
         [119.,  96.,  74.],
         [123.,  95.,  72.]],

        ...,

        [[ 15.,  11.,   7.],
         [ 12.,   8.,   5.],
         [  4.,   3.,   1.],
         ...,
         [124.,  80.,  33.],
         [107.,  66.,  27.],
         [ 89.,  53.,  21.]],

        [[ 15.,  10.,   7.],
         [ 15.,  10.,   7.],
         [ 11.,   7.,   5.],
         ...,
         [105.,  62.,  24.],
         [ 91.,  54.,  20.],
         [ 77.,  45.,  17.]],

        [[ 15.,  10.,   7.],
         [ 14.,   9.,   6.],
         [ 13.,   9.,   6.],
         ...,
         [ 86.,  51.,  18.],
         [ 74.,  42.,  16.],
         [ 62.,  34.,  14.]]],


       [[[  6.,   4.,   5.],
         [  6.,   4.,   5.],
         [  6.,   4.,   4.],
         ...,
         [132., 123., 108.],
         [133., 124., 108.],
         [133., 124., 108.]],

        [[  6.,   5.,   4.],
         [  6.,   5.,   4.],
         [  6.,   5.,   5.],
         ...,
         [137., 129., 115.],
         [137., 129., 115.],
         [139., 131., 116.]],

        [[ 13.,  12.,  10.],
         [ 13.,  12.,  10.],
         [ 10.,   9.,   8.],
         ...,
         [142., 134., 122.],
         [144., 136., 123.],
         [146., 136., 125.]],

        ...,

        [[106., 104., 107.],
         [107., 106., 109.],
         [102., 101., 103.],
         ...,
         [117., 107., 106.],
         [116., 106., 105.],
         [113., 104., 102.]],

        [[102.,  99., 102.],
         [ 99.,  97., 100.],
         [ 99.,  97., 100.],
         ...,
         [118., 108., 107.],
         [116., 106., 105.],
         [116., 106., 105.]],

        [[ 97.,  95.,  98.],
         [ 96.,  94.,  97.],
         [ 97.,  95.,  98.],
         ...,
         [118., 108., 107.],
         [118., 108., 107.],
         [121., 110., 108.]]],


       ...,


       [[[178., 155., 106.],
         [179., 155., 108.],
         [180., 158., 111.],
         ...,
         [184., 172., 130.],
         [183., 173., 132.],
         [184., 174., 135.]],

        [[180., 156., 107.],
         [180., 156., 109.],
         [180., 157., 110.],
         ...,
         [187., 177., 136.],
         [187., 177., 137.],
         [187., 178., 139.]],

        [[182., 158., 110.],
         [181., 158., 111.],
         [181., 158., 111.],
         ...,
         [187., 177., 138.],
         [189., 180., 141.],
         [191., 181., 144.]],

        ...,

        [[172., 159., 120.],
         [171., 157., 118.],
         [167., 152., 111.],
         ...,
         [145., 131.,  86.],
         [143., 129.,  84.],
         [143., 130.,  85.]],

        [[172., 158., 118.],
         [174., 158., 119.],
         [172., 157., 115.],
         ...,
         [148., 134.,  89.],
         [146., 132.,  87.],
         [143., 129.,  84.]],

        [[172., 157., 115.],
         [173., 158., 116.],
         [170., 155., 112.],
         ...,
         [144., 130.,  85.],
         [144., 130.,  86.],
         [145., 131.,  86.]]],


       [[[207., 221., 222.],
         [153., 169., 167.],
         [ 93.,  92.,  93.],
         ...,
         [244., 251., 252.],
         [195., 224., 237.],
         [226., 236., 241.]],

        [[186., 206., 206.],
         [131., 157., 158.],
         [ 23.,  22.,  22.],
         ...,
         [224., 239., 244.],
         [160., 208., 229.],
         [220., 234., 240.]],

        [[185., 203., 203.],
         [149., 175., 176.],
         [ 45.,  44.,  42.],
         ...,
         [218., 233., 235.],
         [146., 201., 224.],
         [225., 240., 245.]],

        ...,

        [[173., 186., 195.],
         [136., 158., 172.],
         [138., 161., 175.],
         ...,
         [103., 113.,  44.],
         [107., 104.,  57.],
         [162., 155., 128.]],

        [[168., 184., 193.],
         [129., 153., 167.],
         [139., 163., 177.],
         ...,
         [ 97., 111.,  49.],
         [129., 118.,  70.],
         [147., 138., 109.]],

        [[195., 207., 212.],
         [170., 188., 196.],
         [187., 203., 212.],
         ...,
         [128., 143.,  96.],
         [123., 133.,  94.],
         [182., 190., 160.]]],


       [[[223., 230., 235.],
         [209., 221., 231.],
         [172., 190., 207.],
         ...,
         [ 71.,  89., 105.],
         [ 78.,  98., 113.],
         [ 97., 117., 133.]],

        [[199., 211., 220.],
         [188., 203., 216.],
         [177., 194., 210.],
         ...,
         [ 77.,  98., 113.],
         [ 96., 117., 133.],
         [114., 135., 153.]],

        [[180., 196., 212.],
         [184., 200., 217.],
         [182., 199., 215.],
         ...,
         [ 97., 118., 134.],
         [112., 132., 150.],
         [133., 154., 173.]],

        ...,

        [[ 36.,  35.,  33.],
         [ 32.,  32.,  30.],
         [ 38.,  38.,  36.],
         ...,
         [ 47.,  57.,  69.],
         [ 40.,  50.,  62.],
         [ 50.,  61.,  72.]],

        [[ 33.,  33.,  31.],
         [ 35.,  35.,  33.],
         [ 37.,  36.,  34.],
         ...,
         [ 44.,  54.,  66.],
         [ 40.,  50.,  62.],
         [ 53.,  65.,  77.]],

        [[ 37.,  36.,  34.],
         [ 35.,  34.,  32.],
         [ 33.,  32.,  30.],
         ...,
         [ 38.,  49.,  60.],
         [ 41.,  52.,  64.],
         [ 56.,  68.,  80.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 12), dtype=float32, numpy=
array([[0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
       [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
       [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>).


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name:

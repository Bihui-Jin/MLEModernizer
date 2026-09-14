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

3.7

# 2. Installed packages

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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import random
import os

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sklearn.utils
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import cv2 as cv

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    DepthwiseConv2D,
    Flatten,
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import json
import logging

import numpy as np
import matplotlib.pyplot as plt


## === cell 1
data_dir = r'../input'
dataset_dir = os.path.join(data_dir, r'train/train')
csv_dir = os.path.join(data_dir , 'train.csv')
print(os.getcwd())
print(dataset_dir)


## === cell 2

from numpy.random import seed

seed(1372)

tf.random.set_seed(1372)


## === cell 3
def resize_and_save(filename, input_dir, output_dir, size=32):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(os.path.join(input_dir, filename))
    image.save(os.path.join(output_dir, filename)) # linux => / windows => \\


## === cell 4
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


## === cell 5
df = pd.read_csv(csv_dir)
df['id'] = dataset_dir + '/' + df['id'].astype(str)
filenames = df['id']
labels = df['has_cactus'].astype(np.float32)


df = sklearn.utils.shuffle(df,random_state=1372)
df = df.reset_index(drop=True)


## === cell 6
print('sample filename ',filenames[0])
print('sample label (1 = exsit) , (0 = dosent exist any cactus) ',labels[0],type(labels[0]))


## === cell 7
def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label


## === cell 8
def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


## === cell 9
def input_fn(is_training, filenames, labels, params):
    """Input function for the dataset.

        Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .repeat()
            .prefetch(32)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .repeat()
            .prefetch(32)  # make sure you always have one batch ready to serve
        )
        
    return dataset


## === cell 10
with open("params.json", "w") as text_file:
    text_file.write("{\n"+
    "\"learning_rate\": 1e-3,"+
    "\"batch_size\": 64,"+
    "\"num_epochs\": 50,"+
    "\"image_size\": 32,"+
    "\"use_random_flip\": true,"+
    "\"num_labels\": 2,"+
    "\"num_parallel_calls\": 4,"+
    "\"save_summary_steps\": 1"+
    "\n}")
    
paramPath = r'./'
json_path = os.path.join(paramPath , 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)


## === cell 11
def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.io.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize(image, [size, size])

    return resized_image, label


## === cell 12
tf.compat.v1.disable_eager_execution()

params = Params(json_path)
train_dataset = input_fn(
    is_training=True, filenames=filenames.values, labels=labels.values, params=params
)

iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
next_element = iterator.get_next()
with tf.compat.v1.Session() as sess:
    one_batch = sess.run(next_element)
    print(one_batch[0].shape, " = 64 batch-size & 32x32x3 image")
    for i in range(3):
        plt.figure()
        sample = one_batch[0]
        label = one_batch[1]
        print(label[i])
        plt.imshow(sample[i])
        plt.grid(False)


## === cell 13
model = Sequential()
        
model.add(Conv2D(3, kernel_size = 3, activation = 'selu', input_shape = (32, 32, 3)))

model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'selu'))
model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'selu'))
model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'selu'))
model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))
model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))
model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation = 'selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation = 'selu'))

model.add(Dense(1, activation = 'sigmoid'))


## === cell 14
model.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]
)
model.summary()


## === cell 15
file_path = 'weights-aerial-cactus.h5'

callbacks = [
        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),
        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, mode = 'min', min_lr = 0.00001),
        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 15, verbose = 1, restore_best_weights = True)
        ]


## === cell 16
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

split = int(0.2 * len(df))  # number of validation samples

train_df = df.iloc[:-split].reset_index(drop=True)
valid_df = df.iloc[-split:].reset_index(drop=True)

train_dataset = input_fn(
    is_training=True,
    filenames=train_df["id"].values,
    labels=train_df["has_cactus"].astype(np.float32).values,
    params=params,
)

valid_dataset = input_fn(
    is_training=False,
    filenames=valid_df["id"].values,
    labels=valid_df["has_cactus"].astype(np.float32).values,
    params=params,
)

steps_per_epoch = int(np.ceil(len(train_df) / float(params.batch_size)))
validation_steps = int(np.ceil(len(valid_df) / float(params.batch_size)))

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    verbose=True,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4282841689.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m# but avoid the TF2/Keras adapter failure.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mif[0m [0;32mnot[0m [0mtf[0m[0;34m.[0m[0mexecuting_eagerly[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0mtf[0m[0;34m.[0m[0mcompat[0m[0;34m.[0m[0mv1[0m[0;34m.[0m[0menable_eager_execution[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0msplit[0m [0;34m=[0m [0mint[0m[0;34m([0m[0;36m0.2[0m [0;34m*[0m [0mlen[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m)[0m  [0;31m# number of validation samples[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36menable_eager_execution[0;34m(config, device_policy, execution_mode)[0m
[1;32m   4961[0m   [0mlogging[0m[0;34m.[0m[0mvlog[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0;34m"Enabling eager execution"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4962[0m   [0;32mif[0m [0mcontext[0m[0;34m.[0m[0mdefault_execution_mode[0m [0;34m!=[0m [0mcontext[0m[0;34m.[0m[0mEAGER_MODE[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4963[0;31m     return enable_eager_execution_internal(
[0m[1;32m   4964[0m         [0mconfig[0m[0;34m=[0m[0mconfig[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4965[0m         [0mdevice_policy[0m[0;34m=[0m[0mdevice_policy[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36menable_eager_execution_internal[0;34m(config, device_policy, execution_mode, server_def)[0m
[1;32m   5025[0m         _default_graph_stack._global_default_graph is not None)  # pylint: disable=protected-access
[1;32m   5026[0m     [0;32mif[0m [0mgraph_mode_has_been_used[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5027[0;31m       raise ValueError(
[0m[1;32m   5028[0m           "tf.enable_eager_execution must be called at program startup.")
[1;32m   5029[0m   [0mcontext[0m[0;34m.[0m[0mdefault_execution_mode[0m [0;34m=[0m [0mcontext[0m[0;34m.[0m[0mEAGER_MODE[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: tf.enable_eager_execution must be called at program startup.

## === cell 17
model.load_weights(file_path)

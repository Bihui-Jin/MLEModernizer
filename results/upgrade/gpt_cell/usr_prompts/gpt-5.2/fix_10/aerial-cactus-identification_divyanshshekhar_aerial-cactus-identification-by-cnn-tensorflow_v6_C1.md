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
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import tensorflow.compat.v2 as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
import math

if hasattr(tf, "executing_eagerly") and not tf.executing_eagerly():
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        pass

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

INPUT_DIR = "../input"
TRAIN_IMG_DIR = INPUT_DIR + "/train/train"
PRED_IMG_DIR = INPUT_DIR + "/test/test"

AUTOTUNE = tf.data.experimental.AUTOTUNE


## === cell 1
df = pd.read_csv(INPUT_DIR+'/train.csv')
df.head()


## === cell 2
train_df, test_df = train_test_split(df, train_size=0.70, random_state=0)

n_training_items = train_df['id'].count()
n_testing_items = test_df['id'].count()


## === cell 3
def img_path(img_file, img_type=0):
    """ 
    img_file: name of image file
    img_type: 0 if for training, 1 for evaluation dataset.
    """
    if img_type==0:
        return TRAIN_IMG_DIR+'/'+img_file
    else:
        return PRED_IMG_DIR+'/'+img_file
    
train_image_paths = [img_path(x) for x in train_df['id']]
train_image_labels = [x for x in train_df['has_cactus']]
    
test_image_paths = [img_path(x) for x in test_df['id']]
test_image_labels = [x for x in test_df['has_cactus']]

path = os.listdir(PRED_IMG_DIR)
pred_images_paths = [img_path(x, 1) for x in path]
n_pred_items = len(pred_images_paths)


## === cell 4
im = Image.open(train_image_paths[0])
print(im.format, im.size, im.mode)
imgplot = plt.imshow(im)


## === cell 5
def load_and_preprocess_image(imagefile):
    
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        image /= 255.0  # normalize to [0,1] range
        return image
    
    image = tf.read_file(imagefile)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


## === cell 6
def load_and_preprocess_image(imagefile):

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        image /= 255.0  # normalize to [0,1] range
        return image

    image = tf.io.read_file(imagefile)
    return preprocess_image(image)


train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
pred_ds = tf.data.Dataset.from_tensor_slices(pred_images_paths)

train_ds = train_ds.map(load_and_preprocess_from_path_label)
test_ds = test_ds.map(load_and_preprocess_from_path_label)
pred_ds = pred_ds.map(load_and_preprocess_image)

train_ds


## === cell 7
BATCH_SIZE = 32
steps_per_epoch = int(math.ceil(n_training_items/BATCH_SIZE))
train_ds1 = (train_ds.cache()
             .apply(
                 tf.data.experimental.shuffle_and_repeat(buffer_size=n_training_items)
             )
             .batch(BATCH_SIZE)
             .prefetch(buffer_size=AUTOTUNE)
            )

test_ds1 = (test_ds.cache()
            .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=n_training_items))
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE))

pred_ds1 = (pred_ds
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
           )

print(train_ds1, pred_ds1)


## === cell 8
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            filters=12,
            strides=1,
            kernel_size=3,
            padding="valid",
            activation=tf.nn.leaky_relu,
            input_shape=(32, 32, 3),
        ),
        keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation=tf.nn.leaky_relu,
            input_shape=(32, 32, 3),
            padding="same",
        ),
        keras.layers.Conv2D(
            filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding="same"
        ),
        keras.layers.AveragePooling2D(pool_size=3, strides=2),
        keras.layers.Conv2D(
            filters=64, kernel_size=3, activation=tf.nn.leaky_relu, padding="same"
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=3, activation=tf.nn.leaky_relu, padding="same"
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()


## === cell 10
model.fit(train_ds1, epochs=6, steps_per_epoch=steps_per_epoch)


## === cell 11
model.evaluate(test_ds1, steps=n_testing_items)


## === cell 12
logits = model.predict(pred_ds1, steps=n_pred_items)
predictions = np.argmax(logits, axis=-1)


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFailedPreconditionError[0m                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1135971339.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mlogits[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mpred_ds1[0m[0;34m,[0m [0msteps[0m[0;34m=[0m[0mn_pred_items[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mpredictions[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0margmax[0m[0;34m([0m[0mlogits[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mFailedPreconditionError[0m: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to MapDataset:5 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: ../input/test/test/test; Is a directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 13
names = np.array([x for x in path])
pred_df = pd.DataFrame(
    {
        "id":names,
        "has_cactus":predictions
    })
pred_df.head()

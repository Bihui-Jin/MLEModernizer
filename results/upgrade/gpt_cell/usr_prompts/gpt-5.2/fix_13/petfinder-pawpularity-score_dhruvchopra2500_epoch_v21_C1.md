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

3.12

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
protobuf==6.33.0
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
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
    _major = int(_pb_ver.split(".")[0])
    if _major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
except Exception:
    pass

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re


## === cell 1
def pythonic_loader_train():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/train/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"
    
    relevant_columns = ['Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur']
    target = 'Pawpularity'
    
    image_names = os.listdir(IMAGES_PATH)
    image_names = image_names[:int(len(image_names)*0.9)]
    np.random.shuffle(image_names)
    metadata_csv = pd.read_csv(CSV_PATH)
    
    for i in image_names:
        img = tf.keras.utils.load_img(path = IMAGES_PATH+i,
                color_mode="rgb",
                target_size=(256,256))
        img = np.array(img)
        
        metadata = metadata_csv[metadata_csv['Id']==i[:-4]]
        features = metadata[relevant_columns].values[0]
        
        y = metadata[target].values[0]
        
        yield({"Image":img,"Feature":features}, y)


## === cell 2
def pythonic_loader_test():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/train/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"
    
    relevant_columns = ['Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur']
    target = 'Pawpularity'
    
    image_names = os.listdir(IMAGES_PATH)
    image_names = image_names[int(len(image_names)*0.1):]
    np.random.shuffle(image_names)
    metadata_csv = pd.read_csv(CSV_PATH)
    
    for i in image_names:
        img = tf.keras.utils.load_img(path = IMAGES_PATH+i,
                color_mode="rgb",
                target_size=(256,256))
        img = np.array(img)
        
        metadata = metadata_csv[metadata_csv['Id']==i[:-4]]
        features = metadata[relevant_columns].values[0]
        
        y = metadata[target].values[0]
        
        yield({"Image":img,"Feature":features}, y)


## === cell 3
train_loader = tf.data.Dataset.from_generator(
            pythonic_loader_train,
            output_types=({'Image': tf.int64,
            'Feature': tf.int64},
               tf.int64))


## === cell 4
test_loader = tf.data.Dataset.from_generator(
                pythonic_loader_train,
                output_types=({'Image': tf.int64,
                'Feature': tf.int64},
               tf.int64))


## === cell 5
input_image = tf.keras.Input(shape=(256, 256, 3))

input_meta = tf.keras.Input(shape=(12,))

l2 = tf.keras.layers.MaxPool2D((2, 2))(input_image)
l3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu")(l2)
l4 = tf.keras.layers.MaxPool2D((2, 2))(l3)
l5 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(l4)
l6 = tf.keras.layers.MaxPool2D((2, 2))(l5)
l7 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(l6)
l_mid1 = tf.keras.layers.MaxPool2D((2, 2))(l7)
l_mid2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(l_mid1)
l8 = tf.keras.layers.Flatten()(l_mid2)
l10 = tf.keras.layers.Dense(1024, activation="gelu")(l8)

combined = tf.keras.layers.concatenate([l10, input_meta])
l12 = tf.keras.layers.Dense(512, activation="gelu")(combined)
bn1 = tf.keras.layers.BatchNormalization()(l12)
le1 = tf.keras.layers.Dense(512, activation="gelu")(bn1)
bn2 = tf.keras.layers.BatchNormalization()(le1)
le2 = tf.keras.layers.Dense(512, activation="gelu")(bn2)
bn3 = tf.keras.layers.BatchNormalization()(le2)
le3 = tf.keras.layers.Dense(512, activation="gelu")(bn3)
bn4 = tf.keras.layers.BatchNormalization()(le3)
le4 = tf.keras.layers.Dense(256, activation="gelu")(bn4)
bn_5 = tf.keras.layers.BatchNormalization()(le4)
le5 = tf.keras.layers.Dense(256, activation="gelu")(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(le5)

layer1 = tf.keras.layers.Dense(256, activation="gelu")(bn_6)
layer2 = tf.keras.layers.BatchNormalization()(layer1)
layer3 = tf.keras.layers.Dense(256, activation="gelu")(layer2)
layer4 = tf.keras.layers.BatchNormalization()(layer3)
layer5 = tf.keras.layers.Dense(256, activation="gelu")(layer4)
layer6 = tf.keras.layers.BatchNormalization()(layer5)

le6 = tf.keras.layers.Dense(128, activation="gelu")(layer6)
bn5 = tf.keras.layers.BatchNormalization()(le6)
le9 = tf.keras.layers.Dense(64, activation="gelu")(bn5)
bn6 = tf.keras.layers.BatchNormalization()(le9)
l13 = tf.keras.layers.Dense(16, activation="gelu")(bn6)
bn7 = tf.keras.layers.BatchNormalization()(l13)
l14 = tf.keras.layers.Dense(1, activation="sigmoid")(bn7)
output = l14 * tf.constant([100], dtype=tf.float32)

model = tf.keras.Model(
    inputs={"Image": input_image, "Feature": input_meta}, outputs={"Label": output}
)


## === cell 6
model.summary()


## === cell 7
initial_learning_rate = 10**(-3)
first_decay_steps=200
lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=first_decay_steps)


## === cell 8
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])


## === cell 9
gen_train = train_loader.batch(64)
gen_train = gen_train.prefetch(128).repeat()

gen_test = test_loader.batch(64)
gen_test = gen_test.prefetch(128)


## === cell 10
print(tf.config.list_logical_devices('GPU'))


## === cell 11

_train_image_count = len(os.listdir("/kaggle/input/petfinder-pawpularity-score/train/"))
steps_per_epoch = int((_train_image_count * 0.95) // 32)

_output_name = "Label"


def _ensure_train_shapes(x, y):
    x = {
        "Image": tf.ensure_shape(x["Image"], (None, 256, 256, 3)),
        "Feature": tf.ensure_shape(x["Feature"], (None, 12)),
    }
    y = tf.ensure_shape(y, (None,))
    y = {_output_name: tf.ensure_shape(tf.cast(y, tf.float32)[:, None], (None, 1))}
    return x, y


gen_train = gen_train.map(_ensure_train_shapes, num_parallel_calls=tf.data.AUTOTUNE)
gen_test = gen_test.map(_ensure_train_shapes, num_parallel_calls=tf.data.AUTOTUNE)

if _output_name not in getattr(model, "output_names", []):
    named_out = tf.keras.layers.Lambda(lambda t: t, name=_output_name)(model.output)
    model = tf.keras.Model(inputs=model.inputs, outputs=named_out)

model.compile(
    optimizer=model.optimizer,
    loss={_output_name: model.loss},
    metrics={_output_name: [tf.keras.metrics.RootMeanSquaredError()]},
)

model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=2,
    validation_data=gen_test,
    validation_steps=1,
)


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3357295400.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     25[0m [0;34m[0m[0m
[1;32m     26[0m model.compile(
[0;32m---> 27[0;31m     [0moptimizer[0m[0;34m=[0m[0mmodel[0m[0;34m.[0m[0moptimizer[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m     [0mloss[0m[0;34m=[0m[0;34m{[0m[0m_output_name[0m[0;34m:[0m [0mmodel[0m[0;34m.[0m[0mloss[0m[0;34m}[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m     [0mmetrics[0m[0;34m=[0m[0;34m{[0m[0m_output_name[0m[0;34m:[0m [0;34m[[0m[0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mmetrics[0m[0;34m.[0m[0mRootMeanSquaredError[0m[0;34m([0m[0;34m)[0m[0;34m][0m[0;34m}[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Functional' object has no attribute 'optimizer'

## === cell 12
def pythonic_loader_test():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/test/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/test.csv"
    
    relevant_columns = ['Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur']
    
    image_names = os.listdir(IMAGES_PATH)
    metadata_csv = pd.read_csv(CSV_PATH)
    
    for i in image_names:
        img = tf.keras.utils.load_img(path = IMAGES_PATH+i,
                color_mode="rgb",
                target_size=(256,256))
        img = np.array(img)
        
        metadata = metadata_csv[metadata_csv['Id']==i[:-4]]
        features = metadata[relevant_columns].values[0]
        
        yield({"Image":img,"Feature":features})

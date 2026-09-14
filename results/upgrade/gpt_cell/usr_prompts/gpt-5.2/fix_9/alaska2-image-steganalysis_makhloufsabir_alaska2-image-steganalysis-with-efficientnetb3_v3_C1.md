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

3.8

# 2. Installed packages

geopandas==0.14.4
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 4. Code solution

## === cell 0
!pip install -q efficientnet


## === cell 1
import os
import sys
import subprocess

try:
    import google.protobuf as _pb

    _pb_ver = getattr(_pb, "__version__", "0")
    if _pb_ver and int(_pb_ver.split(".")[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.layers as l

from keras.regularizers import l2
from keras.optimizers import Adam

import tensorflow.keras.applications.efficientnet as efn

from sklearn.model_selection import train_test_split


## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print(tpu.master())
    print(tpu_strategy.num_replicas_in_sync)
except Exception as e:
    tpu = None
    tpu_strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy:", type(tpu_strategy).__name__)
    print("num_replicas_in_sync:", tpu_strategy.num_replicas_in_sync)


## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

try:
    from kaggle_datasets import KaggleDatasets  # type: ignore

    gcs_path = KaggleDatasets().get_gcs_path()
except Exception:
    gcs_path = None


## === cell 4
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
BATCH_SIZE = 32 * tpu_strategy.num_replicas_in_sync # batch size in tpu
EPOCHS = 1


dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']
df = pd.DataFrame({})
lists = []
cate = []

for dir_ in dir_name:
    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")
    lists = lists+list_
    cate_ = np.tile(dir_,len(list_))
    cate = np.concatenate([cate,cate_])
    
df["cate"] = cate
df["name"] = lists


## === cell 5
base_path = (
    gcs_path if gcs_path is not None else "/kaggle/input/alaska2-image-steganalysis"
)

df["path"] = [
    str(os.path.join(base_path, cate, name))
    for cate, name in zip(df["cate"], df["name"])
]


def cate_label(x):
    if x["cate"] == "Cover":
        res = 0
    else:
        res = 1
    return res


Test_df = df.query("cate=='Test'").sort_values(by="name")
Train_df = df.query("cate!='Test'")

Train_df["labled"] = df.apply(cate_label, axis=1)


## === cell 6
print("Training set: \n",Train_df["cate"].value_counts())

print('\n', Train_df["path"].head())
print('\n',Train_df["labled"].head())
print('\n',Test_df["path"].head())


## === cell 7


X = Train_df["path"]
y = Train_df["labled"]
z = Test_df["path"]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10)

X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

X_test = np.array(Test_df["path"])


## === cell 8
print(X_test[7])
print(X_train[7])
print(X_val[7])
print(y_train[7])
print(y_val[7])


## === cell 9
def decode_image(filename, label=None, image_size=(512,512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label


## === cell 10
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


## === cell 11
with tpu_strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB3(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        l.Dense(1024, activation="relu",kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        l.Dropout(0.4),
        l.BatchNormalization(),
        l.GlobalAveragePooling2D(),
        l.BatchNormalization(),
        l.Activation('relu'),
        l.Dropout(0.4),
        l.Dense(1),
        l.Dense(1, activation='sigmoid')
    ])
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)    
    model.compile(
        optimizer=opt,
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/812978817.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m         [0ml[0m[0;34m.[0m[0mDense[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mactivation[0m[0;34m=[0m[0;34m'sigmoid'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m     ])
[0;32m---> 18[0;31m     [0mopt[0m [0;34m=[0m [0mAdam[0m[0;34m([0m[0mlr[0m[0;34m=[0m[0;36m0.001[0m[0;34m,[0m [0mbeta_1[0m[0;34m=[0m[0;36m0.9[0m[0;34m,[0m [0mbeta_2[0m[0;34m=[0m[0;36m0.999[0m[0;34m,[0m [0mdecay[0m[0;34m=[0m[0;36m0.01[0m[0;34m,[0m [0mamsgrad[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m     model.compile(
[1;32m     20[0m         [0moptimizer[0m[0;34m=[0m[0mopt[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py[0m in [0;36m__init__[0;34m(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)[0m
[1;32m     60[0m         [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m     ):
[0;32m---> 62[0;31m         super().__init__(
[0m[1;32m     63[0m             [0mlearning_rate[0m[0;34m=[0m[0mlearning_rate[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     64[0m             [0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py[0m in [0;36m__init__[0;34m(self, *args, **kwargs)[0m
[1;32m     19[0m [0;32mclass[0m [0mTFOptimizer[0m[0;34m([0m[0mKerasAutoTrackable[0m[0;34m,[0m [0mbase_optimizer[0m[0;34m.[0m[0mBaseOptimizer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 21[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m         [0mself[0m[0;34m.[0m[0m_distribution_strategy[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mdistribute[0m[0;34m.[0m[0mget_strategy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py[0m in [0;36m__init__[0;34m(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)[0m
[1;32m     88[0m             )
[1;32m     89[0m         [0;32mif[0m [0mkwargs[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 90[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34mf"Argument(s) not recognized: {kwargs}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     91[0m [0;34m[0m[0m
[1;32m     92[0m         [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Argument(s) not recognized: {'lr': 0.001}

## === cell 12
model.summary()

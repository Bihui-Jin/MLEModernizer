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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
tf_keras==2.18.0

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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_DISABLE_UPB"] = "1"

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

import numpy as np


## === cell 1
import zipfile
with zipfile.ZipFile("../input/aerial-cactus-identification/train.zip","r") as z:
    z.extractall(".")

with zipfile.ZipFile("../input/aerial-cactus-identification/test.zip","r") as z:
    z.extractall(".")  


## === cell 2

train_dir='train'
test_dir='test'
train=pd.read_csv('../input/aerial-cactus-identification/train.csv')

test_df=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')


## === cell 3
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


## === cell 4
import tensorflow as tf


## === cell 5
train.head(5)


## === cell 6
train.has_cactus=train.has_cactus.astype(str)


## === cell 7
train.shape


## === cell 8
train['has_cactus'].value_counts()


## === cell 9

datagen=ImageDataGenerator(rescale=1./255, rotation_range=20,horizontal_flip=True, shear_range = 0.2,zoom_range = 0.2)


## === cell 10

split_idx = min(15001, len(train) - 1)
val_size = max(1, len(train) - split_idx)

class_counts = train["has_cactus"].value_counts()
unique_classes = class_counts.index.tolist()

if len(unique_classes) >= 2 and val_size >= 2:
    base_samples = train.groupby("has_cactus", group_keys=False).apply(
        lambda x: x.sample(n=1, random_state=42)
    )

    remaining_df = train.drop(index=base_samples.index, errors="ignore")
    remaining_val_n = val_size - len(base_samples)

    if remaining_val_n > 0 and len(remaining_df) > 0:
        remaining_counts = remaining_df["has_cactus"].value_counts()
        remaining_total = len(remaining_df)

        per_class_n = {
            c: int(
                round(remaining_val_n * remaining_counts.get(c, 0) / remaining_total)
            )
            for c in unique_classes
        }

        current_sum = sum(per_class_n.values())
        if current_sum != remaining_val_n:
            largest_class = remaining_counts.idxmax()
            per_class_n[largest_class] = max(
                0, per_class_n.get(largest_class, 0) + (remaining_val_n - current_sum)
            )

        extra_parts = []
        for c, n in per_class_n.items():
            if n <= 0:
                continue
            subset = remaining_df[remaining_df["has_cactus"] == c]
            if len(subset) > 0:
                extra_parts.append(
                    subset.sample(n=min(n, len(subset)), random_state=42)
                )

        extra_df = (
            pd.concat(extra_parts, axis=0)
            if extra_parts
            else remaining_df.sample(
                n=min(remaining_val_n, len(remaining_df)), random_state=42
            )
        )

        if len(extra_df) < remaining_val_n:
            need = remaining_val_n - len(extra_df)
            pool = remaining_df.drop(index=extra_df.index, errors="ignore")
            if len(pool) > 0:
                extra_df = pd.concat(
                    [extra_df, pool.sample(n=min(need, len(pool)), random_state=42)],
                    axis=0,
                )

        val_df = pd.concat([base_samples, extra_df], axis=0)
    else:
        val_df = base_samples

    if len(val_df) > val_size:
        drop_n = len(val_df) - val_size
        droppable_idx = val_df.index[
            val_df.duplicated(subset=["has_cactus"], keep="first")
        ]
        if len(droppable_idx) >= drop_n:
            drop_idx = droppable_idx[:drop_n]
        else:
            protected = set(
                val_df.drop_duplicates(
                    subset=["has_cactus"], keep="first"
                ).index.tolist()
            )
            candidates = [i for i in val_df.index[::-1] if i not in protected]
            drop_idx = candidates[:drop_n]
        val_df = val_df.drop(index=drop_idx)
    elif len(val_df) < val_size:
        need = val_size - len(val_df)
        pool = train.drop(index=val_df.index, errors="ignore")
        if len(pool) > 0:
            val_df = pd.concat(
                [val_df, pool.sample(n=min(need, len(pool)), random_state=42)], axis=0
            )

else:
    val_df = train.groupby("has_cactus", group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(round(val_size * len(x) / len(train)))),
            random_state=42,
        )
    )
    if len(val_df) > val_size:
        val_df = val_df.sample(n=val_size, random_state=42)
    elif len(val_df) < val_size:
        remaining = train.drop(index=val_df.index, errors="ignore")
        if len(remaining) > 0:
            extra_n = min(val_size - len(val_df), len(remaining))
            extra = remaining.sample(n=extra_n, random_state=42)
            val_df = pd.concat([val_df, extra], axis=0)

train_df = train.drop(index=val_df.index, errors="ignore").reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

if len(unique_classes) >= 2 and val_df["has_cactus"].nunique() < 2:
    missing = [c for c in unique_classes if c not in set(val_df["has_cactus"].unique())]
    if missing:
        miss_class = missing[0]
        donor = train_df[train_df["has_cactus"] == miss_class]
        if len(donor) > 0:
            moved = donor.sample(n=1, random_state=42)
            train_df = train_df.drop(index=moved.index).reset_index(drop=True)
            val_df = pd.concat(
                [val_df, moved.reset_index(drop=True)], axis=0
            ).reset_index(drop=True)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    target_size=(32, 32),
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    target_size=(32, 32),
)


## === cell 11
with tf.device('/GPU:0'):
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu', input_shape = (32,32,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu', input_shape = (32,32,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))


## === cell 12
model.summary()


## === cell 13
model.compile(loss='binary_crossentropy',optimizer='Adamax',metrics=['acc'])


## === cell 14
epochs = 10


def _safe_len(gen):
    try:
        return len(gen)
    except Exception:
        return 0


def _as_tf_dataset_from_keras_generator(gen):
    xb, yb = next(iter(gen))
    output_signature = (
        tf.TensorSpec(shape=xb.shape, dtype=tf.as_dtype(xb.dtype)),
        tf.TensorSpec(shape=yb.shape, dtype=tf.as_dtype(yb.dtype)),
    )

    def _gen():
        for batch in gen:
            yield batch

    return tf.data.Dataset.from_generator(_gen, output_signature=output_signature)


train_len = _safe_len(train_generator)
val_len = _safe_len(validation_generator)

steps_per_epoch = max(1, train_len)
validation_steps = max(1, val_len)

x_train = train_generator
x_val = validation_generator

if train_len == 0:
    x_train = _as_tf_dataset_from_keras_generator(train_generator)
    steps_per_epoch = 1

if val_len == 0:
    x_val = _as_tf_dataset_from_keras_generator(validation_generator)
    validation_steps = 1

history = model.fit(
    x_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=x_val,
    validation_steps=validation_steps,
)


## === cell 15
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['acc'],'blue')
plt.plot(history.history['val_acc'],'orange')
plt.xticks(np.arange(0, 10, 1))
plt.yticks(np.arange(0.8,1.1,.05))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()
 
plt.figure(1)
plt.plot(history.history['loss'],'blue')
plt.plot(history.history['val_loss'],'orange')
plt.xticks(np.arange(0, 10, 1))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()


## === cell 16
model_vg=VGG16(weights='imagenet',include_top=False,input_shape=(32, 32, 3))
model_vg.summary()


## === cell 17
model_vg.trainable = False


## === cell 18
from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D,Conv3D,MaxPooling2D,AveragePooling2D,BatchNormalization
model2 = models.Sequential()
model2.add(model_vg)

model2.add(Flatten())
model2.add(Dense(256, use_bias=True))
model2.add(BatchNormalization())
model2.add(Activation("relu"))
model2.add(Dropout(0.5))
model2.add(Dense(64,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(16, activation='tanh'))
model2.add(Dense(1, activation='sigmoid'))


## === cell 19
 
with tf.device('/GPU:0'):
    model2.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['acc'])


## === cell 20
with tf.device("/GPU:0"):

    def _safe_len(gen):
        try:
            return len(gen)
        except Exception:
            return 0

    def _as_tf_dataset_from_keras_generator(gen):
        xb, yb = next(iter(gen))
        output_signature = (
            tf.TensorSpec(shape=xb.shape, dtype=tf.as_dtype(xb.dtype)),
            tf.TensorSpec(shape=yb.shape, dtype=tf.as_dtype(yb.dtype)),
        )

        def _gen():
            for batch in gen:
                yield batch

        return tf.data.Dataset.from_generator(_gen, output_signature=output_signature)

    train_len = _safe_len(train_generator)
    val_len = _safe_len(validation_generator)

    x_train = train_generator
    x_val = validation_generator

    if train_len == 0:
        x_train = _as_tf_dataset_from_keras_generator(train_generator)

    if val_len == 0:
        x_val = _as_tf_dataset_from_keras_generator(validation_generator)

    history_vgg = model2.fit(
        x_train,
        steps_per_epoch=450,
        epochs=15,
        validation_data=x_val,
        validation_steps=450,
    )


## === cell 21
fig = plt.figure(figsize=(12,8))
plt.plot(history_vgg.history['acc'],'blue')
plt.plot(history_vgg.history['val_acc'],'orange')
plt.xticks(np.arange(0, 15, 1))
plt.yticks(np.arange(0.8,1.1,.05))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()
 
plt.figure(1)
plt.plot(history_vgg.history['loss'],'blue')
plt.plot(history_vgg.history['val_loss'],'orange')
plt.xticks(np.arange(0, 15, 1))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()


## === cell 22
test_dir='test/'

X_test = []
imges = test_df['id'].values

for img_id in imges:
    X_test.append(cv2.imread(test_dir + img_id )) 
                  
X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255

y_test_pred  = model.predict_proba(X_test)

test_df['has_cactus'] = y_test_pred
test_df.to_csv('submission.csv', index=False)


## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3145816284.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     12[0m [0mX_test[0m [0;34m/=[0m [0;36m255[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m
[0;32m---> 14[0;31m [0my_test_pred[0m  [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;34m[0m[0m
[1;32m     16[0m [0mtest_df[0m[0;34m[[0m[0;34m'has_cactus'[0m[0;34m][0m [0;34m=[0m [0my_test_pred[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Sequential' object has no attribute 'predict_proba'

# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd

train_labels = pd.read_csv(
    "../input/histopathologic-cancer-detection/train_labels.csv", dtype=str
)
print(train_labels.shape)



## === cell 1
train_labels.head()



## === cell 2
train_labels.dtypes



## === cell 3
train_labels["label"] = train_labels["label"].astype(float)



## === cell 4
import os

print(len(os.listdir("../input/histopathologic-cancer-detection/train/")))
print(len(os.listdir("../input/histopathologic-cancer-detection/test/")))



## === cell 5
len(train_labels)



## === cell 6
train_labels["label"].value_counts()



## === cell 8
train_labels_pos = train_labels[train_labels["label"] == 1]
train_labels_neg = train_labels[train_labels["label"] == 0]



## === cell 9
train_labels_neg = train_labels_neg.sample(
    n=train_labels_pos.shape[0], random_state=12345
)



## === cell 10
print(train_labels_neg.shape[0])
print(train_labels_pos.shape[0])



## === cell 11
train_labels_balanced = (
    pd.concat([train_labels_neg, train_labels_pos])
    .sample(frac=1, random_state=12345)
    .reset_index(drop=True)
)
train_labels_balanced.head()



## === cell 12
train_labels_balanced.shape



## === cell 13
train_labels_balanced["label"].value_counts()



## === cell 18
from sklearn.model_selection import train_test_split



## === cell 19
train_df, valid_df = train_test_split(
    train_labels_balanced,
    test_size=0.25,
    random_state=1234,
    stratify=train_labels_balanced.label,
)



## === cell 20
import os
import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    from google.protobuf import message_factory as _message_factory

    MF = getattr(_message_factory, "MessageFactory", None)
    if MF is not None:
        if getattr(MF, "GetPrototype", None) is None:
            _get_msg_cls = getattr(_message_factory, "GetMessageClass", None)

            def _GetPrototype(self, descriptor):
                if _get_msg_cls is not None:
                    return _get_msg_cls(descriptor)
                pool = getattr(self, "pool", None)
                if pool is not None and hasattr(pool, "GetMessageClass"):
                    return pool.GetMessageClass(descriptor)
                raise AttributeError(
                    "No compatible GetPrototype/GetMessageClass implementation available."
                )

            try:
                setattr(MF, "GetPrototype", _GetPrototype)
            except Exception:
                pass
except Exception:
    pass

import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import PReLU
from keras.initializers import Constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


## === cell 21
train_df = train_df.copy()
valid_df = valid_df.copy()
train_df["id"] = train_df["id"] + ".tif"
valid_df["id"] = valid_df["id"] + ".tif"



## === cell 22
train_df["label"] = train_df["label"].astype(str)
valid_df["label"] = valid_df["label"].astype(str)



## === cell 23
train_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../input/histopathologic-cancer-detection/train/",
    x_col="id",
    y_col="label",
    batch_size=64,
    seed=SEED,
    shuffle=True,
    class_mode="binary",
    target_size=(96, 96),
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory="../input/histopathologic-cancer-detection/train/",
    x_col="id",
    y_col="label",
    batch_size=64,
    seed=SEED,
    shuffle=True,
    class_mode="binary",
    target_size=(96, 96),
)



## === cell 24
pass



## === cell 25
model4 = Sequential()
model4.add(Conv2D(32, (3, 3), padding="same", input_shape=(96, 96, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Flatten())
model4.add(Dropout(0.25))
model4.add(Dense(512))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(256))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(64))
model4.add(PReLU(alpha_initializer=Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(1, activation="sigmoid"))
opt = tf.keras.optimizers.RMSprop(0.001)
model4.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])



## === cell 26
model4.summary()



## === cell 27
import math

STEP_SIZE_TRAIN = math.ceil(train_generator.n / train_generator.batch_size)
STEP_SIZE_VALID = math.ceil(valid_generator.n / valid_generator.batch_size)

history4 = model4.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=30,
    verbose=1,
)


## === cell 29
test_set = sorted(os.listdir("../input/histopathologic-cancer-detection/test/"))



## === cell 30
test_df = pd.DataFrame({"id": test_set})
test_df.head()



## === cell 31
test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="../input/histopathologic-cancer-detection/test/",
    x_col="id",
    batch_size=64,
    seed=SEED,
    shuffle=False,
    class_mode=None,
    target_size=(96, 96),
)



## === cell 32
STEP_SIZE_TEST = math.ceil(test_generator.n / test_generator.batch_size)

preds = model4.predict(
    test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1,
    workers=max(2, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=16,
)



## === cell 33
preds = preds.reshape(-1)
predictions = (preds >= 0.5).astype(np.int8)
predictions[:10]



## === cell 34
submission = test_df.copy()
submission["id"] = submission["id"].str[:-4]
submission["label"] = predictions[: len(submission)]
submission.head()



## === cell 35
submission.to_csv("submission.csv", index=False)

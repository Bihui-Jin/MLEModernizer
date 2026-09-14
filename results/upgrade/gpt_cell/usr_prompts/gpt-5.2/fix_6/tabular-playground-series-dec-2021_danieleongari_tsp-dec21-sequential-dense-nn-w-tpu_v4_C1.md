# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import gc
import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"

_train_head = pd.read_csv(TRAIN_PATH, nrows=0)
_test_head = pd.read_csv(TEST_PATH, nrows=0)

dtype_train = {}
for c in _train_head.columns:
    if c in ("Id", "Cover_Type"):
        dtype_train[c] = np.int32
    elif c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dtype_train[c] = np.int8
    else:
        dtype_train[c] = np.float32

dtype_test = {}
for c in _test_head.columns:
    if c == "Id":
        dtype_test[c] = np.int32
    elif c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dtype_test[c] = np.int8
    else:
        dtype_test[c] = np.float32

train = pd.read_csv(TRAIN_PATH, dtype=dtype_train)
test = pd.read_csv(TEST_PATH, dtype=dtype_test)
train




## === cell 2
pass




## === cell 3
train = train.drop(columns=["Soil_Type7", "Soil_Type15"])
test = test.drop(columns=["Soil_Type7", "Soil_Type15"])




## === cell 4
aspect = train["Aspect"].to_numpy(dtype=np.float32, copy=False)
train["Aspect_cos"] = np.cos(np.deg2rad(aspect))
train["Aspect_sin"] = np.sin(np.deg2rad(aspect))
train = train.drop(columns=["Aspect"])

aspect_t = test["Aspect"].to_numpy(dtype=np.float32, copy=False)
test["Aspect_cos"] = np.cos(np.deg2rad(aspect_t))
test["Aspect_sin"] = np.sin(np.deg2rad(aspect_t))
test = test.drop(columns=["Aspect"])




## === cell 5
for df in (train, test):
    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        arr = df[col].to_numpy(copy=False)
        np.clip(arr, 0, 255, out=arr)




## === cell 6
h = train["Horizontal_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
v = train["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
ah = np.abs(h)
av = np.abs(v)
train["Sum_Hydrology"] = ah + av
train["Sub_Hydrology"] = ah - av

h = test["Horizontal_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
v = test["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
ah = np.abs(h)
av = np.abs(v)
test["Sum_Hydrology"] = ah + av
test["Sub_Hydrology"] = ah - av




## === cell 7
mask = train["Cover_Type"].to_numpy(copy=False) != 5
train = train.loc[mask].reset_index(drop=True)




## === cell 8
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
target = le.fit_transform(train["Cover_Type"].to_numpy(copy=False))
target = to_categorical(target)

train_ids = train["Id"].copy()
test_ids = test["Id"].copy()

train = train.drop(columns=["Cover_Type", "Id"])
test = test.drop(columns=["Id"])

gc.collect()




## === cell 9
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()

X_train = train.to_numpy(dtype=np.float32, copy=False)
X_test = test.to_numpy(dtype=np.float32, copy=False)

X_train = rb.fit_transform(X_train).astype(np.float32, copy=False)
X_test = rb.transform(X_test).astype(np.float32, copy=False)

train = X_train
test = X_test




## === cell 10
gc.collect()




## === cell 11
shapes = {
    "nsamples": train.shape[0],
    "nfeatures": train.shape[1],
    "ncategories": target.shape[1],
}
print(shapes)




## === cell 12
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Number of replicas:", strategy.num_replicas_in_sync)
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using strategy:", type(strategy).__name__)




## === cell 13
CURRENT_MODEL = 2




## === cell 14
if CURRENT_MODEL == 1:
    with strategy.scope():  # necessary for using the TPU
        model1 = keras.models.Sequential(
            [
                keras.layers.Input((shapes["nfeatures"],)),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(shapes["ncategories"], activation="softmax"),
            ]
        )
        model1.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )




## === cell 15
if CURRENT_MODEL == 1:
    n = train.shape[0]
    val_size = int(0.2 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model1.fit(
        x=x_tr,
        y=y_tr,
        epochs=20,
        batch_size=128 * getattr(strategy, "num_replicas_in_sync", 1),
        validation_data=(x_val, y_val),
        callbacks=[earlystop],
        verbose=2,
    )




## === cell 16
pass




## === cell 17
if CURRENT_MODEL == 2:
    dropout_rate = 0.1
    with strategy.scope():  # necessary for using the TPU
        model2 = keras.models.Sequential(
            [
                keras.layers.Input((shapes["nfeatures"],)),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dense(shapes["ncategories"], activation="softmax"),
            ]
        )
        model2.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )




## === cell 18
if CURRENT_MODEL == 2:
    n = train.shape[0]
    val_size = int(0.05 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model2.fit(
        x=x_tr,
        y=y_tr,
        epochs=20,
        batch_size=128 * getattr(strategy, "num_replicas_in_sync", 1),
        validation_data=(x_val, y_val),
        verbose=2,
    )




## === cell 19
pass




## === cell 20
if CURRENT_MODEL == 1:
    test_pred = model1.predict(test, verbose=0, batch_size=4096)
elif CURRENT_MODEL == 2:
    test_pred = model2.predict(test, verbose=0, batch_size=4096)




## === cell 21
sub = pd.DataFrame(
    {
        "Id": test_ids.values,
        "Cover_Type": le.inverse_transform(np.argmax(test_pred, axis=1)),
    }
)
sub.to_csv("submission.csv", index=False)
display(sub.head())
print("Saved submission.csv with shape:", sub.shape)

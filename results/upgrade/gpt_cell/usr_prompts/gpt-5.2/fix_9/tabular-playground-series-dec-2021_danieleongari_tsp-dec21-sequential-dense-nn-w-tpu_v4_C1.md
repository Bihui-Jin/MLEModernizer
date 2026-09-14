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


def _make_dtype_map(cols, is_train: bool):
    dtype = {}
    for c in cols:
        if c == "Id" or (is_train and c == "Cover_Type"):
            dtype[c] = np.int32
        elif c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
            dtype[c] = np.int8
        else:
            dtype[c] = np.float32
    return dtype


dtype_train = _make_dtype_map(_train_head.columns, is_train=True)
dtype_test = _make_dtype_map(_test_head.columns, is_train=False)

train = pd.read_csv(TRAIN_PATH, dtype=dtype_train, engine="c", low_memory=False)
test = pd.read_csv(TEST_PATH, dtype=dtype_test, engine="c", low_memory=False)
train



## === cell 2
pass



## === cell 3
train = train.drop(columns=["Soil_Type7", "Soil_Type15"])
test = test.drop(columns=["Soil_Type7", "Soil_Type15"])

train_ids = train["Id"].to_numpy(copy=False)
test_ids = test["Id"].to_numpy(copy=False)

mask = train["Cover_Type"].to_numpy(copy=False) != 5
train = train.loc[mask].reset_index(drop=True)




## === cell 4
def _build_matrix(df: pd.DataFrame, is_train: bool):
    y_raw = df["Cover_Type"].to_numpy(copy=False) if is_train else None

    drop_cols = ["Id"] + (["Cover_Type"] if is_train else [])
    feat_cols = [c for c in df.columns if c not in drop_cols]

    aspect_idx = feat_cols.index("Aspect")
    base_cols_no_aspect = feat_cols[:aspect_idx] + feat_cols[aspect_idx + 1 :]

    X_base = df[base_cols_no_aspect].to_numpy(dtype=np.float32, copy=False)

    aspect = df["Aspect"].to_numpy(dtype=np.float32, copy=False)
    aspect_rad = np.deg2rad(aspect)
    aspect_cos = np.cos(aspect_rad, dtype=np.float32)
    aspect_sin = np.sin(aspect_rad, dtype=np.float32)

    hs_cols = ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    for col in hs_cols:
        try:
            j = base_cols_no_aspect.index(col)
        except ValueError:
            j = None
        if j is not None:
            np.clip(X_base[:, j], 0.0, 255.0, out=X_base[:, j])

    h = df["Horizontal_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
    v = df["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
    ah = np.abs(h)
    av = np.abs(v)
    sum_h = ah + av
    sub_h = ah - av

    n = X_base.shape[0]
    X = np.empty((n, X_base.shape[1] + 4), dtype=np.float32)
    X[:, : X_base.shape[1]] = X_base
    k = X_base.shape[1]
    X[:, k + 0] = aspect_cos
    X[:, k + 1] = aspect_sin
    X[:, k + 2] = sum_h
    X[:, k + 3] = sub_h

    final_cols = base_cols_no_aspect + [
        "Aspect_cos",
        "Aspect_sin",
        "Sum_Hydrology",
        "Sub_Hydrology",
    ]
    return X, y_raw, final_cols


X_train_raw, y_raw, final_cols = _build_matrix(train, is_train=True)
X_test_raw, _, _ = _build_matrix(test, is_train=False)

del train, test
gc.collect()



## === cell 5
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
)
train = scaler.fit_transform(X_train_raw).astype(np.float32, copy=False)
test = scaler.transform(X_test_raw).astype(np.float32, copy=False)

del X_train_raw, X_test_raw
gc.collect()



## === cell 6
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
target = le.fit_transform(y_raw)
target = to_categorical(target)

del y_raw
gc.collect()



## === cell 7
gc.collect()



## === cell 8
shapes = {
    "nsamples": train.shape[0],
    "nfeatures": train.shape[1],
    "ncategories": target.shape[1],
}
print(shapes)



## === cell 9
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



## === cell 10
CURRENT_MODEL = 2



## === cell 11
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



## === cell 12
if CURRENT_MODEL == 1:
    n = train.shape[0]
    val_size = int(0.2 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    batch_size = 128 * getattr(strategy, "num_replicas_in_sync", 1)

    x_tr_tf = tf.convert_to_tensor(x_tr)
    y_tr_tf = tf.convert_to_tensor(y_tr)
    x_val_tf = tf.convert_to_tensor(x_val)
    y_val_tf = tf.convert_to_tensor(y_val)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr_tf, y_tr_tf))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val_tf, y_val_tf))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model1.fit(
        x=train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=[earlystop],
        verbose=2,
    )



## === cell 13
pass



## === cell 14
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



## === cell 15
if CURRENT_MODEL == 2:
    n = train.shape[0]
    val_size = int(0.05 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    batch_size = 128 * getattr(strategy, "num_replicas_in_sync", 1)

    x_tr_tf = tf.convert_to_tensor(x_tr)
    y_tr_tf = tf.convert_to_tensor(y_tr)
    x_val_tf = tf.convert_to_tensor(x_val)
    y_val_tf = tf.convert_to_tensor(y_val)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr_tf, y_tr_tf))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val_tf, y_val_tf))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model2.fit(
        x=train_ds,
        epochs=20,
        validation_data=val_ds,
        verbose=2,
    )



## === cell 16
pass



## === cell 17
if CURRENT_MODEL == 1:
    test_pred = model1.predict(test, verbose=0, batch_size=4096)
elif CURRENT_MODEL == 2:
    test_pred = model2.predict(test, verbose=0, batch_size=4096)



## === cell 18
sub = pd.DataFrame(
    {
        "Id": test_ids,
        "Cover_Type": le.inverse_transform(np.argmax(test_pred, axis=1)),
    }
)
sub.to_csv("submission.csv", index=False)
display(sub.head())
print("Saved submission.csv with shape:", sub.shape)

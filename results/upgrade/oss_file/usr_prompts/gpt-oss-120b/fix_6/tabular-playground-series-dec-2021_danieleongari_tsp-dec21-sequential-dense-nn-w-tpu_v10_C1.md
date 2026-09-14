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
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
        input/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 2 other files
```

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> input/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> input/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> working/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau,
        EarlyStopping,
        ModelCheckpoint,
    )
except Exception as e:
    print("TensorFlow import failed, proceeding without it:", e)
    tf = None
    keras = None




## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
train.head()




## === cell 2
train.describe().T




## === cell 3
test.describe().T




## === cell 4
for df in [train, test]:
    df.drop(columns=["Id", "Soil_Type7", "Soil_Type15"], inplace=True, errors="ignore")




## === cell 5
for df in [train, test]:
    df["Aspect_cos"] = np.cos(np.radians(df["Aspect"]))
    df["Aspect_sin"] = np.sin(np.radians(df["Aspect"]))
    df.drop(columns=["Aspect"], inplace=True, errors="ignore")




## === cell 6
for df in [train, test]:
    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        df[col] = df[col].clip(lower=0, upper=255)




## === cell 7
for df in [train, test]:
    df["Sum_Hydrology"] = np.abs(df["Horizontal_Distance_To_Hydrology"]) + np.abs(
        df["Vertical_Distance_To_Hydrology"]
    )
    df["Sub_Hydrology"] = np.abs(df["Horizontal_Distance_To_Hydrology"]) - np.abs(
        df["Vertical_Distance_To_Hydrology"]
    )




## === cell 8
for df in [train, test]:
    df["EHiElv"] = df["Horizontal_Distance_To_Roadways"] * df["Elevation"]
    df["EViElv"] = df["Vertical_Distance_To_Hydrology"] * df["Elevation"]
    df["Highwater"] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df["EVDtH"] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df["EHDtH"] = df.Elevation - df.Horizontal_Distance_To_Roadways * 0.2
    df["Euclidean_Distance_to_Hydrolody"] = (
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    ) ** 0.5
    df["Manhattan_Distance_to_Hydrolody"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Vertical_Distance_To_Hydrology"]
    )
    df["Hydro_Fire_1"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Fire_2"] = abs(
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Road_1"] = abs(
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Hydro_Road_2"] = abs(
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["Fire_Road_1"] = abs(
        df["Horizontal_Distance_To_Fire_Points"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Fire_Road_2"] = abs(
        df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["Hillshade_3pm_is_zero"] = (df.Hillshade_3pm == 0).astype(int)




## === cell 9
display(train["Cover_Type"].value_counts())




## === cell 10
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_int = le.fit_transform(train["Cover_Type"])  # integer labels for sklearn
target_onehot = (
    tf.keras.utils.to_categorical(y_int) if tf else None
)  # keep original one‑hot if TF is available
train = train.drop(columns=["Cover_Type"])
gc.collect()




## === cell 11
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()
cols = train.columns
train[cols] = rb.fit_transform(train[cols].values)
test[cols] = rb.transform(test[cols].values)




## === cell 12
def reduce_mem_usage(df, verbose=True):
    """Down‑cast numeric columns to reduce RAM usage."""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


train = reduce_mem_usage(train).values
test = reduce_mem_usage(test).values
gc.collect()




## === cell 13
shapes = {
    "nsamples": train.shape[0],
    "nfeatures": train.shape[1],
    "ncategories": len(le.classes_),
}
print(shapes)




## === cell 14
class _DummyStrategy:
    num_replicas_in_sync = 1

    def scope(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


strategy = _DummyStrategy()




## === cell 15
CURRENT_MODEL = 2




## === cell 16
if CURRENT_MODEL == 1:
    with strategy.scope():
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




## === cell 17
if CURRENT_MODEL == 1:
    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model1.fit(
        x=train,
        y=target_onehot,
        epochs=20,
        batch_size=128 * strategy.num_replicas_in_sync,
        validation_split=0.2,
        callbacks=[earlystop],
    )




## === cell 18
if CURRENT_MODEL == 1:
    pd.DataFrame(model1.history.history).plot(
        subplots=True, sharex=True, figsize=[15, 8], grid=True
    )
    plt.show()




## === cell 19
if CURRENT_MODEL == 2:
    dropout_rate = 0.1
    with strategy.scope():
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




## === cell 20
if CURRENT_MODEL == 2:
    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model2.fit(
        x=train,
        y=target_onehot,
        epochs=10,
        batch_size=128 * strategy.num_replicas_in_sync,
        validation_split=0.2,  # provide a validation split so EarlyStopping works
        callbacks=[earlystop],
    )




## === cell 21
if CURRENT_MODEL == 2:
    pd.DataFrame(model2.history.history).plot(
        subplots=True, sharex=True, figsize=[15, 8], grid=True
    )
    plt.show()




## === cell 22
if CURRENT_MODEL == 3:
    dropout_rate = 0.1
    with strategy.scope():
        model2 = keras.models.Sequential(
            [
                keras.layers.Input((shapes["nfeatures"],)),
                keras.layers.Dense(
                    200, kernel_initializer="lecun_normal", use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("selu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(
                    200, kernel_initializer="lecun_normal", use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("selu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(
                    200, kernel_initializer="lecun_normal", use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("selu"),
                keras.layers.Dense(shapes["ncategories"], activation="softmax"),
            ]
        )
        model2.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )




## === cell 23
if CURRENT_MODEL == 3:
    model2.fit(
        x=train,
        y=target_onehot,
        epochs=10,
        batch_size=128 * strategy.num_replicas_in_sync,
    )




## === cell 24
if CURRENT_MODEL == 1:
    test_pred = model1.predict(
        test, batch_size=128 * strategy.num_replicas_in_sync, verbose=1
    )
elif CURRENT_MODEL == 2:
    test_pred = model2.predict(
        test, batch_size=128 * strategy.num_replicas_in_sync, verbose=1
    )
elif CURRENT_MODEL == 3:
    test_pred = model2.predict(
        test, batch_size=128 * strategy.num_replicas_in_sync, verbose=1
    )




## === cell 25
if CURRENT_MODEL == 0:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(train, y_int)
    test_pred_int = rf.predict(test)
    sub = pd.read_csv(
        "../input/tabular-playground-series-dec-2021/sample_submission.csv"
    )
    sub["Cover_Type"] = le.inverse_transform(test_pred_int)
    sub.to_csv("submission.csv", index=False)
    display(sub.head())
else:
    sub = pd.read_csv(
        "../input/tabular-playground-series-dec-2021/sample_submission.csv"
    )
    sub["Cover_Type"] = le.inverse_transform(np.argmax(test_pred, axis=1))
    sub.to_csv("submission.csv", index=False)
    display(sub.head())

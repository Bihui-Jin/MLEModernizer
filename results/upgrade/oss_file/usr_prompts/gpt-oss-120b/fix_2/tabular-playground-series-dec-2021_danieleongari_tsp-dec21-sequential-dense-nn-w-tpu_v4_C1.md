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
    tf = None
    keras = None
    print("TensorFlow import failed, proceeding with scikit‑learn model:", e)



## === cell 1
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
train




## === cell 2
train.describe().T




## === cell 3
train = train.drop(columns=["Id", "Soil_Type7", "Soil_Type15"])
test = test.drop(columns=["Id", "Soil_Type7", "Soil_Type15"])




## === cell 4
train["Aspect_cos"] = np.cos(np.radians(train["Aspect"]))
train["Aspect_sin"] = np.sin(np.radians(train["Aspect"]))
train = train.drop(columns=["Aspect"])
test["Aspect_cos"] = np.cos(np.radians(test["Aspect"]))
test["Aspect_sin"] = np.sin(np.radians(test["Aspect"]))
test = test.drop(columns=["Aspect"])




## === cell 5
for df in (train, test):
    for col in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]:
        df[col] = df[col].clip(lower=0, upper=255)




## === cell 6
train["Sum_Hydrology"] = np.abs(train["Horizontal_Distance_To_Hydrology"]) + np.abs(
    train["Vertical_Distance_To_Hydrology"]
)
train["Sub_Hydrology"] = np.abs(train["Horizontal_Distance_To_Hydrology"]) - np.abs(
    train["Vertical_Distance_To_Hydrology"]
)

test["Sum_Hydrology"] = np.abs(test["Horizontal_Distance_To_Hydrology"]) + np.abs(
    test["Vertical_Distance_To_Hydrology"]
)
test["Sub_Hydrology"] = np.abs(test["Horizontal_Distance_To_Hydrology"]) - np.abs(
    test["Vertical_Distance_To_Hydrology"]
)




## === cell 7
print("\n*** Original targets count:")
display(train["Cover_Type"].value_counts())
train = train.drop(index=train[train["Cover_Type"] == 5].index).reset_index(drop=True)
print("\n*** Processed targets count:")
display(train["Cover_Type"].value_counts())




## === cell 8
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import (
    to_categorical,
)  # will be used only if tf is available

le = LabelEncoder()
target_raw = le.fit_transform(train["Cover_Type"])
target = (
    to_categorical(target_raw) if tf is not None else None
)  # keep shape info for later

train = train.drop(columns=["Cover_Type"])

gc.collect()




## === cell 9
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()

cols = train.columns

train[cols] = rb.fit_transform(train[cols].values)
test[cols] = rb.transform(test[cols].values)




## === cell 10
def reduce_mem_usage(df, verbose=True):
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
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


train = reduce_mem_usage(train).values
test = reduce_mem_usage(test).values

gc.collect()




## === cell 11
if tf is not None:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("TPU strategy initialized")
    except Exception as e:
        print("TPU not available, using default strategy:", e)
        strategy = tf.distribute.get_strategy()
else:
    strategy = None
    print("TensorFlow not available, will use scikit‑learn model.")




## === cell 12
CURRENT_MODEL = 2  # we will use the scikit‑learn RandomForest implementation




## === cell 13
if CURRENT_MODEL == 2:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model2 = rf
    print("RandomForest model prepared.")




## === cell 14
if CURRENT_MODEL == 2:
    model2.fit(train, target_raw)
    print("Training completed.")




## === cell 15
if CURRENT_MODEL == 2:
    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(
        train, target_raw, test_size=0.05, random_state=42, stratify=target_raw
    )
    rf_tmp = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42
    )
    rf_tmp.fit(X_tr, y_tr)
    acc = rf_tmp.score(X_va, y_va)
    print(f"Quick validation accuracy: {acc:.5f}")




## === cell 16
if CURRENT_MODEL == 2:
    test_proba = model2.predict_proba(test)
else:
    test_proba = model2.predict(test, verbose=1)




## === cell 17
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
sub["Cover_Type"] = le.inverse_transform(np.argmax(test_proba, axis=1))
sub.to_csv("submission.csv", index=False)
display(sub.head())
print("Submission file written to submission.csv")

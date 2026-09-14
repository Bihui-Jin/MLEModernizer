# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.95427

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random, os, gc, warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = type("tfa", (), {})
    tfa.layers = type("layers", (), {"WeightNormalization": lambda layer, **kw: layer})
from tensorflow.keras import layers, callbacks, backend as K
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Concatenate
from tensorflow.random import set_seed

set_seed(42)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_csv("./input/tabular-playground-series-dec-2021/train.csv")
test_data = pd.read_csv("./input/tabular-playground-series-dec-2021/test.csv")
sample = pd.read_csv("./input/tabular-playground-series-dec-2021/sample_submission.csv")
train_data = train_data[train_data.Cover_Type != 5]  # drop class 5
train = train_data.drop("Id", axis=1)  # drop unused id
test = test_data.drop("Id", axis=1)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1680818047.py in <cell line: 0>()
      1 # Adjust paths to the typical Kaggle environment layout
----> 2 train_data = pd.read_csv("./input/tabular-playground-series-dec-2021/train.csv")
      3 test_data = pd.read_csv("./input/tabular-playground-series-dec-2021/test.csv")
      4 sample = pd.read_csv("./input/tabular-playground-series-dec-2021/sample_submission.csv")
      5 train_data = train_data[train_data.Cover_Type != 5]  # drop class 5

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './input/tabular-playground-series-dec-2021/train.csv'

## === cell 2
train.head(3)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3462641729.py in <cell line: 0>()
----> 1 train.head(3)
      2 

NameError: name 'train' is not defined

## === cell 3
train.describe().style.background_gradient(cmap="RdPu")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/183480767.py in <cell line: 0>()
----> 1 train.describe().style.background_gradient(cmap="RdPu")
      2 

NameError: name 'train' is not defined

## === cell 4
df_var = train.var().reset_index()
df_var.columns = ["feature", "variation"]
df_var.sort_values("variation", ascending=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1990286095.py in <cell line: 0>()
----> 1 df_var = train.var().reset_index()
      2 df_var.columns = ["feature", "variation"]
      3 df_var.sort_values("variation", ascending=True)
      4 

NameError: name 'train' is not defined

## === cell 5
corrMatrix = train.corr(method="pearson", min_periods=1)
corrMatrix.style.background_gradient(axis=None)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3073869060.py in <cell line: 0>()
----> 1 corrMatrix = train.corr(method="pearson", min_periods=1)
      2 corrMatrix.style.background_gradient(axis=None)
      3 

NameError: name 'train' is not defined

## === cell 6
cor_targ = train.corrwith(train["Cover_Type"]).reset_index()
cor_targ.columns = ["feature", "CorrelatioWithTarget"]
cor_targ.sort_values("CorrelatioWithTarget", ascending=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/356112896.py in <cell line: 0>()
----> 1 cor_targ = train.corrwith(train["Cover_Type"]).reset_index()
      2 cor_targ.columns = ["feature", "CorrelatioWithTarget"]
      3 cor_targ.sort_values("CorrelatioWithTarget", ascending=False)
      4 

NameError: name 'train' is not defined

## === cell 7
ax = plt.figure(figsize=(12, 6))
cover_type = train["Cover_Type"].value_counts().sort_index()
sns.barplot(x=cover_type.index, y=cover_type, palette="BuPu_r")
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3501806.py in <cell line: 0>()
      1 ax = plt.figure(figsize=(12, 6))
----> 2 cover_type = train["Cover_Type"].value_counts().sort_index()
      3 sns.barplot(x=cover_type.index, y=cover_type, palette="BuPu_r")
      4 plt.show()
      5 

NameError: name 'train' is not defined

## === cell 8
test.head(3)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2555458980.py in <cell line: 0>()
----> 1 test.head(3)
      2 

NameError: name 'test' is not defined

## === cell 9
test.describe().style.background_gradient(axis=1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2647418329.py in <cell line: 0>()
----> 1 test.describe().style.background_gradient(axis=1)
      2 

NameError: name 'test' is not defined

## === cell 10
plt.figure(figsize=(15, 8))
features = train.columns.values[0:54]
sns.distplot(
    train[features].mean(axis=1), color="red", kde=True, bins=120, label="train"
)
sns.distplot(
    test[features].mean(axis=1), color="darkblue", kde=True, bins=120, label="test"
)
plt.title("Distribution of mean values per row in the train and test data")
plt.legend()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3073736092.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 8))
----> 2 features = train.columns.values[0:54]
      3 sns.distplot(
      4     train[features].mean(axis=1), color="red", kde=True, bins=120, label="train"
      5 )

NameError: name 'train' is not defined

## === cell 11
plt.figure(figsize=(15, 5))
sns.distplot(
    train[features].mean(axis=0), color="orange", kde=True, bins=120, label="train"
)
sns.distplot(
    test[features].mean(axis=0), color="blue", kde=True, bins=120, label="test"
)
plt.title("Distribution of mean values per column in the train and test set")
plt.legend()
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/346072136.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 5))
      2 sns.distplot(
----> 3     train[features].mean(axis=0), color="orange", kde=True, bins=120, label="train"
      4 )
      5 sns.distplot(

NameError: name 'train' is not defined

## === cell 12
plt.figure(figsize=(15, 5))
sns.distplot(
    train[features].std(axis=1), color="#2F4F4F", kde=True, bins=120, label="train"
)
sns.distplot(
    test[features].std(axis=1), color="#FF6347", kde=True, bins=120, label="test"
)
plt.title("Distribution of std per row in the train and test data ")
plt.legend()
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2935170570.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 5))
      2 sns.distplot(
----> 3     train[features].std(axis=1), color="#2F4F4F", kde=True, bins=120, label="train"
      4 )
      5 sns.distplot(

NameError: name 'train' is not defined

## === cell 13
plt.figure(figsize=(15, 5))
sns.distplot(
    train[features].std(axis=0), color="#778899", kde=True, bins=120, label="train"
)
sns.distplot(
    test[features].std(axis=0), color="#800080", kde=True, bins=120, label="test"
)
plt.title("Distribution of std per column in the train and test data")
plt.legend()
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2579884197.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 5))
      2 sns.distplot(
----> 3     train[features].std(axis=0), color="#778899", kde=True, bins=120, label="train"
      4 )
      5 sns.distplot(

NameError: name 'train' is not defined

## === cell 14
num_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
i = 1
plt.figure()
fig, ax = plt.subplots(figsize=(18, 15))
for col in num_cols:
    plt.subplot(5, 3, i)
    sns.distplot(train[col], color="yellow", kde=True, bins=100, label="train")
    sns.distplot(test[col], color="Darkblue", kde=True, bins=100, label="test")
    i += 1
plt.legend()
plt.title(" numirical features Distribution in both train and test data")
plt.show()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4032359180.py in <cell line: 0>()
     16 for col in num_cols:
     17     plt.subplot(5, 3, i)
---> 18     sns.distplot(train[col], color="yellow", kde=True, bins=100, label="train")
     19     sns.distplot(test[col], color="Darkblue", kde=True, bins=100, label="test")
     20     i += 1

NameError: name 'train' is not defined

## === cell 15
train = train.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)
test = test.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3533036817.py in <cell line: 0>()
----> 1 train = train.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)
      2 test = test.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)
      3 

NameError: name 'train' is not defined

## === cell 16
y_target = train["Cover_Type"].copy()  # target variable
X_train = train.copy().drop("Cover_Type", axis=1)  # features




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2188905992.py in <cell line: 0>()
----> 1 y_target = train["Cover_Type"].copy()  # target variable
      2 X_train = train.copy().drop("Cover_Type", axis=1)  # features
      3 
      4 

NameError: name 'train' is not defined

## === cell 17
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




## === cell 18
X_train = reduce_mem_usage(X_train)
test = reduce_mem_usage(test)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1526271897.py in <cell line: 0>()
      1 # Reduce memory usage for training and test features
----> 2 X_train = reduce_mem_usage(X_train)
      3 test = reduce_mem_usage(test)
      4 
      5 

NameError: name 'X_train' is not defined

## === cell 19
def stat_features(df):
    df["r_skew"] = df.skew(axis=1)
    df["r_sum"] = df.sum(axis=1)
    return df




## === cell 20
del train_data
del test_data



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/728340305.py in <cell line: 0>()
----> 1 del train_data
      2 del test_data
      3 

NameError: name 'train_data' is not defined

## === cell 21
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1758792015.py in <cell line: 0>()
      1 scaler = StandardScaler()
----> 2 X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
      3 test[num_cols] = scaler.transform(test[num_cols])
      4 

NameError: name 'X_train' is not defined

## === cell 22
i = 1
plt.figure()
fig, ax = plt.subplots(figsize=(18, 15))
for col in num_cols:
    plt.subplot(4, 4, i)
    sns.distplot(X_train[col], color="yellow", kde=True, bins=120, label="train")
    sns.distplot(test[col], color="Darkblue", kde=True, bins=120, label="test")
    i += 1
plt.legend()
plt.title(
    " numirical featur Distribution after normalization in both train and test data"
)
plt.show()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2591138197.py in <cell line: 0>()
      4 for col in num_cols:
      5     plt.subplot(4, 4, i)
----> 6     sns.distplot(X_train[col], color="yellow", kde=True, bins=120, label="train")
      7     sns.distplot(test[col], color="Darkblue", kde=True, bins=120, label="test")
      8     i += 1

NameError: name 'X_train' is not defined

## === cell 23
label_encod = LabelEncoder()
y_encoded = label_encod.fit_transform(y_target)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2768582372.py in <cell line: 0>()
      1 label_encod = LabelEncoder()
----> 2 y_encoded = label_encod.fit_transform(y_target)
      3 

NameError: name 'y_target' is not defined

## === cell 24
print(y_encoded.shape, y_target.shape, X_train.shape, test.shape)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2522046641.py in <cell line: 0>()
----> 1 print(y_encoded.shape, y_target.shape, X_train.shape, test.shape)
      2 

NameError: name 'y_encoded' is not defined

## === cell 25
n_classes = 6


def get_model(X_train):
    inputs = layers.Input(shape=(X_train.shape[1],))
    hidden = layers.Dense(
        units=350, kernel_initializer="lecun_normal", activation="selu"
    )(inputs)
    flatten = layers.Flatten()(hidden)
    dropout = layers.Dropout(0.2)(flatten)
    hidden1 = tfa.layers.WeightNormalization(
        layers.Dense(units=128, activation="selu", kernel_initializer="lecun_normal")
    )(dropout)
    dropout1 = layers.Dropout(0.2)(layers.Concatenate()([hidden1, dropout]))
    hidden1b = tfa.layers.WeightNormalization(
        layers.Dense(units=64, activation="selu")
    )(dropout1)
    dropout2 = layers.Dropout(0.3)(layers.Concatenate()([dropout, dropout1, hidden1b]))
    hidden2 = tfa.layers.WeightNormalization(layers.Dense(units=32, activation="selu"))(
        dropout2
    )
    output = layers.Dense(n_classes, activation="softmax")(hidden2)
    model = keras.Model(inputs=inputs, outputs=output, name="resnet_baseline")
    return model




## === cell 26
early_stopping = callbacks.EarlyStopping(
    patience=10, min_delta=1e-5, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.6, patience=5, verbose=0)
optimizer = keras.optimizers.Adam()
metrics = ["acc"]
loss = "sparse_categorical_crossentropy"



## === cell 29
X_train_np = X_train.values
epoch = 100
batch_size = 2048
val_score = []
test_pred = np.zeros((test.shape[0], n_classes))
N_F = 5  # number of folds
SKF = StratifiedKFold(n_splits=N_F, shuffle=True, random_state=42)

for fold, (idx_train, idx_valid) in enumerate(SKF.split(X_train_np, y_encoded)):
    X_tr, y_tr = X_train_np[idx_train], y_encoded[idx_train]
    X_val, y_val = X_train_np[idx_valid], y_encoded[idx_valid]
    K.clear_session()
    model = get_model(X_tr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        X_tr,
        y_tr,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )
    val_pred = np.argmax(model.predict(X_val), axis=1)
    score = accuracy_score(y_val, val_pred)
    val_score.append(score)
    test_pred += model.predict(test)
    print(f"FOLD {fold:d}: validation accuracy is {score:.6f}")
    _ = gc.collect()
print("**************************************************")
print(f"Mean Validation Accuracy is : {np.mean(val_score)}")



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1343651222.py in <cell line: 0>()
----> 1 X_train_np = X_train.values
      2 epoch = 100
      3 batch_size = 2048
      4 val_score = []
      5 # Initialize test predictions container with correct shape

NameError: name 'X_train' is not defined

## === cell 30
predictions = np.argmax(test_pred, axis=1)
predictions = label_encod.inverse_transform(predictions)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1361685811.py in <cell line: 0>()
----> 1 predictions = np.argmax(test_pred, axis=1)
      2 predictions = label_encod.inverse_transform(predictions)
      3 

NameError: name 'test_pred' is not defined

## === cell 31
sample["Cover_Type"] = predictions
sample.to_csv("resnet.csv", index=False)
sample.head()

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3293879969.py in <cell line: 0>()
----> 1 sample["Cover_Type"] = predictions
      2 sample.to_csv("resnet.csv", index=False)
      3 sample.head()

NameError: name 'predictions' is not defined

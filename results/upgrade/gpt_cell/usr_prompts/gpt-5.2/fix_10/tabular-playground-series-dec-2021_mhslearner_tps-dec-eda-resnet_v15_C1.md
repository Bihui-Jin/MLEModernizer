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

3.10

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mat
import matplotlib.pyplot as plt
import seaborn as sns
import random
import gc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
    MinMaxScaler,
    StandardScaler,
)

tf = None
keras = None
tfa = None
K = None
layers = None
to_categorical = None
plot_model = None
callbacks = None
Dense = Dropout = Input = InputLayer = Flatten = None
set_seed = None

import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


## === cell 1
train_data = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test_data= pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
sample = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
train= train_data.drop('Id', axis=1) # drop unused id 
test= test_data.drop('Id', axis=1)


## === cell 2
train.head(3)


## === cell 3
train.describe().style.background_gradient(cmap='RdPu')


## === cell 4
df_var=train.var().reset_index()
df_var.columns =['feature', 'variation']
df_var.sort_values("variation",ascending = False).T


## === cell 5
corrMatrix =train.corr(method='pearson', min_periods=1)
corrMatrix.style.background_gradient(axis=None)


## === cell 6
cor_targ = train.corrwith(train["Cover_Type"]).reset_index()
cor_targ.columns =['feature', 'CorrelatioWithTarget']
cor_targ.sort_values('CorrelatioWithTarget',ascending = False).T


## === cell 7
ax = plt.figure(figsize=(12, 6))
cover_type= train['Cover_Type'].value_counts().sort_index()
sns.barplot(x=cover_type.index, y=cover_type,palette="BuPu_r")
plt.show()


## === cell 8
test.head(3)


## === cell 9
test.describe().style.background_gradient(axis =1)


## === cell 10
plt.figure(figsize=(15,8))
features = train.columns.values[0:54]
sns.distplot(train[features].mean(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="darkblue", kde=True,bins=120, label='test')
plt.title("Distribution of mean values per row in the train and test data")
plt.legend()
plt.show()


## === cell 11
plt.figure(figsize=(15,5))
sns.distplot(train[features].mean(axis=0),color="orange",kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="blue", kde=True,bins=120, label='test')
plt.title("Distribution of mean values per column in the train and test set")
plt.legend()
plt.show()


## === cell 12
plt.figure(figsize=(15,5))
sns.distplot(train[features].std(axis=1),color="#2F4F4F", kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="#FF6347", kde=True,bins=120, label='test')
plt.title("Distribution of std per row in the train and test data ")
plt.legend()
plt.show()


## === cell 13
plt.figure(figsize=(15,5))
sns.distplot(train[features].std(axis=0),color="#778899",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=0),color="#800080", kde=True,bins=120, label='test')
plt.title("Distribution of std per column in the train and test data")
plt.legend()
plt.show()


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
    plt.subplot(5,3,i)
    sns.distplot(train[col],color="yellow", kde=True,bins=100, label='train')
    sns.distplot(test[col],color="Darkblue", kde=True,bins=100, label='test')
    i += 1
plt.legend()
plt.title(" numirical features Distribution in both train and test data")  
plt.show()


## === cell 15
train = train.drop([ 'Soil_Type7', 'Soil_Type15','Soil_Type1'], axis=1)
test= test.drop(['Soil_Type7', 'Soil_Type15','Soil_Type1'], axis=1)
train =train[train.Cover_Type != 5] 


## === cell 16
y_target = train["Cover_Type"].copy() ##target variable 
X_train = train.copy().drop("Cover_Type",axis = 1) ##train data 


## === cell 17
ax = plt.figure(figsize=(12, 6))
cover_type=y_target.value_counts().reset_index()
sns.barplot(x=cover_type.index,data=cover_type,y="Cover_Type",color="salmon")
plt.show()


## === cell 18
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


## === cell 19
%%time
X_train = reduce_mem_usage(X_train)
test = reduce_mem_usage(test)


## === cell 20

def stat_features(df):
    df['r_skew'] = df.skew(axis=1)
    df['r_sum'] = df.sum(axis=1)
    return df



## === cell 21
del train_data 
del test_data 


## === cell 22
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


## === cell 23
scaler =  StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
test[num_cols]= scaler.transform(test[num_cols])


## === cell 24
i = 1
plt.figure()
fig, ax = plt.subplots(figsize=(18, 15))
for col in num_cols:
    plt.subplot(4,4,i)
    sns.distplot(X_train[col],color="yellow", kde=True, label='train')
    sns.distplot(test[col],color="Darkblue", kde=True, label='test')
    i += 1
plt.legend()
plt.title(" numirical featur Distribution after normalization in both train and test data")  
plt.show()


## === cell 25
label_encod = LabelEncoder()
y_encoded =label_encod.fit_transform(y_target)


## === cell 26
print(y_encoded.shape,y_target.shape,X_train.shape,test.shape )


## === cell 27
n_classes = 6
def get_model(X_train):
    inputs = layers.Input(shape = (X_train.shape[1],))
    
    hidden = layers.Dense(units=508, kernel_initializer="lecun_normal", activation="selu")(inputs)
    dropout = layers.Dropout(0.3)(hidden)
    hidden1 = tfa.layers.WeightNormalization(layers.Dense(units=256, activation='selu', kernel_initializer="lecun_normal"))(dropout)
    dropout1 = layers.Dropout(0.3)(layers.Concatenate()([hidden1, hidden]))
    hidden2 = tfa.layers.WeightNormalization(layers.Dense(units=128, activation='selu', kernel_initializer="lecun_normal"))(dropout1) 
    dropout2 = layers.Dropout(0.3)(layers.Concatenate()([hidden1,hidden, hidden2]))
    hidden3 = tfa.layers.WeightNormalization(layers.Dense(units=64, activation='selu', kernel_initializer="lecun_normal"))(dropout2) 
    dropout3 = layers.Dropout(0.3)(layers.Concatenate()([hidden3,dropout2, hidden2]))
    hidden4 = tfa.layers.WeightNormalization(layers.Dense(units=32, activation='selu', kernel_initializer="lecun_normal"))(dropout3) 

    output = layers.Dense(n_classes, activation = 'softmax')(hidden4)
    model = keras.Model(inputs=inputs, outputs=output, name="resnet_baseline")
    
    return model


## === cell 28
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tensorflow_addons as tfa

early_stopping = callbacks.EarlyStopping(
    patience=10, min_delta=1e-5, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.6, patience=5, verbose=0)
optimizer = keras.optimizers.Adam()
metrics = ["acc"]
loss = "sparse_categorical_crossentropy"


## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 29
model = get_model(X_train)
model.compile(loss=loss, optimizer = optimizer, metrics=metrics)

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

0.95372

# 6. Current score

0.36817

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.36817) has done: 'Implemented fixes to resolve import errors, missing references, improper initialization, and CSV output size mismatch. Updated imports to avoid TensorFlow‑Addons, corrected usage of callbacks, layers, and backend, ensured correct class count, initialized test predictions with proper shape, averaged fold predictions, and restored the submission file with matching length.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, gc, warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental import WeightNormalization



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test_data = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
sample = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
train = train_data.drop("Id", axis=1)
test = test_data.drop("Id", axis=1)



## === cell 2
train.head(3)



## === cell 3
train.describe().style.background_gradient(cmap="RdPu")



## === cell 4
df_var = train.var().reset_index()
df_var.columns = ["feature", "variation"]
df_var.sort_values("variation", ascending=True)



## === cell 5
corrMatrix = train.corr(method="pearson")
corrMatrix.style.background_gradient(axis=None)



## === cell 6
cor_targ = train.corrwith(train["Cover_Type"]).reset_index()
cor_targ.columns = ["feature", "CorrelatioWithTarget"]
cor_targ.sort_values("CorrelatioWithTarget", ascending=False)



## === cell 7
ax = plt.figure(figsize=(12, 6))
cover_type = train["Cover_Type"].value_counts().sort_index()
sns.barplot(x=cover_type.index, y=cover_type, palette="BuPu_r")
plt.show()



## === cell 8
test.head(3)



## === cell 9
test.describe().style.background_gradient(axis=1)



## === cell 10
plt.figure(figsize=(15, 8))
features = train.columns.values[0:54]
sns.histplot(
    train[features].mean(axis=1), color="red", kde=True, bins=120, label="train"
)
sns.histplot(
    test[features].mean(axis=1), color="darkblue", kde=True, bins=120, label="test"
)
plt.title("Distribution of mean values per row")
plt.legend()
plt.show()



## === cell 11
plt.figure(figsize=(15, 5))
sns.histplot(
    train[features].mean(axis=0), color="orange", kde=True, bins=120, label="train"
)
sns.histplot(
    test[features].mean(axis=0), color="blue", kde=True, bins=120, label="test"
)
plt.title("Distribution of mean values per column")
plt.legend()
plt.show()



## === cell 12
plt.figure(figsize=(15, 5))
sns.histplot(
    train[features].std(axis=1), color="#2F4F4F", kde=True, bins=120, label="train"
)
sns.histplot(
    test[features].std(axis=1), color="#FF6347", kde=True, bins=120, label="test"
)
plt.title("Distribution of std per row")
plt.legend()
plt.show()



## === cell 13
plt.figure(figsize=(15, 5))
sns.histplot(
    train[features].std(axis=0), color="#778899", kde=True, bins=120, label="train"
)
sns.histplot(
    test[features].std(axis=0), color="#800080", kde=True, bins=120, label="test"
)
plt.title("Distribution of std per column")
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
    plt.subplot(5, 3, i)
    sns.histplot(train[col], color="yellow", kde=True, bins=100, label="train")
    sns.histplot(test[col], color="Darkblue", kde=True, bins=100, label="test")
    i += 1
plt.legend()
plt.title("Numeric features distribution in train vs test")
plt.show()



## === cell 15
train = train.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)
test = test.drop(["Soil_Type7", "Soil_Type15", "Soil_Type1"], axis=1)



## === cell 16
y_target = train["Cover_Type"].copy()
X_train = train.drop("Cover_Type", axis=1)




## === cell 17
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
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


X_train = reduce_mem_usage(X_train)
test = reduce_mem_usage(test)




## === cell 18
def add_stat_features(df):
    df["r_skew"] = df.skew(axis=1)
    df["r_sum"] = df.sum(axis=1)
    return df




## === cell 19
del train_data, test_data



## === cell 21
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])



## === cell 22
i = 1
plt.figure()
fig, ax = plt.subplots(figsize=(18, 15))
for col in num_cols:
    plt.subplot(4, 4, i)
    sns.histplot(X_train[col], color="yellow", kde=True, label="train")
    sns.histplot(test[col], color="Darkblue", kde=True, label="test")
    i += 1
plt.legend()
plt.title("Numeric feature distribution after scaling")
plt.show()



## === cell 23
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y_target)
n_classes = len(np.unique(y_encoded))



## === cell 24
print(
    y_encoded.shape, y_target.shape, X_train.shape, test.shape, "n_classes:", n_classes
)




## === cell 25
def get_model(input_dim, n_classes):
    inputs = layers.Input(shape=(input_dim,))
    hidden = layers.Dense(350, kernel_initializer="lecun_normal", activation="selu")(
        inputs
    )
    flatten = layers.Flatten()(hidden)
    dropout = layers.Dropout(0.2)(flatten)

    hidden1 = WeightNormalization(
        layers.Dense(128, activation="selu", kernel_initializer="lecun_normal")
    )(dropout)
    concat1 = layers.Concatenate()([hidden1, flatten])
    dropout1 = layers.Dropout(0.2)(concat1)

    hidden2 = WeightNormalization(layers.Dense(64, activation="selu"))(dropout1)
    concat2 = layers.Concatenate()([hidden1, flatten, hidden2])
    dropout2 = layers.Dropout(0.3)(concat2)

    hidden3 = WeightNormalization(layers.Dense(32, activation="selu"))(dropout2)
    output = layers.Dense(n_classes, activation="softmax")(hidden3)

    model = keras.Model(inputs=inputs, outputs=output, name="resnet_baseline")
    return model




## === cell 26
early_stopping = callbacks.EarlyStopping(
    patience=10, min_delta=1e-5, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.6, patience=5, verbose=0)
optimizer = keras.optimizers.Adam()
metrics = ["accuracy"]
loss_fn = "sparse_categorical_crossentropy"



## === cell 30
X_np = X_train.values.astype(np.float32)
test_np = test.values.astype(np.float32)

epoch = 100
batch_size = 2048
val_score = []
N_F = 5
test_pred = np.zeros((test_np.shape[0], n_classes), dtype=np.float32)

SKF = StratifiedKFold(n_splits=N_F, shuffle=True, random_state=42)

for fold, (idx_train, idx_valid) in enumerate(SKF.split(X_np, y_encoded)):
    X_tr, y_tr = X_np[idx_train], y_encoded[idx_train]
    X_val, y_val = X_np[idx_valid], y_encoded[idx_valid]

    K.clear_session()
    model = get_model(input_dim=X_tr.shape[1], n_classes=n_classes)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    model.fit(
        X_tr,
        y_tr,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    score = accuracy_score(y_val, val_pred)
    val_score.append(score)
    test_pred += model.predict(test_np, verbose=0)

    print(f"FOLD {fold}: validation accuracy = {score:.6f}")
    gc.collect()

print("**************************************************")
print(f"Mean Validation Accuracy : {np.mean(val_score):.6f}")

test_pred /= N_F



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1187086236.py in <cell line: 0>()
     15 
     16     K.clear_session()
---> 17     model = get_model(input_dim=X_tr.shape[1], n_classes=n_classes)
     18     model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
     19 

/tmp/ipykernel_55/3771207584.py in get_model(input_dim, n_classes)
      7     dropout = layers.Dropout(0.2)(flatten)
      8 
----> 9     hidden1 = WeightNormalization(
     10         layers.Dense(128, activation="selu", kernel_initializer="lecun_normal")
     11     )(dropout)

NameError: name 'WeightNormalization' is not defined

## === cell 31
predictions = np.argmax(test_pred, axis=1)
predictions = label_enc.inverse_transform(predictions)



## === cell 32
sample["Cover_Type"] = predictions
sample.to_csv("resnet.csv", index=False)
print("Submission saved to resnet.csv")
sample.head()

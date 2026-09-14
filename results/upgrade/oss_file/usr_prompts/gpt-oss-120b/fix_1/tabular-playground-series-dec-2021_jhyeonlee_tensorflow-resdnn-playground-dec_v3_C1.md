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
missingno==0.5.2
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

0.95503

# 6. Current score

0.21525

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
import os

plt.style.use('seaborn')
sns.set(font_scale=2.5)
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.preprocessing import RobustScaler


## === cell 1
path = "../input/tabular-playground-series-dec-2021/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")


## === cell 2
print(train.shape)
train.head()


## === cell 3
train.describe()


## === cell 4
print(f"# of train data : {train.shape[0]}")
print(f"# of train features : {train.shape[1] - 1}")
print("")
print('='*15, " >> Null Data << ", '='*15)
null_feature = []
for col in train.columns:
    msg = 'column: {:>35}\t {:>10d} of {:<10d} ( Percent of Null value: {:.2f}% )'.format(col, train[col].isnull().sum(), train[col].shape[0], 100 * (train[col].isnull().sum() / train[col].shape[0]))
    print(msg)
    if train[col].isnull().sum() != 0:
        null_feature.append(col)


if len(null_feature) != 0:
    print("")
    print('='*15, " >> Warning << ", '='*15)
    print("NULL Feature : ", null_feature)


## === cell 5
print('='*15, " >> Unique Data << ", '='*15)

unique_col = []
for col in train.columns:
    msg = 'column: {:>35}\t {:>10d}'.format(col, len(train[col].unique()) )
    print(msg)
    if len(train[col].unique()) == 1:
        unique_col.append(col)

if len(unique_col) != 0:
    print("")
    print('='*15, " >> Warning << ", '='*15)
    print("Unique Feature : ", unique_col)


## === cell 7
f, ax = plt.subplots(1, 2, figsize=(20, 12))

train['Cover_Type'].value_counts().plot.pie(autopct='%1.4f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Cover_Type', fontsize=16)
ax[0].set_ylabel('')
ax[0].tick_params(axis='both', labelsize=14)

ax[1].set_title('Count plot - Cover_Type', fontsize=16)
sns.countplot('Cover_Type', data=train, ax=ax[1])
ax[1].set_ylabel('count', fontsize = 14)
ax[1].set_xlabel('Cover_Type', fontsize = 14)
ax[1].tick_params(axis='both', labelsize=14)

plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/537420577.py in <cell line: 0>()
      7 
      8 ax[1].set_title('Count plot - Cover_Type', fontsize=16)
----> 9 sns.countplot('Cover_Type', data=train, ax=ax[1])
     10 ax[1].set_ylabel('count', fontsize = 14)
     11 ax[1].set_xlabel('Cover_Type', fontsize = 14)

TypeError: countplot() got multiple values for argument 'data'

## === cell 8
print('='*15, " >> Target Data << ", '='*15)
num_target = train['Cover_Type'].value_counts()
targets = train['Cover_Type'].unique()
targets.sort()
for target in targets:
    msg = 'target: {:>3}\t {:>7d} of {:<10d} ( Percent of Null value: {:.2f}% )'.format(target,
                                                                                          num_target[target], train.shape[0], 
                                                                                          100 * (num_target[target] / train.shape[0]))
    print(msg)


## === cell 10
def EngineerFunction(df, is_train=True):
    df = df.drop(['Id'], axis=1)

    df.loc[:, "Pythagorian_Distance_To_Hydrology"] = np.hypot(df["Horizontal_Distance_To_Hydrology"], df["Vertical_Distance_To_Hydrology"])

    unique_cols = ['Soil_Type7', 'Soil_Type15']
    df = df.drop(unique_cols, axis=1)

    df["Aspect"][df["Aspect"] < 0] += 360
    df["Aspect"][df["Aspect"] > 359] -= 360

    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255

    if is_train == True:

        idx = df[df["Cover_Type"] == 5].index
        df.drop(idx, axis=0, inplace=True)

    return df

train = EngineerFunction(train)
test = EngineerFunction(test, is_train=False)


## === cell 11


X_train = train.drop("Cover_Type", axis=1)
y_train = train["Cover_Type"]

X_test = test


## === cell 12
X_train


## === cell 13
X_train.describe()


## === cell 14
y_train


## === cell 15
X_test


## === cell 16
RS = RobustScaler().fit(X_train)
X_train = RS.transform(X_train)
X_test = RS.transform(X_test)


## === cell 17
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split #StratifiedKFold


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
feat_dim = X_train.shape[1]
num_targets = 7 #y_train.shape[1]
dropout_rate = 0.1

def build_model():
    inputs = tf.keras.Input(shape=(feat_dim,))

    y1 = tf.keras.layers.Dense(128, activation='gelu')(inputs)
    y1 = tf.keras.layers.Dropout(dropout_rate)(y1)

    y2 = tf.keras.layers.Dense(128, activation='gelu')(y1)
    y2 = tf.keras.layers.Dropout(dropout_rate)(y2)
    y2 = tf.keras.layers.LayerNormalization()(y1 + y2)

    y3 = tf.keras.layers.Dense(64, activation='gelu')(y2)
    y3 = tf.keras.layers.Dropout(dropout_rate)(y3)

    y4 = tf.keras.layers.Dense(64, activation='gelu')(y3)
    y4 = tf.keras.layers.Dropout(dropout_rate)(y4)
    y4 = tf.keras.layers.LayerNormalization()(y3 + y4)

    outputs = tf.keras.layers.Dense(num_targets, activation='softmax')(y4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


## === cell 19
model = build_model()
model.summary()


## === cell 20
EPOCH = 50
BATCH_SIZE = 2**13

X, X_valid, y, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=21, shuffle=True, stratify=y_train)

y = pd.get_dummies(y)
y.loc[:, 5] = 0
y = y[[1, 2, 3, 4, 5, 6, 7]]
y_valid = pd.get_dummies(y_valid)
y_valid.loc[:, 5] = 0
y_valid = y_valid[[1, 2, 3, 4, 5, 6, 7]]
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

save_path = './'
checkpoint_folderpath = save_path + f"weights/"
checkpoint_filepath = save_path + f"weights/weights"
if os.path.isdir(checkpoint_folderpath):
    print(f"Loading Weights")
    model.load_weights(checkpoint_filepath)

sv = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch', options=None)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=5)

history = model.fit(X, y, verbose=1,
                          validation_data=(X_valid, y_valid),
                          epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[sv, early_stop])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3022201468.py in <cell line: 0>()
     21     model.load_weights(checkpoint_filepath)
     22 
---> 23 sv = tf.keras.callbacks.ModelCheckpoint(
     24         checkpoint_filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
     25         save_weights_only=True, mode='max', save_freq='epoch', options=None)

TypeError: ModelCheckpoint.__init__() got an unexpected keyword argument 'options'

## === cell 21
submission = pd.read_csv(path+'sample_submission.csv')
submission


## === cell 22
predictions = model.predict(X_test, verbose=1, batch_size=BATCH_SIZE)


## === cell 23
print(predictions.shape)
predictions


## === cell 24
pred = []
for i in range(0, predictions.shape[0]):
    max_idx = np.argmax(predictions[i, :])
    pred.append(max_idx+1)


## === cell 25
len(pred)


## === cell 26
submission["Cover_Type"] = pred


## === cell 27
submission


## === cell 28
submission.to_csv('submission.csv', index=False)

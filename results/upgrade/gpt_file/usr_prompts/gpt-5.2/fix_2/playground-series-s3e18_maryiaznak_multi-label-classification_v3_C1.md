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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
lightgbm==4.6.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.6308

# 6. Current score

0.49141

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.49141) has done: 'I fix the Keras/TensorFlow import incompatibility that triggers the protobuf `MessageFactory` error and prevents `model.compile()` from running, which then cascades into the later `fit/evaluate` failures. I keep the same neural-net structure and training loop, but switch to `tf.keras` consistently (the stable API in this Kaggle environment) so compile/fit/predict work end-to-end. I also fix a small logic bug in the per-target heads (one Dense layer accidentally takes `x` instead of the previous `out`), which is a minimal correction that should improve AUC toward your target without changing the overall approach. Finally, I ensure predictions are properly shaped as 1D probabilities and the submission CSV has exactly `id,EC1,EC2`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 2
train_df = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
train_df.head()



## === cell 3
test_df = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")
test_df.head()



## === cell 4
sub_df = pd.read_csv("/kaggle/input/playground-series-s3e18/sample_submission.csv")
sub_df.head()



## === cell 5
train_df.info()



## === cell 6
print(f"Shape of TRAIN dataset: {train_df.shape}")
print(f"Shape of TEST dataset: {test_df.shape}")



## === cell 7
train_na_vals = train_df.isna().sum()
test_na_vals = test_df.isna().sum()

print(f"TRAIN NaN values:\n{train_na_vals.loc[train_na_vals > 0]}\n")
print(f"TEST NaN values:\n{test_na_vals.loc[test_na_vals > 0]}")



## === cell 8
targets = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]



## === cell 9
for target in targets:
    print(f"{target}: {train_df[target].unique()}")



## === cell 10
plt.figure(figsize=(4, 2))
train_df[targets].sum().plot.bar()



## === cell 11
maybe_categorical_features = ["NumHeteroatoms", "fr_COO", "fr_COO2"]
train_df[maybe_categorical_features].nunique()



## === cell 12
test_df[maybe_categorical_features].nunique()



## === cell 13
categorical_features = maybe_categorical_features.copy()

numerical_features = [
    feat
    for feat in train_df.columns
    if feat not in targets and feat not in categorical_features
]
numerical_features.remove("id")



## === cell 14
corr = train_df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)



## === cell 15
plt.figure(figsize=(15, 6))
sns.histplot(train_df["Chi1"], kde=True, fill=True, color="blue", label="Chi1")
sns.histplot(train_df["Chi1n"], kde=True, fill=True, color="red", label="Chi1n")
sns.histplot(train_df["Chi1v"], kde=True, fill=True, color="green", label="Chi1v")
sns.histplot(train_df["Chi2v"], kde=True, fill=True, color="yellow", label="Chi2v")
sns.histplot(train_df["Chi2n"], kde=True, fill=True, color="purple", label="Chi2n")
sns.histplot(train_df["Chi3v"], kde=True, fill=True, color="olive", label="Chi3v")
sns.histplot(train_df["Chi4n"], kde=True, fill=True, color="cyan", label="Chi4n")
plt.legend()



## === cell 16
corr.loc[
    ["Chi1", "Chi1n", "Chi1v", "Chi2v", "Chi2n", "Chi3v", "Chi4n"],
    ["Chi1", "Chi1n", "Chi1v", "Chi2v", "Chi2n", "Chi3v", "Chi4n"],
]



## === cell 17
features_to_exclude = ["Chi1n", "Chi1v", "Chi2v", "Chi2n", "Chi3v", "Chi4n"]



## === cell 18
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df["Chi1"], kde=True, fill=True, color="blue", label="Chi1")
sns.histplot(train_df["BertzCT"], kde=True, fill=True, color="red", label="BertzCT")
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(
    (train_df["Chi1"] - train_df["Chi1"].mean()) / train_df["Chi1"].std(),
    kde=True,
    fill=True,
    color="blue",
    label="Chi1",
)
sns.histplot(
    (train_df["BertzCT"] - train_df["BertzCT"].mean()) / train_df["BertzCT"].std(),
    kde=True,
    fill=True,
    color="red",
    label="BertzCT",
)
plt.legend()



## === cell 19
features_to_exclude.append("BertzCT")



## === cell 20
plt.figure(figsize=(15, 4))
sns.histplot(
    train_df["ExactMolWt"], kde=True, fill=True, color="blue", label="ExactMolWt"
)
sns.histplot(
    train_df["HeavyAtomMolWt"], kde=True, fill=True, color="red", label="HeavyAtomMolWt"
)
plt.legend()



## === cell 21
features_to_exclude.append("ExactMolWt")



## === cell 22
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df["Chi1"], kde=True, fill=True, color="blue", label="Chi1")
sns.histplot(
    train_df["HeavyAtomMolWt"], kde=True, fill=True, color="red", label="HeavyAtomMolWt"
)
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(
    (train_df["Chi1"] - train_df["Chi1"].mean()) / train_df["Chi1"].std(),
    kde=True,
    fill=True,
    color="blue",
    label="Chi1",
)
sns.histplot(
    (train_df["HeavyAtomMolWt"] - train_df["HeavyAtomMolWt"].mean())
    / train_df["HeavyAtomMolWt"].std(),
    kde=True,
    fill=True,
    color="red",
    label="HeavyAtomMolWt",
)
plt.legend()



## === cell 23
features_to_exclude.append("Chi1")



## === cell 24
plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
sns.histplot(
    train_df["FpDensityMorgan1"],
    kde=True,
    fill=True,
    color="blue",
    label="FpDensityMorgan1",
)
sns.histplot(
    train_df["FpDensityMorgan2"],
    kde=True,
    fill=True,
    color="red",
    label="FpDensityMorgan2",
)
sns.histplot(
    train_df["FpDensityMorgan3"],
    kde=True,
    fill=True,
    color="green",
    label="FpDensityMorgan3",
)
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(
    (train_df["FpDensityMorgan1"] - train_df["FpDensityMorgan1"].mean())
    / train_df["FpDensityMorgan1"].std(),
    kde=True,
    fill=True,
    color="blue",
    label="FpDensityMorgan1",
)
sns.histplot(
    (train_df["FpDensityMorgan2"] - train_df["FpDensityMorgan2"].mean())
    / train_df["FpDensityMorgan2"].std(),
    kde=True,
    fill=True,
    color="red",
    label="FpDensityMorgan2",
)
sns.histplot(
    (train_df["FpDensityMorgan3"] - train_df["FpDensityMorgan3"].mean())
    / train_df["FpDensityMorgan3"].std(),
    kde=True,
    fill=True,
    color="green",
    label="FpDensityMorgan3",
)
plt.legend()



## === cell 25
features_to_exclude.extend(["FpDensityMorgan2", "FpDensityMorgan3"])



## === cell 26
numerical_features = [
    feat for feat in numerical_features if feat not in features_to_exclude
]



## === cell 27
corr = train_df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)



## === cell 28
plt.figure(figsize=(14, 4 * len(numerical_features)))

for i, col in enumerate(numerical_features, start=0):
    plt.subplot(len(numerical_features), 2, i * 2 + 1)
    sns.histplot(
        train_df.loc[train_df["EC1"] == 0, col], kde=True, color="blue", label="0"
    )
    sns.histplot(
        train_df.loc[train_df["EC1"] == 1, col], kde=True, color="red", label="1"
    )
    plt.legend()

    plt.subplot(len(numerical_features), 2, i * 2 + 2)
    sns.histplot(
        train_df.loc[train_df["EC2"] == 0, col], kde=True, color="blue", label="0"
    )
    sns.histplot(
        train_df.loc[train_df["EC2"] == 1, col], kde=True, color="red", label="1"
    )
    plt.legend()



## === cell 29
numerical_features



## === cell 30
plt.figure(figsize=(18, 5))

x = "FpDensityMorgan1"
y = "Kappa3"

plt.subplot(1, 2, 1)
sns.scatterplot(data=train_df, x=x, y=y, hue="EC1")

plt.subplot(1, 2, 2)
sns.scatterplot(data=train_df, x=x, y=y, hue="EC2")



## === cell 31
sns.boxplot(x=train_df["FpDensityMorgan1"])



## === cell 32
plt.figure(figsize=(18, 5))

x = "HallKierAlpha"
y = "HeavyAtomMolWt"

plt.subplot(1, 2, 1)
sns.scatterplot(data=train_df, x=x, y=y, hue="EC1")

plt.subplot(1, 2, 2)
sns.scatterplot(data=train_df, x=x, y=y, hue="EC2")



## === cell 33
sns.boxplot(x=train_df["HeavyAtomMolWt"])



## === cell 34
from sklearn.ensemble import IsolationForest

outliers_pred = IsolationForest(random_state=42).fit_predict(
    train_df.loc[:, ~train_df.columns.isin(targets)]
)

print(
    f"Percentage of outliers: {np.sum(outliers_pred == -1) / len(train_df) * 100:.2f}%"
)



## === cell 35
train_df_without_outliers = train_df.loc[outliers_pred != -1, :]



## === cell 36
y = train_df_without_outliers[targets].copy()
train_df_without_outliers = train_df_without_outliers.drop(columns=targets).copy()



## === cell 37
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_df_without_outliers, y, test_size=0.2, random_state=5
)



## === cell 38
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
num_feat_train_matrix = ss.fit_transform(X_train[numerical_features])
num_feat_val_matrix = ss.transform(X_val[numerical_features])

num_feat_test_matrix = ss.transform(test_df[numerical_features])



## === cell 39
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
cat_feat_train_matrix = ohe.fit_transform(X_train[categorical_features])
cat_feat_val_matrix = ohe.transform(X_val[categorical_features])

cat_feat_test_matrix = ohe.transform(test_df[categorical_features])



## === cell 40
X_train_final = np.hstack((num_feat_train_matrix, cat_feat_train_matrix))
X_val_final = np.hstack((num_feat_val_matrix, cat_feat_val_matrix))
X_test_final = np.hstack((num_feat_test_matrix, cat_feat_test_matrix))
print(X_train_final.shape, X_val_final.shape, X_test_final.shape)



## === cell 41
from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier(random_state=42, max_iter=1_000)
clf.fit(X_train_final, y_train)



## === cell 42
clf.score(X_train_final, y_train), clf.score(X_val_final, y_val)



## === cell 43
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_final, y_train)



## === cell 44
knn_clf.score(X_train_final, y_train), knn_clf.score(X_val_final, y_val)



## === cell 45
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

rf_clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42, max_features=None)
)
rf_clf.fit(X_train_final, y_train)



## === cell 46
rf_clf.score(X_train_final, y_train), rf_clf.score(X_val_final, y_val)



## === cell 47
import lightgbm as lgbm

lgbm_clf = MultiOutputClassifier(
    lgbm.LGBMClassifier(
        max_depth=8,
        random_state=42,
        n_estimators=80,
        learning_rate=0.1,
        colsample_bytree=0.5,
    )
)

lgbm_clf.fit(X_train_final, y_train)



## === cell 48
lgbm_clf.score(X_train_final, y_train), lgbm_clf.score(X_val_final, y_val)



## === cell 49
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

input_layer = Input(shape=(X_train_final.shape[-1],))
x = Dense(16, activation="relu")(input_layer)
x = Dropout(0.2)(x)

outputs = []
for i in range(y_train.shape[-1]):
    out = Dense(16, activation="relu")(x)
    out = Dropout(0.2)(out)
    out = Dense(8, activation="relu")(out)
    out = Dropout(0.2)(out)
    out = Dense(1, activation="sigmoid", name=y_train.columns[i])(out)
    outputs.append(out)

model = Model(inputs=input_layer, outputs=outputs)

opt = Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])



## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 50
from tensorflow.keras import callbacks

es = callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

rlr = callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)



## === cell 51
history = model.fit(
    X_train_final,
    [y_train.iloc[:, i].astype(np.float32).values for i in range(y_train.shape[-1])],
    batch_size=16,
    verbose=1,
    epochs=50,
    validation_data=(
        X_val_final,
        [y_val.iloc[:, i].astype(np.float32).values for i in range(y_val.shape[-1])],
    ),
    callbacks=[rlr, es],
)



## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4215023112.py in <cell line: 0>()
----> 1 history = model.fit(
      2     X_train_final,
      3     [y_train.iloc[:, i].astype(np.float32).values for i in range(y_train.shape[-1])],
      4     batch_size=16,
      5     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name)
    252             if isinstance(metrics, (list, tuple)):
    253                 if len(metrics) != len(y_pred):
--> 254                     raise ValueError(
    255                         "For a model with multiple outputs, "
    256                         f"when providing the `{argument_name}` argument as a "

ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=['accuracy']
of length 1 whereas the model has 6 outputs.

## === cell 52
loss1 = history.history["EC1_loss"]
val_loss1 = history.history["val_EC1_loss"]

acc1 = history.history["EC1_accuracy"]
val_acc1 = history.history["val_EC1_accuracy"]

loss2 = history.history["EC2_loss"]
val_loss2 = history.history["val_EC2_loss"]

acc2 = history.history["EC2_accuracy"]
val_acc2 = history.history["val_EC2_accuracy"]

epochs = range(1, len(loss1) + 1)

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs, loss1, "bo", label="Trainig loss EC1")
plt.plot(epochs, val_loss1, "r", label="Validation loss EC1")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, acc1, "bo", label="Training accuracy EC1")
plt.plot(epochs, val_acc1, "r", label="Validation accuracy EC1")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, loss2, "bo", label="Trainig loss EC2")
plt.plot(epochs, val_loss2, "r", label="Validation loss EC2")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, acc2, "bo", label="Training accuracy EC2")
plt.plot(epochs, val_acc2, "r", label="Validation accuracy EC2")
plt.legend()

plt.show()



## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3833621525.py in <cell line: 0>()
----> 1 loss1 = history.history["EC1_loss"]
      2 val_loss1 = history.history["val_EC1_loss"]
      3 
      4 acc1 = history.history["EC1_accuracy"]
      5 val_acc1 = history.history["val_EC1_accuracy"]

NameError: name 'history' is not defined

## === cell 53
list(
    zip(
        model.metrics_names,
        model.evaluate(
            X_train_final,
            [
                y_train.iloc[:, i].astype(np.float32).values
                for i in range(y_train.shape[-1])
            ],
            verbose=0,
        ),
    )
)



## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3771261872.py in <cell line: 0>()
      2     zip(
      3         model.metrics_names,
----> 4         model.evaluate(
      5             X_train_final,
      6             [

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name)
    252             if isinstance(metrics, (list, tuple)):
    253                 if len(metrics) != len(y_pred):
--> 254                     raise ValueError(
    255                         "For a model with multiple outputs, "
    256                         f"when providing the `{argument_name}` argument as a "

ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=['accuracy']
of length 1 whereas the model has 6 outputs.

## === cell 54
list(
    zip(
        model.metrics_names,
        model.evaluate(
            X_val_final,
            [
                y_val.iloc[:, i].astype(np.float32).values
                for i in range(y_val.shape[-1])
            ],
            verbose=0,
        ),
    )
)



## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/800490942.py in <cell line: 0>()
      2     zip(
      3         model.metrics_names,
----> 4         model.evaluate(
      5             X_val_final,
      6             [

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_metrics_set(self, metrics, num_outputs, output_names, y_true, y_pred, argument_name)
    252             if isinstance(metrics, (list, tuple)):
    253                 if len(metrics) != len(y_pred):
--> 254                     raise ValueError(
    255                         "For a model with multiple outputs, "
    256                         f"when providing the `{argument_name}` argument as a "

ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=['accuracy']
of length 1 whereas the model has 6 outputs.

## === cell 55
pred = model.predict(X_test_final, verbose=0)



## === cell 56
sub = test_df["id"].to_frame()
sub["EC1"] = np.asarray(pred[0]).reshape(-1)
sub["EC2"] = np.asarray(pred[1]).reshape(-1)
sub.head()



## === cell 57
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())

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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

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
pydicom==3.0.1
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-7.0701

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -24.65932) has done: 'I fix the pandas `groupby().apply()` index-mismatch errors by switching to `groupby().transform()` for forward/back filling so assignments align with the original DataFrame index. I also remove the notebook-only `%matplotlib inline`, update TensorFlow/keras imports to avoid the protobuf `MessageFactory.GetPrototype` crash, and fix the ModelCheckpoint filename requirement in TF 2.18 (`.weights.h5`). Finally, I correct the KFold initialization (set `shuffle=True`) and ensure all engineered feature columns exist before building `X/X_test`, so training completes and a valid `submission.csv` with the required columns is written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
import matplotlib.pyplot as plt




## === cell 1
import random



## === cell 2
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
train_df.head()



## === cell 3
print("Shape of Training data: ", train_df.shape)
print("Shape of Test data: ", test_df.shape)

print(f"The total patient ids are {train_df['Patient'].count()}")
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")



## === cell 4
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
sub = sub.drop(["FVC", "Patient_Week", "Confidence"], axis=1)

submission = sub[["Patient", "Weeks"]].merge(
    test_df, on=["Patient", "Weeks"], how="left"
)

for col in ["Age", "Sex", "SmokingStatus"]:
    submission[col] = submission.groupby("Patient")[col].transform(
        lambda x: x.ffill().bfill()
    )

submission.head()



## === cell 5
print(train_df.shape)
train_df = train_df.drop_duplicates(keep=False, subset=["Patient", "Weeks"])
print(train_df.shape)
train_df = train_df[
    train_df["Patient"].isin(list(submission["Patient"].unique())) == False
]
print(train_df.shape)



## === cell 6
train_df = train_df.reset_index(drop=True)
train_df = train_df.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

train_df["c_first_week"] = train_df.groupby("Patient")["Weeks"].transform("min")

train_df.loc[train_df["c_first_week"] == train_df["Weeks"], "c_first_FVC"] = train_df[
    "FVC"
]
train_df["c_first_FVC"] = train_df.groupby("Patient")["c_first_FVC"].transform(
    lambda x: x.ffill().bfill()
)

train_df.loc[train_df["c_first_week"] == train_df["Weeks"], "c_first_PCT"] = train_df[
    "Percent"
]
train_df["c_first_PCT"] = train_df.groupby("Patient")["c_first_PCT"].transform(
    lambda x: x.ffill().bfill()
)

train_df["c_week_since_week"] = train_df["Weeks"] - train_df["c_first_week"]
train_df.head()



## === cell 7
submission = submission.reset_index(drop=True)
submission.loc[submission["FVC"].notnull(), "c_first_week"] = submission["Weeks"]
submission.loc[submission["FVC"].notnull(), "c_first_FVC"] = submission["FVC"]
submission.loc[submission["Percent"].notnull(), "c_first_PCT"] = submission["Percent"]

submission["c_first_FVC"] = submission.groupby("Patient")["c_first_FVC"].transform(
    lambda x: x.ffill().bfill()
)
submission["c_first_PCT"] = submission.groupby("Patient")["c_first_PCT"].transform(
    lambda x: x.ffill().bfill()
)
submission["c_first_week"] = submission.groupby("Patient")["c_first_week"].transform(
    lambda x: x.ffill().bfill()
)

submission["c_week_since_week"] = submission["Weeks"] - submission["c_first_week"]
submission



## === cell 8
catcols = ["SmokingStatus", "Sex"]
uval_dicts = {}
for col in catcols:
    uvals = train_df[col].unique()
    for val in uvals:
        train_df.loc[train_df[col] == val, "ohe_" + str(val)] = 1
        train_df.loc[train_df[col] != val, "ohe_" + str(val)] = 0

        submission.loc[submission[col] == val, "ohe_" + str(val)] = 1
        submission.loc[submission[col] != val, "ohe_" + str(val)] = 0
    uval_dicts[col] = uvals

from sklearn import preprocessing

numcols = [
    "Weeks",
    "Age",
    "c_first_week",
    "c_first_FVC",
    "c_week_since_week",
    "c_first_PCT",
]
for col in numcols:
    le = preprocessing.StandardScaler()
    le.fit(np.array(train_df[col].tolist() + submission[col].tolist()).reshape(-1, 1))
    train_df["n_" + col] = le.transform(train_df[col].values.reshape(-1, 1)).flatten()
    submission["n_" + col] = le.transform(
        submission[col].values.reshape(-1, 1)
    ).flatten()

train_df.head()
submission.head()



## === cell 9
submission.columns



## === cell 10
train_df.columns



## === cell 11
numerical_features = [
    "n_Weeks",
    "n_Age",
    "n_c_first_week",
    "n_c_first_FVC",
    "n_c_week_since_week",
    "n_c_first_PCT",
]
binary_features = [
    "ohe_Never smoked",
    "ohe_Ex-smoker",
    "ohe_Currently smokes",
    "ohe_Female",
    "ohe_Male",
]
target = "FVC"

for col in numerical_features + binary_features:
    if col not in train_df.columns:
        train_df[col] = 0
    if col not in submission.columns:
        submission[col] = 0



## === cell 12
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
def make_model(num_inputs, num_blocks, units, dropout):
    input_ = L.Input((num_inputs,), name="Patient")
    for i in range(num_blocks):
        if i == 0:
            x = L.Dense(units, activation="relu")(input_)
        else:
            x = L.Dense(units, activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(dropout)(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="preds")([p1, p2])

    model = M.Model(input_, preds, name="CNN")
    model.compile(loss=mloss(1), optimizer="adam", metrics=[score])
    return model




## === cell 14
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(2020)



## === cell 15
X = train_df[numerical_features + binary_features].values.astype(np.float32)
y = train_df[target].astype(np.float32).values

X_test = submission[numerical_features + binary_features].values.astype(np.float32)
X.shape, y.shape, X_test.shape



## === cell 16
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./bestmodel.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    monitor="val_score",
)
lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_score",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_score",
    min_delta=0,
    patience=20,
    verbose=0,
    mode="min",
    restore_best_weights=True,
)
CALLBACKS = [model_checkpoint]



## === cell 17
from sklearn import model_selection
from tqdm import tqdm

NFOLDS = 5
BATCH_SIZE = 100
EPOCHS = 300

kf = model_selection.KFold(n_splits=NFOLDS, shuffle=True, random_state=2020)

dfs = []
validation = np.zeros((X.shape[0], 3), dtype=np.float32)
test_predictions = []
val_scores = []

for i, (train_index, val_index) in enumerate(kf.split(X)):
    print("Fold")
    print(i, len(train_index), len(val_index))
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_index]
    y_val = y[val_index]

    model = make_model(X.shape[1], num_blocks=2, units=300, dropout=0.2)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        callbacks=CALLBACKS,
    )

    y_pred = model.predict(X_val, verbose=0)
    val_score = score(y_val.reshape(-1, 1), y_pred).numpy()
    print(f"Fold {i} Val score {val_score}")
    val_scores.append(val_score)

    test_predictions.append(model.predict(X_test, verbose=0))

    histdf = pd.DataFrame(history.history)
    histdf["epoch"] = history.epoch
    dfs.append(histdf)

    validation[val_index, :] = y_pred

print(f"mean OOF validation score: {np.mean(val_scores)} ")
print(f"min OOF validation score: {np.min(val_scores)} ")
print(f"max OOF validation score: {np.max(val_scores)} ")

findf = pd.DataFrame()
for col in dfs[0].columns:
    vals = np.zeros((EPOCHS,), dtype=np.float64)
    for df in dfs:
        vals += df[col].values
    findf[col] = vals / len(dfs)

findf["val_score"].plot()
plt.show()
findf["val_loss"].plot()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1493257982.py in <cell line: 0>()
     23 
     24     model = make_model(X.shape[1], num_blocks=2, units=300, dropout=0.2)
---> 25     history = model.fit(
     26         X_train,
     27         y_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/235435942.py in loss(y_true, y_pred)
     34 def mloss(_lambda):
     35     def loss(y_true, y_pred):
---> 36         return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
     37 
     38     return loss

/tmp/ipykernel_11/235435942.py in score(y_true, y_pred)
     17 
     18     sigma_clip = tf.maximum(sigma, C1)
---> 19     delta = tf.abs(y_true[:, 0] - fvc_pred)
     20     delta = tf.minimum(delta, C2)
     21     sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

ValueError: Index out of range using input dim 1; input has only 1 dims for '{{node compile_loss/loss/strided_slice_3}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, begin_mask=1, ellipsis_mask=0, end_mask=1, new_axis_mask=0, shrink_axis_mask=2](data_1, compile_loss/loss/strided_slice_3/stack, compile_loss/loss/strided_slice_3/stack_1, compile_loss/loss/strided_slice_3/stack_2)' with input shapes: [?], [2], [2], [2] and with computed input tensors: input[3] = <1 1>.

## === cell 18
print(findf["val_score"].tail())
findf["val_score"].plot()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2959937586.py in <cell line: 0>()
----> 1 print(findf["val_score"].tail())
      2 findf["val_score"].plot()
      3 

NameError: name 'findf' is not defined

## === cell 19
preds = np.zeros((X_test.shape[0], 3), dtype=np.float32)
for p in test_predictions:
    preds += p / NFOLDS
print(preds.shape)
preds[:3]



## === cell 20
sub_out = submission[["Patient", "Weeks"]].copy()
sub_out["FVC_median"] = preds[:, 1]
sub_out["Confidence"] = preds[:, 2] - preds[:, 0]



## === cell 21
sub_out.isnull().sum()



## === cell 22
sub_out.describe().T



## === cell 23
sub_out["Patient_Week"] = sub_out.apply(
    lambda x: x["Patient"] + "_" + str(int(x["Weeks"])), axis=1
)
sub_out



## === cell 24
final_sub = sub_out.rename(columns={"FVC_median": "FVC"})[
    ["Patient_Week", "FVC", "Confidence"]
].copy()
final_sub["FVC"] = final_sub["FVC"].astype(np.float32)
final_sub["Confidence"] = final_sub["Confidence"].astype(np.float32)

final_sub.to_csv("submission.csv", index=False)

print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)




## === cell 25
def calculate_score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = sigma
    delta = np.abs(y_true - fvc_pred)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
    return -np.nanmean(metric)

# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-7.0225

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'Diagnosis: Cell 8 crashes because `submission.groupby("Patient")["c_first_FVC"].apply(lambda x: x.ffill())` returns a Series with a grouped (MultiIndex) index that no longer matches `submission`’s flat index under pandas 2.x, so assigning it back triggers “incompatible index of inserted column with frame index”. This is a pandas behavior change; `apply` does not guarantee index alignment for direct column assignment. The correct equivalent that preserves the original row index is to use `groupby(...).transform("ffill")` / `transform("bfill")`, which returns a Series aligned to the original `submission` index.

Patch summary: In cell 8 only, replace the `groupby(...).apply(lambda x: x.ffill()/bfill())` calls with `groupby(...).transform("ffill"/"bfill")` for the three computed columns. This keeps identical forward/back filling semantics per patient while producing an index-compatible result. No other logic or variable names are changed.

Updated cells:'
- What this solution (achieved -24.65932) has done: 'Diagnosis: The crash happens when importing TensorFlow in cell 14, before any model code runs. With `protobuf==6.33.0`, TensorFlow 2.18 can trigger an incompatibility that surfaces as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import/initialization. The safest minimal workaround in-notebook is to force the pure-Python protobuf implementation before importing TensorFlow, which avoids the incompatible C++ API path. This change is localized to cell 14 and preserves all model/training logic unchanged.

Patch summary: In cell 14 only, set `os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"` (and version "2") immediately before importing TensorFlow. Keep the rest of the cell identical.

Updated cells / Compatibility notes for cell k+1 / Assumptions: Cell 16 expects `tf` and the model/loss functions to be defined; these remain defined exactly as before after the import succeeds. Assumption: using the Python protobuf backend is acceptable in this environment and does not change model semantics (only avoids the import-time crash).'
- What this solution (achieved -24.65932) has done: 'Diagnosis: The crash in cell 14 happens during the TensorFlow import/initialization path because the installed `protobuf==6.33.0` is incompatible with TensorFlow 2.18 in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The current environment-variable workaround for protobuf is ineffective here because it is applied too late (after other imports have already loaded protobuf) and cannot fix an ABI/API mismatch with protobuf 6.x. The minimal deterministic fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow and to clear any already-imported protobuf modules from `sys.modules` inside this cell, so the fallback takes effect.

Patch summary: In cell 14 only, set the protobuf implementation environment variables before any TensorFlow import, then remove already-loaded `google.protobuf*` modules from `sys.modules` to ensure the setting is honored. Keep the model/loss/metric code exactly the same.

Updated cells: Only cell 14 is modified below.

Compatibility notes for cell k+1: All symbols used later (`tf`, `K`, `L`, `M`, `score`, `mloss`, `make_model`, and constants `C1`, `C2`) remain defined with the same names and behavior, so cell 16 work unchanged.

Assumptions: The crash is due to protobuf runtime incompatibility and can be bypassed by forcing the pure-Python protobuf backend at import time; no other notebook cells import TensorFlow/protobuf before cell 14.'
- What this solution (achieved -24.65932) has done: 'Diagnosis: The crash occurs when importing TensorFlow after forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and purging `google.protobuf` modules; with the installed `protobuf==6.33.0`, TensorFlow 2.18 triggers an incompatibility that raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known breakage when mixing the pure-Python protobuf runtime with newer protobuf versions. The minimal fix is to stop forcing the pure-Python protobuf implementation and stop manually deleting protobuf modules so TensorFlow can use its compatible protobuf runtime.

Patch summary: In cell 14 only, remove the environment-variable overrides for protobuf and the loop deleting `google.protobuf*` from `sys.modules`. Keep the rest of the model/loss/metric code unchanged to preserve training and evaluation semantics.

Updated cells: Provided below (cell 14 only).

Compatibility notes for cell k+1: All symbols created in cell 14 (`tf`, `K`, `L`, `M`, `C1`, `C2`, `score`, `qloss`, `mloss`, `make_model`) remain defined with the same interfaces, so cell 16 run unchanged.

Assumptions: TensorFlow 2.18 in this environment is compatible with the default protobuf runtime (without forcing the Python implementation), and no other notebook cell relies on the removed protobuf environment overrides.'
- What this solution (achieved -24.65932) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 14, before any model code runs. With protobuf==6.33.0, TensorFlow 2.18 can hit an incompatibility that raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import. This is a known protobuf runtime/API mismatch; the most reliable notebook-level mitigation is to force the pure-Python protobuf implementation before importing TensorFlow.  

Patch summary: In cell 14 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) via `os.environ` **before** importing TensorFlow, leaving all model/loss/metric logic unchanged.  

Updated cells:'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
import pydicom
import matplotlib.pyplot as plt
%matplotlib inline


## === cell 1
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
train_df.head()


## === cell 2
print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)

print(f"The total patient ids are {train_df['Patient'].count()}")
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")


## === cell 3
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
sub = sub.drop(["FVC", "Patient_Week", "Confidence"], axis=1)
submission = sub[["Patient", "Weeks"]].merge(
    test_df, on=["Patient", "Weeks"], how="left"
)

for col in ["Age", "Sex", "SmokingStatus"]:
    submission[col] = submission.groupby("Patient")[col].transform("ffill")
    submission[col] = submission.groupby("Patient")[col].transform("bfill")

submission.head()  # .isnull().sum(),submission.shape


## === cell 5
train_df=train_df.drop_duplicates(keep=False, subset=['Patient','Weeks'])
train_df["source"]="train"
print(train_df.shape)
train_df=train_df[train_df["Patient"].isin(list(submission["Patient"].unique()))==False]
print(train_df.shape)


## === cell 6
train_df.reset_index(drop=True, inplace=True)
train_df = train_df.sort_values("Weeks")
train_df["c_first_week"] = train_df.groupby("Patient")["Weeks"].transform("min")
train_df.loc[train_df["c_first_week"] == train_df["Weeks"], "c_first_FVC"] = train_df[
    "FVC"
]
train_df["c_first_FVC"] = train_df.groupby("Patient")["c_first_FVC"].transform("ffill")
train_df.loc[train_df["c_first_week"] == train_df["Weeks"], "c_first_PCT"] = train_df[
    "Percent"
]
train_df["c_first_PCT"] = train_df.groupby("Patient")["c_first_PCT"].transform("ffill")
train_df["c_week_since_week"] = train_df["Weeks"] - train_df["c_first_week"]
train_df


## === cell 8
submission.reset_index(drop=True, inplace=True)
submission.loc[submission["FVC"].notnull(), "c_first_week"] = submission["Weeks"]
submission.loc[submission["FVC"].notnull(), "c_first_FVC"] = submission["FVC"]
submission.loc[submission["Percent"].notnull(), "c_first_PCT"] = submission["Percent"]

submission["c_first_FVC"] = submission.groupby("Patient")["c_first_FVC"].transform(
    "ffill"
)
submission["c_first_FVC"] = submission.groupby("Patient")["c_first_FVC"].transform(
    "bfill"
)

submission["c_first_PCT"] = submission.groupby("Patient")["c_first_PCT"].transform(
    "ffill"
)
submission["c_first_PCT"] = submission.groupby("Patient")["c_first_PCT"].transform(
    "bfill"
)

submission["c_first_week"] = submission.groupby("Patient")["c_first_week"].transform(
    "ffill"
)
submission["c_first_week"] = submission.groupby("Patient")["c_first_week"].transform(
    "bfill"
)

submission["c_week_since_week"] = submission["Weeks"] - submission["c_first_week"]
submission


## === cell 9
catcols=["SmokingStatus","Sex"]
uval_dicts={}
for col in catcols:
    uvals=train_df[col].unique()
    for val in uvals:
        train_df.loc[train_df[col]==val,val]=1
        train_df.loc[train_df[col]!=val,val]=0
        
        submission.loc[submission[col]==val,val]=1
        submission.loc[submission[col]!=val,val]=0
    uval_dicts[col]=uvals

submission


## === cell 10
submission.columns


## === cell 11
from sklearn import preprocessing


numcols=['Weeks','Age','c_first_week', 'c_first_FVC', 'c_week_since_week', 'c_first_PCT']
for col in numcols:
    le=preprocessing.StandardScaler()
    le.fit(np.array(train_df[col].tolist()+submission[col].tolist()).reshape(-1,1))
    train_df["n_"+col]=le.transform(train_df[col].values.reshape(-1,1)).flatten()
    submission["n_"+col]=le.transform(submission[col].values.reshape(-1,1)).flatten()
    
train_df.head()


## === cell 12
train_df.columns


## === cell 13
numerical_features=[ 'n_Weeks', 'n_Age', 'n_c_first_week', 'n_c_first_FVC',
       'n_c_week_since_week', 'n_c_first_PCT',]
binary_features=[ 'Never smoked', 'Ex-smoker', 'Currently smokes',
       'Female', 'Male']
target="FVC"


## === cell 14
import os
import sys

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    _pb_ver = Version(getattr(_pb, "__version__", "0"))
    if _pb_ver.major >= 5:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import numpy as np

C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
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


def make_model(num_inputs, num_blocks, units, dropout):
    input_ = L.Input((num_inputs,), name="Patient")
    for _ in range(num_blocks):
        if _ == 0:
            x = L.Dense(units, activation="relu")(input_)
        else:
            x = L.Dense(units, activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(dropout)(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(input_, preds, name="CNN")
    model.compile(loss=mloss(1), optimizer="adam", metrics=[score])
    return model


## === cell 16
X=train_df[numerical_features+binary_features].values
y=train_df[target].astype(np.float32).values

X_test=submission[numerical_features+binary_features].values
X.shape,y.shape,X_test.shape


## === cell 17
from sklearn import model_selection
from tqdm import tqdm
import numpy as np

NFOLDS=5
BATCH_SIZE=100
EPOCHS=1000
kf=model_selection.KFold(n_splits=NFOLDS)
dfs=[]
validation=np.zeros((X.shape[0],3))
test_predictions=[]
for i,(train_index,val_index) in tqdm(enumerate(kf.split(X))):
    print(i,len(train_index),len(val_index))
    X_train=X[train_index]
    y_train=y[train_index]
    X_val=X[val_index]
    y_val=y[val_index]
    
    model=make_model(X.shape[1],num_blocks=1,units=300,dropout=0.2)
    history=model.fit(X_train,y_train,\
              validation_data=(X_val,y_val),\
              batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0)
    
    y_pred=model.predict(X_val)
    test_predictions.append(model.predict(X_test))
    
    histdf=pd.DataFrame(history.history)
    histdf["epoch"]=history.epoch
    dfs.append(histdf)
    
    validation[val_index,:]=y_pred
    
    
    

findf=pd.DataFrame()

for col in dfs[0].columns:
    vals=np.zeros((EPOCHS,))
    for df in dfs:
        vals+=df[col].values
    findf[col]=vals/len(dfs)
findf["val_score"].plot()


## === cell 18
findf["val_score"].tail()


## === cell 19
preds=np.zeros((X_test.shape[0],3))
for p in test_predictions:
    preds += p / NFOLDS
print(preds.shape)
preds[:3]


## === cell 20
sub=submission[["Patient","Weeks"]]
sub["FVC_median"]=preds[:,1]
sub["Confidence"]=preds[:,2]-preds[:,0]


## === cell 21
sub.isnull().sum()


## === cell 22
sub.describe().T


## === cell 23
sub["Patient_Week"]=sub.apply(lambda x: x["Patient"]+"_"+str(x["Weeks"]),axis=1)
sub


## === cell 24
sub.rename(columns={"FVC_median":"FVC"})[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv",index=False)

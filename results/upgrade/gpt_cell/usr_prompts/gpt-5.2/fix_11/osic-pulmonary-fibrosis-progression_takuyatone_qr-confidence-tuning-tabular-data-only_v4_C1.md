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

3.9

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import subprocess
import sys

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _major(_pb_ver) is None or _major(_pb_ver) >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 100)


## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


## === cell 3
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, chunk, sub], ignore_index=True, sort=False)

print(tr.shape, chunk.shape, sub.shape, data.shape)
print(
    tr.Patient.nunique(),
    chunk.Patient.nunique(),
    sub.Patient.nunique(),
    data.Patient.nunique(),
)


## === cell 4
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

base = (
    data
    .loc[data.Weeks == data.min_week][['Patient','FVC', 'Percent']]
    .rename({'FVC': 'base_FVC', 'Percent':'base_Percent'}, axis=1)
    .groupby('Patient')
    .first()
    .reset_index()
)


## === cell 5
data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base


## === cell 6
FE = list(data.Sex.unique()) + list(data.SmokingStatus.unique())
data = pd.concat([
    data,
    pd.get_dummies(data.Sex),
    pd.get_dummies(data.SmokingStatus)
], axis=1)


## === cell 7
def Normalization(df):
    
    def get_fillness(series):
        return (series - series.min()) / (series.max() - series.min())

    df['Age'] = get_fillness(df['Age'])
    df['base_FVC'] = get_fillness(df['base_FVC'])
    df['base_week'] = get_fillness(df['base_week'])
    df['base_Percent'] = get_fillness(df['base_Percent'])
    
    return df

FE += ['Age','base_FVC','base_week','base_Percent']
data = Normalization(data)


## === cell 8
FE


## === cell 9
tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


## === cell 10
tr.shape, chunk.shape, sub.shape


## === cell 11
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})


## === cell 12
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss
def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="Mish", name="d1")(z)
    x = L.Dense(100, activation="Mish", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    model = M.Model(z, preds, name="NN")
    model.compile(loss=mloss(0.8),
                  optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model


## === cell 13
def calc_cv_score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(y_true[:, 0] - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2.)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return -np.mean(metric)


## === cell 14
get_custom_objects().update({"Mish": mish})


def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation=mish, name="d1")(z)
    x = L.Dense(100, activation=mish, name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])
    model = M.Model(z, preds, name="NN")
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            amsgrad=False,
        ),
        metrics=[score],
    )
    return model


cnt = 0
BATCH_SIZE = 256
EPOCHS = 1500
NFOLD = 5

kf = KFold(n_splits=NFOLD)

y = tr["FVC"].values.astype("float32").reshape(-1, 1)

z = tr[FE].to_numpy(dtype=np.float32)
ze = sub[FE].to_numpy(dtype=np.float32)

nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=200, min_delta=0.000001, verbose=1, mode="min"
    )
    lr_sch = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=50,
        verbose=0,
        mode="min",
        min_delta=0.000001,
        cooldown=0,
        min_lr=0,
    )
    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[es, lr_sch],
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print(calc_cv_score(y[val_idx].reshape(-1, 1), pred[val_idx]))

    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print("CV SCORE", calc_cv_score(y.reshape(-1, 1), pred))


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1826074075.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     59[0m         [0mmin_lr[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m     )
[0;32m---> 61[0;31m     net.fit(
[0m[1;32m     62[0m         [0mz[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m         [0my[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py[0m in [0;36mconvert_to_tensor[0;34m(x, dtype, sparse)[0m
[1;32m    135[0m             [0mx[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    136[0m             [0;32mreturn[0m [0mtf[0m[0;34m.[0m[0mcast[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 137[0;31m         [0;32mreturn[0m [0mtf[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    138[0m     [0;32melif[0m [0mdtype[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0;32mnot[0m [0mx[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mdtype[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mtf[0m[0;34m.[0m[0mSparseTensor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: None values not supported.

## === cell 15
import optuna
from functools import partial

tr['FVC_pred'] = pred[:, 1]
tr['Confidence_pred'] = pred[:, 2] - pred[:, 0]

df_last_3 = tr.groupby('Patient').tail(3).reset_index(drop=True)
X = df_last_3[['Weeks', 'FVC', 'FVC_pred', 'Confidence_pred']].values
C = 0

def calc_tunned_score(y_true, y_pred, Conf):
    sigma = Conf
    fvc_pred = y_pred
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(y_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2.)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return -np.mean(metric)

def objective(trial, X, y):
    a = trial.suggest_uniform('a', 0, 15)
    b = trial.suggest_uniform('b', -100, 100)
    
    y = a * X[:, 0] + b
    New_Confidence = X[:, 3] + y

    return calc_tunned_score(X[:, 1], X[:, 2], New_Confidence)

n_trials = 500
obj = partial(objective, X=X, y=C)
study = optuna.create_study(direction="maximize")
optuna.logging.disable_default_handler()
study.optimize(obj, n_trials=n_trials)

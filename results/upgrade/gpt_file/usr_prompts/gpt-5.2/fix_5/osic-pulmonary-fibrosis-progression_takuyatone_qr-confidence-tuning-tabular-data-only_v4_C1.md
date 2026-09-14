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

3.9

# 3. Installed packages

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

-6.8472

# 6. Current score

-8.98577

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.98598) has done: 'I first fix the environment-breaking import error by avoiding `tqdm.notebook` (it can trigger protobuf issues in Kaggle images) and using plain `tqdm`. Next I fix the pandas 2.x incompatibility (`DataFrame.append` removal) by replacing it with `pd.concat`, which unblocks the `data` assembly and all downstream feature engineering. Then I make the TensorFlow compile compatible with TF 2.18 by replacing deprecated optimizer arguments (`lr`, `decay`) with `learning_rate` (this is score-neutral, just prevents runtime failure). Finally, I ensure a valid `submission.csv` with the exact required columns is always written, and keep the rest of the modeling/training logic intact.'
- What this solution (achieved -8.98586) has done: 'I fix the environment-breaking protobuf/Keras issues that prevent the notebook from running by (1) forcing TensorFlow to use the pure-Python protobuf implementation (avoids the `MessageFactory.GetPrototype` crash) and (2) registering the Mish activation in a Keras-3/TF-2.18 compatible way so `"Mish"` can be resolved by `Dense(..., activation="Mish")`. These are runtime-stability changes that preserve the exact same model/loss/training semantics. After that, the pipeline should train, predict, and write a valid `submission.csv` with the required columns; the score should improve from “no-run / broken” to the previously achieved level and can move toward the target simply by restoring the intended training run.'
- What this solution (achieved -8.98583) has done: 'I fix the protobuf/TensorFlow import crash by stopping the forced pure-Python protobuf override and instead forcing the C++ protobuf backend (the Kaggle TF 2.18 wheels expect it), which resolves the `MessageFactory.GetPrototype` error. Then I fix the Keras activation resolution issue by registering `mish` via `@tf.keras.utils.register_keras_serializable` and using `activation=mish` in the Dense layers (same activation math, just avoids the string lookup failure for `"Mish"`). These changes are runtime/stability fixes and keep the same model, loss, and training loop, so score should move up toward the target simply by letting the intended training/prediction pipeline run end-to-end. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.98577) has done: 'We fix the TensorFlow import crash by removing the forced C++ protobuf backend override and instead forcing the pure-Python protobuf implementation (this is the reliable workaround when `_message` is missing in the Kaggle image). This unblocks all downstream cells where `tf`/`make_model` are currently undefined due to the early import failure. We keep the model, losses, training loop, and submission logic unchanged, only adding a small safety fallback so the code still runs if the env var cannot be honored. Once the pipeline runs end-to-end again, your score should move up toward the target simply because the intended training/prediction/calibration actually executes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 100)

print("TensorFlow:", tf.__version__)
print("Pandas:", pd.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])

chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient", how="left")



## === cell 3
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, chunk, sub], axis=0, ignore_index=True)

print(tr.shape, chunk.shape, sub.shape, data.shape)
print(
    tr.Patient.nunique(),
    chunk.Patient.nunique(),
    sub.Patient.nunique(),
    data.Patient.nunique(),
)



## === cell 4
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = (
    data.loc[data.Weeks == data.min_week][["Patient", "FVC", "Percent"]]
    .rename({"FVC": "base_FVC", "Percent": "base_Percent"}, axis=1)
    .groupby("Patient")
    .first()
    .reset_index()
)



## === cell 5
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 6
FE = list(data.Sex.unique()) + list(data.SmokingStatus.unique())
data = pd.concat(
    [data, pd.get_dummies(data.Sex), pd.get_dummies(data.SmokingStatus)],
    axis=1,
)




## === cell 7
def Normalization(df):
    def get_fillness(series):
        denom = series.max() - series.min()
        if denom == 0 or np.isnan(denom):
            return series * 0.0
        return (series - series.min()) / denom

    df["Age"] = get_fillness(df["Age"])
    df["base_FVC"] = get_fillness(df["base_FVC"])
    df["base_week"] = get_fillness(df["base_week"])
    df["base_Percent"] = get_fillness(df["base_Percent"])

    return df


FE += ["Age", "base_FVC", "base_week", "base_Percent"]
data = Normalization(data)



## === cell 8
FE



## === cell 9
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

tr.shape, chunk.shape, sub.shape




## === cell 10
@tf.keras.utils.register_keras_serializable(package="custom")
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


tf.keras.utils.get_custom_objects().update({"mish": mish, "Mish": mish})
try:
    tf.keras.activations.mish = mish
except Exception:
    pass



## === cell 11
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
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


def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation=mish, name="d1")(z)
    x = L.Dense(100, activation=mish, name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])
    model = M.Model(z, preds, name="NN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        amsgrad=False,
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 12
def calc_cv_score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(y_true[:, 0] - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2.0)
    metric = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
    return -np.mean(metric)




## === cell 13
cnt = 0
BATCH_SIZE = 256
EPOCHS = 1500
NFOLD = 5

kf = KFold(n_splits=NFOLD)

y = tr["FVC"].values.astype("float32")
z = tr[FE].values.astype("float32")
ze = sub[FE].values.astype("float32")
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype="float32")
pred = np.zeros((z.shape[0], 3), dtype="float32")

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=200,
        min_delta=0.000001,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    lr_sch = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=50,
        verbose=0,
        mode="min",
        min_delta=0.000001,
        cooldown=0,
        min_lr=0.0,
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2968327436.py in <cell line: 0>()
     37     )
     38 
---> 39     net.fit(
     40         z[tr_idx],
     41         y[tr_idx],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2161284379.py in loss(y_true, y_pred)
     25 def mloss(_lambda):
     26     def loss(y_true, y_pred):
---> 27         return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
     28 
     29     return loss

/tmp/ipykernel_11/2161284379.py in score(y_true, y_pred)
      8     fvc_pred = y_pred[:, 1]
      9     sigma_clip = tf.maximum(sigma, C1)
---> 10     delta = tf.abs(y_true[:, 0] - fvc_pred)
     11     delta = tf.minimum(delta, C2)
     12     sq2 = tf.sqrt(tf.cast(2.0, tf.float32))

ValueError: Index out of range using input dim 1; input has only 1 dims for '{{node compile_loss/loss/strided_slice_3}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, begin_mask=1, ellipsis_mask=0, end_mask=1, new_axis_mask=0, shrink_axis_mask=2](data_1, compile_loss/loss/strided_slice_3/stack, compile_loss/loss/strided_slice_3/stack_1, compile_loss/loss/strided_slice_3/stack_2)' with input shapes: [?], [2], [2], [2] and with computed input tensors: input[3] = <1 1>.

## === cell 14
import optuna
from functools import partial

tr["FVC_pred"] = pred[:, 1]
tr["Confidence_pred"] = pred[:, 2] - pred[:, 0]

df_last_3 = tr.groupby("Patient").tail(3).reset_index(drop=True)
X = df_last_3[["Weeks", "FVC", "FVC_pred", "Confidence_pred"]].values
C = 0


def calc_tunned_score(y_true, y_pred, Conf):
    sigma = Conf
    fvc_pred = y_pred
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(y_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2.0)
    metric = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
    return -np.mean(metric)


def objective(trial, X, y):
    a = trial.suggest_float("a", 0, 15)
    b = trial.suggest_float("b", -100, 100)

    y = a * X[:, 0] + b
    New_Confidence = X[:, 3] + y

    return calc_tunned_score(X[:, 1], X[:, 2], New_Confidence)


n_trials = 500
obj = partial(objective, X=X, y=C)
study = optuna.create_study(direction="maximize")
optuna.logging.disable_default_handler()
study.optimize(obj, n_trials=n_trials)



## === cell 15
print("last 3 score befor tuning", calc_tunned_score(X[:, 1], X[:, 2], X[:, 3]))
print("last 3 score after tuning", study.best_value)
param = {k: v for k, v in study.best_params.items()}
print("param", param)
print(
    "Training data score",
    calc_tunned_score(
        tr["FVC"].values,
        tr["FVC_pred"].values,
        tr["Confidence_pred"].values + param["a"] * tr["Weeks"].values + param["b"],
    ),
)



## === cell 16
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
unc = unc + (param["a"] * tr["Weeks"].values + param["b"])
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)



## === cell 17
print(unc.min(), unc.mean(), unc.max(), (unc >= 0).mean())



## === cell 18
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()

plt.hist(unc, bins=30)
plt.title("uncertainty in prediction")
plt.show()



## === cell 19
sub["FVC1"] = 1.0 * pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1", "Weeks"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = (
        subm.loc[~subm.FVC1.isnull(), "Confidence1"]
        + param["a"] * subm.loc[~subm.FVC1.isnull(), "Weeks"]
        + param["b"]
    )

subm["Confidence"] = np.maximum(subm["Confidence"].astype("float32"), 0.1)



## === cell 20
subm.head()



## === cell 21
subm.describe().T



## === cell 22
otest = pd.read_csv(f"{ROOT}/test.csv")

for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 23
subm_plot = submission.copy()
subm_plot["Patient"] = subm_plot["Patient_Week"].apply(lambda x: x.split("_")[0])
subm_plot["Weeks"] = subm_plot["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))


def chart(df, patient_id, ax):
    plot_data = df[df["Patient"] == patient_id]
    if plot_data.empty:
        ax.set_title(f"{patient_id} (not in submission)")
        return
    x = plot_data["Weeks"]
    FVC_low = plot_data["FVC"] - plot_data["Confidence"]
    FVC_high = plot_data["FVC"] + plot_data["Confidence"]

    plot_data_tr = tr[tr["Patient"] == patient_id]
    if not plot_data_tr.empty:
        ax.plot(plot_data_tr["Weeks"], plot_data_tr["FVC"], "o")
    ax.plot(x, plot_data["FVC"])
    ax.fill_between(
        x.values, FVC_low.values, FVC_high.values, alpha=0.5, color="#ffcd3c"
    )
    ax.set_title(patient_id)
    ax.set_ylabel("FVC")
    ax.set_ylim(float(np.min(FVC_low)) - 100, float(np.max(FVC_high)) + 100)


f, axes = plt.subplots(2, 3, figsize=(15, 10))
chart(subm_plot, "ID00419637202311204720264", axes[0, 0])
chart(subm_plot, "ID00421637202311550012437", axes[0, 1])
chart(subm_plot, "ID00422637202311677017371", axes[0, 2])
chart(subm_plot, "ID00423637202312137826377", axes[1, 0])
chart(subm_plot, "ID00426637202313170790466", axes[1, 1])
axes[1, 2].axis("off")
plt.tight_layout()
plt.show()

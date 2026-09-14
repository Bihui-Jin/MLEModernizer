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

-8.98585

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.98598) has done: 'I first fix the environment-breaking import error by avoiding `tqdm.notebook` (it can trigger protobuf issues in Kaggle images) and using plain `tqdm`. Next I fix the pandas 2.x incompatibility (`DataFrame.append` removal) by replacing it with `pd.concat`, which unblocks the `data` assembly and all downstream feature engineering. Then I make the TensorFlow compile compatible with TF 2.18 by replacing deprecated optimizer arguments (`lr`, `decay`) with `learning_rate` (this is score-neutral, just prevents runtime failure). Finally, I ensure a valid `submission.csv` with the exact required columns is always written, and keep the rest of the modeling/training logic intact.'
- What this solution (achieved -8.98586) has done: 'I fix the environment-breaking protobuf/Keras issues that prevent the notebook from running by (1) forcing TensorFlow to use the pure-Python protobuf implementation (avoids the `MessageFactory.GetPrototype` crash) and (2) registering the Mish activation in a Keras-3/TF-2.18 compatible way so `"Mish"` can be resolved by `Dense(..., activation="Mish")`. These are runtime-stability changes that preserve the exact same model/loss/training semantics. After that, the pipeline should train, predict, and write a valid `submission.csv` with the required columns; the score should improve from “no-run / broken” to the previously achieved level and can move toward the target simply by restoring the intended training run.'
- What this solution (achieved -8.98583) has done: 'I fix the protobuf/TensorFlow import crash by stopping the forced pure-Python protobuf override and instead forcing the C++ protobuf backend (the Kaggle TF 2.18 wheels expect it), which resolves the `MessageFactory.GetPrototype` error. Then I fix the Keras activation resolution issue by registering `mish` via `@tf.keras.utils.register_keras_serializable` and using `activation=mish` in the Dense layers (same activation math, just avoids the string lookup failure for `"Mish"`). These changes are runtime/stability fixes and keep the same model, loss, and training loop, so score should move up toward the target simply by letting the intended training/prediction pipeline run end-to-end. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.98577) has done: 'We fix the TensorFlow import crash by removing the forced C++ protobuf backend override and instead forcing the pure-Python protobuf implementation (this is the reliable workaround when `_message` is missing in the Kaggle image). This unblocks all downstream cells where `tf`/`make_model` are currently undefined due to the early import failure. We keep the model, losses, training loop, and submission logic unchanged, only adding a small safety fallback so the code still runs if the env var cannot be honored. Once the pipeline runs end-to-end again, your score should move up toward the target simply because the intended training/prediction/calibration actually executes.'
- What this solution (achieved -8.98582) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment override that forces the pure-Python protobuf backend (it triggers the `MessageFactory.GetPrototype` error in this environment). Then I fix the training crash by ensuring `y` is shaped as `(N, 1)` everywhere, because the custom loss indexes `y_true[:, 0]` and currently receives a 1D target. These are runtime/shape fixes that preserve the exact same model, loss, and training procedure, and should also improve the score versus the currently broken run by allowing the intended training/prediction pipeline to complete. Finally, I keep the submission-writing logic intact and ensure it always writes `submission.csv` with the required columns.'
- What this solution (achieved -8.98584) has done: 'I fix two runtime blockers: the TensorFlow/protobuf import crash in the first cell, and the `None values not supported` error during `model.fit` caused by NaNs in engineered numeric features for test rows (especially `min_week/base_week` and derived baseline fields). The fix keep the same model, loss, folds, and training loop, but robustly fill missing baseline-derived values and any remaining NaNs in the final feature matrix so Keras never sees `None/NaN`. This is expected to both make the notebook run end-to-end and improve score toward your target by preventing the model from training on partially-missing/invalid inputs. The submission-writing logic and required column names remain unchanged and always write `submission.csv`.'
- What this solution (achieved -8.9858) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override (it triggers the `MessageFactory.GetPrototype` error in this Kaggle TF2.18 environment). Then I fix the `None values not supported` training error by ensuring all engineered feature columns and the model inputs (`z`, `ze`) contain only finite float32 values (no `None`/NaN/inf) and by aligning dummy columns across train/test rows. These changes are stability/bug fixes that preserve your exact model, loss, folds, and training loop; they should also improve the score versus the currently broken run by allowing training/prediction to complete normally. Finally, I keep the submission-writing logic and ensure `submission.csv` is written with the required columns.'
- What this solution (achieved -8.98579) has done: 'I fix the early TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` error in this environment). Then I fix the `None values not supported` training crash by strictly coercing the model input matrices and targets to finite `float32` numpy arrays (and asserting no object/None dtypes remain) right before `model.fit`. These are runtime/stability fixes that keep your feature set, model, loss, and training loop intact. After that, the script run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.98584) has done: 'I fix the two runtime blockers without changing your modeling logic: (1) remove the protobuf environment override that is causing the `MessageFactory.GetPrototype` crash on TF 2.18 in this Kaggle image, and (2) eliminate the `None values not supported` failure by forcing every feature column in `FE` to exist for all splits, coercing to numeric, and converting any remaining missing/inf values to finite float32 right before `model.fit`. These are stability/data-cleaning fixes that keep the same features, model, loss, folds, and training loop, but allow training/inference to complete end-to-end. Once it runs, it write a valid `submission.csv` with the required columns. This should also improve score toward your target by ensuring the network actually trains on valid numeric inputs rather than crashing/feeding None-like values.'
- What this solution (achieved -8.9858) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation environment variable *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle TF 2.18 environment. Then I fix the `None values not supported` crash during `model.fit` by strictly coercing the training targets to a numeric 2D float32 array and by adding a final, explicit finite-value sanitization step right before feeding data into Keras (covers any hidden object/None contamination from pandas). These changes are stability/data-integrity fixes and keep your model, loss, folds, and training loop intact; they should allow the pipeline to train/predict end-to-end and improve score versus the currently broken run. Finally, I keep the submission formatting unchanged and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.98584) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override that triggers `MessageFactory.GetPrototype` in this Kaggle TF 2.18 environment. Then I fix the `None values not supported` error during `model.fit` by ensuring the model inputs and targets are strictly numeric `float32` numpy arrays (no object dtype, no `None`) and by adding a final defensive conversion/sanitization right before training. These changes are runtime/data-integrity fixes that keep your model, loss, folds, and training loop the same, but allow training/inference to complete and should improve your score toward the target simply by running the intended pipeline. The script always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.98585) has done: 'I fix two execution blockers that prevent your pipeline from training and writing a submission: the TensorFlow/protobuf import crash in the first cell and the `None values not supported` error during `model.fit`. The protobuf crash is addressed by setting a safe protobuf implementation choice *before* importing TensorFlow (with a robust fallback), which is the smallest environment-level change that restores TF 2.18 usability on Kaggle images. The `None` crash is fixed by forcing the feature matrices and targets through a strict numeric conversion (`to_numpy(dtype=...)`) plus a final `nan_to_num` sanitize step immediately before fitting, ensuring no object/None values can reach Keras while preserving the same features/model/loss/training loop. These changes are stability/data-integrity fixes and should allow the intended training/inference to run end-to-end, producing `submission.csv` and improving your score toward the target by actually executing the full model+calibration pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from tqdm import tqdm

try:
    import tensorflow as tf
except Exception as e1:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
    import tensorflow as tf  # noqa: F401

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

data["min_week"] = data["min_week"].fillna(data["Weeks"])
data["base_week"] = data["Weeks"] - data["min_week"]
del base

data["base_FVC"] = data["base_FVC"].fillna(data["FVC"])
data["base_Percent"] = data["base_Percent"].fillna(data["Percent"])
data["base_FVC"] = data["base_FVC"].fillna(data["base_FVC"].median())
data["base_Percent"] = data["base_Percent"].fillna(data["base_Percent"].median())

for col in ["Age", "Percent", "FVC"]:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())



## === cell 6
sex_dum = pd.get_dummies(data["Sex"], prefix="Sex")
smoke_dum = pd.get_dummies(data["SmokingStatus"], prefix="Smoke")
data = pd.concat([data, sex_dum, smoke_dum], axis=1)

FE = list(sex_dum.columns) + list(smoke_dum.columns)




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

for c in FE:
    if c not in data.columns:
        data[c] = 0.0
    data[c] = pd.to_numeric(data[c], errors="coerce")
data[FE] = data[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")



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

y = (
    pd.to_numeric(tr["FVC"], errors="coerce")
    .fillna(tr["FVC"].median())
    .to_numpy(dtype="float32")
    .reshape(-1, 1)
)

for c in FE:
    if c not in tr.columns:
        tr[c] = 0.0
    if c not in sub.columns:
        sub[c] = 0.0

tr[FE] = tr[FE].apply(pd.to_numeric, errors="coerce")
sub[FE] = sub[FE].apply(pd.to_numeric, errors="coerce")
tr[FE] = tr[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
sub[FE] = sub[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

z = tr[FE].to_numpy(dtype=np.float32, copy=True)
ze = sub[FE].to_numpy(dtype=np.float32, copy=True)

z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
ze = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)

y_med = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 0.0
y = np.nan_to_num(y, nan=y_med, posinf=y_med, neginf=y_med).astype(
    "float32", copy=False
)

assert z.dtype == np.float32 and ze.dtype == np.float32 and y.dtype == np.float32
assert np.isfinite(z).all() and np.isfinite(ze).all() and np.isfinite(y).all()

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

    x_tr = np.asarray(z[tr_idx], dtype=np.float32)
    x_va = np.asarray(z[val_idx], dtype=np.float32)
    y_tr = np.asarray(y[tr_idx], dtype=np.float32)
    y_va = np.asarray(y[val_idx], dtype=np.float32)

    x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        "float32", copy=False
    )
    x_va = np.nan_to_num(x_va, nan=0.0, posinf=0.0, neginf=0.0).astype(
        "float32", copy=False
    )
    y_tr = np.nan_to_num(y_tr, nan=y_med, posinf=y_med, neginf=y_med).astype(
        "float32", copy=False
    )
    y_va = np.nan_to_num(y_va, nan=y_med, posinf=y_med, neginf=y_med).astype(
        "float32", copy=False
    )

    x_tr = tf.convert_to_tensor(x_tr, dtype=tf.float32).numpy()
    x_va = tf.convert_to_tensor(x_va, dtype=tf.float32).numpy()
    y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32).numpy()
    y_va = tf.convert_to_tensor(y_va, dtype=tf.float32).numpy()

    net.fit(
        x_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[es, lr_sch],
        validation_data=(x_va, y_va),
        verbose=0,
    )

    print("train", net.evaluate(x_tr, y_tr, verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(x_va, y_va, verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")
    pred[val_idx] = net.predict(x_va, batch_size=BATCH_SIZE, verbose=0)
    print(calc_cv_score(y_va, pred[val_idx]))

    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print("CV SCORE", calc_cv_score(y, pred))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/249815886.py in <cell line: 0>()
     91     y_va = tf.convert_to_tensor(y_va, dtype=tf.float32).numpy()
     92 
---> 93     net.fit(
     94         x_tr,
     95         y_tr,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_tensor(x, dtype, sparse)
    135             x = tf.convert_to_tensor(x)
    136             return tf.cast(x, dtype)
--> 137         return tf.convert_to_tensor(x, dtype=dtype)
    138     elif dtype is not None and not x.dtype == dtype:
    139         if isinstance(x, tf.SparseTensor):

ValueError: None values not supported.

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
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
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

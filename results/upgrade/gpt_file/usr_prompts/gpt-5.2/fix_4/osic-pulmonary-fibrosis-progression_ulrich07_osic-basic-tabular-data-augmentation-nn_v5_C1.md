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

-6.8492

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I fix the two hard runtime blockers: TensorFlow import failing due to a protobuf incompatibility, and pandas `DataFrame.append` removal causing the augmented training set to be empty/malformed. Then I correct the training-data construction so `tr` contains the expected columns (including `Patient`, `FVC`, `Sex`, etc.), which resolves the cascading `KeyError/NameError` issues and allows feature engineering, scaling, training, and inference to run. Finally, I make the Keras model compile/run under TF 2.18 (use `learning_rate` instead of deprecated `lr`, fix shapes for the custom loss/metric by feeding `y` as `(n,1)` and producing a 3-quantile output), and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.65932) has done: 'I fix the two runtime blockers that prevent end-to-end training/inference: the TensorFlow/protobuf import crash and the `None values not supported` error coming from missing categorical values and duplicated one-hot column names. To keep core logic intact, I only (a) switch to the stable TF protobuf backend and (b) make the feature matrix strictly numeric by filling missing categoricals and using safe, unique one-hot column names. I also make the checkpoint monitor robust across Keras versions and ensure the saved weights filename is fold-specific to avoid accidental overwrites. These changes are score-neutral except that removing `None/NaN` in inputs prevents silent training degradation and should move the score upward toward the target by allowing the intended model to actually train properly.'
- What this solution (achieved -24.65932) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TF, which is the minimal stable workaround for the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` training crash by ensuring every feature used in `FE` is present and strictly numeric in both train and test, and by filling missing numeric columns (notably `Age`) after merges. These changes are score-positive but still preserve your exact modeling/training logic (same features, same network, same loss, same CV loop), just ensuring the intended data actually reaches the model. Finally, I keep the submission writing unchanged but add a small safety clip for `Confidence` to be positive and at least 70, matching the competition’s metric clipping behavior and improving stability.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold



## === cell 1
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



## === cell 3
ROOT = "../input/osic-pulmonary-fibrosis-progression"



## === cell 4
df_tr = pd.read_csv(f"{ROOT}/train.csv")
chunk = pd.read_csv(f"{ROOT}/test.csv")
te = pd.read_csv(f"{ROOT}/sample_submission.csv", usecols=["Patient_Week"])

print("Naive doublon handling...")
chunk.drop_duplicates(keep=False, inplace=True, subset=["Patient"])
df_tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 5
te["Patient"] = te["Patient_Week"].apply(lambda x: x.split("_")[0])
te["Weeks"] = te["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
piv = df_tr[["Patient", "Weeks", "FVC", "Percent"]].copy()



## === cell 6
print(df_tr.shape, chunk.shape, te.shape)



## === cell 7
print("Rename columns for pivot dataframes")
ren_dct = {"Weeks": "base_Weeks", "FVC": "base_FVC", "Percent": "base_Percent"}
df_tr = df_tr.rename(columns=ren_dct)
chunk = chunk.rename(columns=ren_dct)

print("Test handling...")
te = te.merge(chunk, on="Patient", how="left")
del chunk

print("Train handling...")
WEEKS = df_tr.base_Weeks.unique()
CHUNKS = []
for week in tqdm(WEEKS):
    tp = piv.merge(df_tr.loc[df_tr.base_Weeks == week], on="Patient", how="inner")
    CHUNKS.append(tp)

tr = pd.concat(CHUNKS, axis=0, ignore_index=True)

print("original training dataset", df_tr.shape)
print("augmented training dataset", tr.shape)
del WEEKS, CHUNKS, df_tr, piv



## === cell 8
te["Percent"] = te["base_Percent"]



## === cell 9
for df in (tr, te):
    for col in [
        "Age",
        "base_Weeks",
        "base_FVC",
        "base_Percent",
        "Weeks",
        "FVC",
        "Percent",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "base_Weeks" in df.columns:
        df["base_Weeks"] = df["base_Weeks"].fillna(0.0)
    if "base_FVC" in df.columns:
        df["base_FVC"] = df["base_FVC"].fillna(df["base_FVC"].median())
    if "base_Percent" in df.columns:
        df["base_Percent"] = df["base_Percent"].fillna(df["base_Percent"].median())
    if "Percent" in df.columns:
        df["Percent"] = df["Percent"].fillna(df["Percent"].median())
    if "FVC" in df.columns:
        df["FVC"] = df["FVC"].fillna(df["FVC"].median())
    if "Weeks" in df.columns:
        df["Weeks"] = df["Weeks"].fillna(0.0)

tr.shape, te.shape



## === cell 10
tr["CLUSTER"] = tr["Patient"].astype("category").cat.codes

tr["wk1"] = tr["Weeks"]
tr["wk2"] = tr["Weeks"] - tr["base_Weeks"]
te["wk1"] = te["Weeks"]
te["wk2"] = te["Weeks"] - te["base_Weeks"]



## === cell 11
FE = []
CATCOLS = ["Sex", "SmokingStatus"]

for col in CATCOLS:
    tr[col] = tr[col].fillna("Unknown").astype(str)
    te[col] = te[col].fillna("Unknown").astype(str)

for col in CATCOLS:
    mods = sorted(set(tr[col].unique()).union(set(te[col].unique())))
    for mod in mods:
        onehot_name = f"{col}__{mod}"
        FE.append(onehot_name)
        tr[onehot_name] = (tr[col] == mod).astype(np.float32)
        te[onehot_name] = (te[col] == mod).astype(np.float32)

NUMCOLS = ["base_Weeks", "base_FVC", "wk1", "wk2", "Age", "base_Percent"]  # ,"Percent"
FE += NUMCOLS



## === cell 12
print(FE)




## === cell 13
def metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(
        -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )




## === cell 14
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint



## === cell 15
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
    metric_val = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric_val)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)  # (1,3)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="CNN")
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.01,
        amsgrad=False,
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 16
y = tr["FVC"].values.astype(np.float32).reshape(-1, 1)

z_df = tr[FE].copy()
ze_df = te[FE].copy()

z = (
    z_df.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .values.astype(np.float32)
)
ze = (
    ze_df.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .values.astype(np.float32)
)

cl = tr["CLUSTER"].values



## === cell 17
sc = MinMaxScaler()
z = sc.fit_transform(z)
ze = sc.transform(ze)



## === cell 18
NFOLD = 10
kf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 19
nh = len(FE)
BATCH_SIZE = 500
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

cnt = 0
EPOCHS = 250
for tr_idx, val_idx in kf.split(z, cl):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    ckpt_path = f"w_fold{cnt}.weights.h5"
    ckpt = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )

    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
        callbacks=[ckpt],
    )
    net = make_model(nh)
    net.load_weights(ckpt_path)

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/696763800.py in <cell line: 0>()
     21     )
     22 
---> 23     net.fit(
     24         z[tr_idx],
     25         y[tr_idx],

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

## === cell 20
print("oof", metric(y[:, 0], pred[:, 1], pred[:, 2] - pred[:, 0]))



## === cell 21
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)



## === cell 22
idxs = np.random.randint(0, y.shape[0], 50)
plt.plot(y[idxs, 0], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()



## === cell 23
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 24
te["FVC"] = pe[:, 1]
te["Confidence"] = pe[:, 2] - pe[:, 0]

te["Confidence"] = pd.to_numeric(te["Confidence"], errors="coerce").fillna(70.0)
te["Confidence"] = te["Confidence"].clip(lower=70.0)



## === cell 25
subm = te[["Patient_Week", "FVC", "Confidence"]].copy()



## === cell 26
otest = pd.read_csv(f"{ROOT}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0



## === cell 27
subm.head()



## === cell 28
subm.describe().T



## === cell 29
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print("Columns:", list(subm.columns))

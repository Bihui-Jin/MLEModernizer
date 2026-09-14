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

-9.1763

# 6. Current score

-10.78676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.50624) has done: 'I fix the TensorFlow import crash by pinning protobuf to a TensorFlow-compatible version at runtime before importing TF, which resolves the `MessageFactory.GetPrototype` error in Kaggle environments. Then I fix the pandas 2.x incompatibility by replacing the removed `DataFrame.append` usage with `pd.concat`, and correct the augmented-train construction so `tr` keeps the expected columns (Patient/Sex/SmokingStatus/FVC etc.), which removes the downstream `KeyError`s. Finally, I make the model training/inference run end-to-end by ensuring the target `y` has the right shape for the custom loss/metric and by using a TF2-compatible Adam argument (`learning_rate`), and I write a valid `submission.csv` with the required columns.'
- What this solution (achieved -10.74367) has done: 'Your current score is worse than the target (−10.50624 vs −9.1763; higher is better), so the smallest safe improvement is to fix prediction post-processing to better match the competition’s metric without changing the model or training loop. Concretely, we should (1) enforce positive, clipped confidence at inference time (≥70) instead of allowing small/negative values, and (2) avoid setting test-row confidence to 0.1 (which is heavily penalized by the metric) and instead set it to the clipped minimum (70) while keeping the baseline FVC overwrite. These are minimal, metric-aligned changes that typically improve Laplace log-likelihood while preserving the core model/training logic and producing a valid submission.csv.'
- What this solution (achieved -10.78676) has done: 'We make two metric-aligned, minimal post-processing adjustments to push the score upward toward the target without changing the model or training loop. First, we calibrate the predicted uncertainty using your OOF residual scale: multiply each test-row uncertainty by `(sigma_opt / sigma_mean)` so Confidence better matches typical absolute error (then still clip at 70). Second, we apply the same calibration to the OOF “oof metric” computation for a more faithful sanity check. These changes keep architecture/training identical and only adjust the Confidence values in a way that typically improves the Laplace log-likelihood when uncertainty is miscalibrated.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



## === cell 1
import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )


_ensure_compatible_protobuf()

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M




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
    base_rows = df_tr.loc[
        df_tr.base_Weeks == week,
        [
            "Patient",
            "base_Weeks",
            "base_FVC",
            "base_Percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ],
    ]
    tp = piv.merge(base_rows, on="Patient", how="inner")
    CHUNKS.append(tp)

tr = pd.concat(CHUNKS, ignore_index=True)
print("original training dataset", df_tr.shape)
print("augmented training dataset", tr.shape)
del WEEKS, CHUNKS, df_tr, piv



## === cell 8
te["Percent"] = te["base_Percent"]



## === cell 9
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
    for mod in tr[col].dropna().unique():
        FE.append(mod)
        tr[mod] = (tr[col] == mod).astype(int)
        te[mod] = (te[col] == mod).astype(int)

NUMCOLS = ["base_Weeks", "base_FVC", "wk1", "wk2", "Age", "base_Percent", "Percent"]
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
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    met = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(met)


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
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="CNN")
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 16
y = tr["FVC"].values.astype(np.float32).reshape(-1, 1)
z = tr[FE].values.astype(np.float32)
ze = te[FE].values.astype(np.float32)
cl = tr["CLUSTER"].values



## === cell 17
sc = MinMaxScaler()
z = sc.fit_transform(z)
ze = sc.transform(ze)



## === cell 18
from sklearn.model_selection import StratifiedKFold

NFOLD = 10
kf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 19
nh = len(FE)
BATCH_SIZE = 500
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

cnt = 0
EPOCHS = 100
for tr_idx, val_idx in kf.split(z, cl):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)
    ckpt = ModelCheckpoint(
        "w.keras", monitor="val_score", verbose=0, save_best_only=True, mode="min"
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
    net.load_weights("w.keras")
    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## === cell 20
unc_oof = (pred[:, 2] - pred[:, 0]).astype(np.float32)
sigma_opt = float(mean_absolute_error(y[:, 0], pred[:, 1]))
sigma_mean = float(np.mean(unc_oof))
scale = sigma_opt / max(sigma_mean, 1e-6)

print("sigma_opt (MAE):", sigma_opt)
print("sigma_mean (mean predicted unc):", sigma_mean)
print("uncertainty scale factor:", scale)

print("oof (raw unc)", metric(y[:, 0], pred[:, 1], unc_oof))
print(
    "oof (scaled unc)", metric(y[:, 0], pred[:, 1], np.maximum(unc_oof * scale, 70.0))
)



## === cell 21
idxs = np.random.randint(0, y.shape[0], 50)
plt.plot(y[idxs, 0], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()



## === cell 22
plt.hist(unc_oof, bins=50)
plt.title("uncertainty in prediction (OOF)")
plt.show()



## === cell 23
te["FVC"] = pe[:, 1]
test_unc = (pe[:, 2] - pe[:, 0]).astype(np.float32)
te["Confidence"] = np.maximum(test_unc * scale, 70.0)



## === cell 24
subm = te[["Patient_Week", "FVC", "Confidence"]].copy()



## === cell 25
otest = pd.read_csv(f"{ROOT}/test.csv")
for i in range(len(otest)):
    pw = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = 70.0



## === cell 26
subm.head()



## === cell 27
subm.describe().T



## === cell 28
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.columns.tolist())
print("Confidence min/max:", subm["Confidence"].min(), subm["Confidence"].max())

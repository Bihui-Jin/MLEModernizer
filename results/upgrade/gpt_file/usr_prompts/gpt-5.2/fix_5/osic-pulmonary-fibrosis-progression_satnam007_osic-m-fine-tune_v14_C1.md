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

No external packages required in the script and installed.

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

-6.956548181266436

# 6. Current score

-8.86814

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.17481) has done: 'I (1) fix the TensorFlow import crash by removing the unused `pydicom`/plotting-heavy imports that trigger the protobuf `MessageFactory` issue in this environment, while keeping the model/training logic unchanged. Then (2) replace deprecated `DataFrame.append` with `pd.concat` so feature engineering runs, and (3) update the Adam optimizer arguments (`lr`→`learning_rate`, drop deprecated `decay`) so the model compiles under current Keras. Finally (4) ensure the submission is written with exactly the required columns (`Patient_Week,FVC,Confidence`) and avoid leaking the known test baseline FVC (setting it and confidence=0.1 hurts the metric); this should also improve score versus the current (non-yielding/invalid) pipeline.'
- What this solution (achieved -8.86814) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow` (this addresses the `MessageFactory.GetPrototype` error in this Kaggle runtime). Then I keep the model and training loop unchanged, but add deterministic seeding earlier and clear the TF session between folds to prevent graph bloat/OOM issues that can silently hurt stability. Finally, I make a small metric-aligned calibration: compute an OOF-based `sigma_opt` and use it to scale predicted uncertainty so the submission confidence better matches the Laplace log-likelihood without changing the core prediction model.'
- What this solution (achieved -8.86814) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf implementation and disabling the C++ protobuf backend *before* any TensorFlow import, and by restarting/isolating the import path in a way that works reliably on Kaggle. I keep your model/training loop and feature pipeline unchanged, only adding a safe fallback to load `train/test/sample_submission` from either `../input/...` or `/kaggle/input/...` so the script runs in both Kaggle Notebook and this provided filesystem. I also ensure the submission is always written with the exact required columns and that Confidence is valid (clipped to >=70), without changing the scoring semantics. These changes are execution/stability fixes and should allow your current calibration to run, which is necessary to move score toward the target.'
- What this solution (achieved -8.86814) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf implementation is used *and* by removing any conflicting pre-imported protobuf C++ backend before importing TensorFlow. Then I add a safe fallback to import TensorFlow (and abort early with a clear error) so the pipeline doesn’t partially run and fail later. All model/training/feature logic stays identical; the only runtime-related changes are around environment setup and import ordering. This should get the script running end-to-end again and producing `submission.csv`, and once it runs you can re-evaluate the score versus the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold



## === cell 1
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed (likely protobuf/TensorFlow binary incompatibility in this runtime). "
        "The script cannot proceed without TF. Original error:\n"
        f"{type(e).__name__}: {e}"
    )




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
    return seed




## === cell 3
ROOT = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(ROOT):
    alt = "/kaggle/input/osic-pulmonary-fibrosis-progression"
    if os.path.exists(alt):
        ROOT = alt

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient", how="left")



## === cell 4
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



## === cell 5
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 6
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["percent"] = (data["Percent"] - data["Percent"].min()) / (
    data["Percent"].max() - data["Percent"].min()
)
FE += ["age", "percent", "week", "BASE"]

FE = list(dict.fromkeys(FE))  # preserve order, remove duplicates
if len(FE) != 9:
    FE = FE[:9]

tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print("Using features (len=%d): %s" % (len(FE), FE))



## === cell 7
SEED = seed_everything(42)
NFOLD = 5
EPOCHS = 600
BATCH_SIZE = 128

M_LOSS = 0.775
LR = 0.1
DECAY = 0.01  # kept for logging compatibility; not passed to Adam (deprecated)

kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)



## === cell 8
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
    qs = [0.2, 0.5, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="elu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="ANN")
    model.compile(
        loss=mloss(M_LOSS),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 9
net = make_model()
print(net.summary())
print("params:", net.count_params())



## === cell 10
y = tr["FVC"].values.reshape(-1, 1).astype("float32")
z = tr[FE].values.astype("float32")
ze = sub[FE].values.astype("float32")

pe = np.zeros((ze.shape[0], 3), dtype="float32")
pred = np.zeros((z.shape[0], 3), dtype="float32")
delta_oof = np.zeros((z.shape[0], 3), dtype="float32")



## === cell 11
cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")

    tf.keras.backend.clear_session()
    net = make_model()

    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    delta_oof += net.predict(z, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    print()



## === cell 12
sigma_opt = mean_absolute_error(y.reshape(-1), pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt(MAE):", sigma_opt, "sigma_mean(unc):", sigma_mean)

o_clipped = np.maximum(delta_oof[:, 2] - delta_oof[:, 0], 70)
delta_abs = np.minimum(np.abs(delta_oof[:, 1] - y.reshape(-1)), 1000)
sqrt2 = np.sqrt(2.0)
logL_score = np.mean((-(sqrt2 * delta_abs) / o_clipped) - np.log(sqrt2 * o_clipped))
print("OOF mean Laplace log-likelihood (approx):", logL_score)



## === cell 13
sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]

eps = 1e-6
scale = float(sigma_opt / (sigma_mean + eps)) if np.isfinite(sigma_mean) else 1.0
scale = float(np.clip(scale, 0.5, 2.0))  # keep adjustment small/stable
sub["Confidence1"] = sub["Confidence1"] * scale

subm = sub[["Patient_Week", "FVC1", "Confidence1"]].copy()
subm.rename(columns={"FVC1": "FVC", "Confidence1": "Confidence"}, inplace=True)

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").astype(
    "float32"
)
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)

subm["FVC"] = (
    pd.to_numeric(subm["FVC"], errors="coerce")
    .fillna(sub["FVC"].median())
    .astype("float32")
)

sample = pd.read_csv(f"{ROOT}/sample_submission.csv")[["Patient_Week"]]
subm = sample.merge(subm, on="Patient_Week", how="left")
subm["FVC"] = subm["FVC"].fillna(subm["FVC"].median()).astype("float32")
subm["Confidence"] = subm["Confidence"].fillna(70.0).astype("float32")

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())

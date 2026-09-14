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

-6.9287

# 6. Current score

-8.52749

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.90271) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and always writes a valid `submission.csv`: (1) avoid the `protobuf`/TF import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, (2) import `random` for seeding, and (3) replace the removed `DataFrame.append` with `pd.concat`. I also fix a few logic/type issues that prevent training (missing `FE` due to earlier failure, `y` shape mismatch with the custom loss expecting 2D, and deprecated Adam args `lr/decay` in TF 2.18). Finally, I keep the model/training loop intact, but ensure the submission has the required columns and that baseline rows from `test.csv` are set with reasonable confidence (>=70) to avoid metric penalties.'
- What this solution (achieved -7.86488) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TF 2.18 and protobuf 6.x by forcing TF to use the pure-Python protobuf runtime and the legacy Keras implementation before importing TensorFlow. I keep the model/training loop and loss exactly the same, but add a small compatibility guard so the script still runs even if TensorFlow can’t be imported (it fall back to a safe baseline submission rather than erroring). This ensures a valid `submission.csv` is always written end-to-end in the Kaggle environment. If TensorFlow loads successfully, the original training-based predictions are used (which should keep/improve score vs the current run that crashes).'
- What this solution (achieved -8.04389) has done: 'I remove the TensorFlow import “fallback” path and instead make TensorFlow import reliably by downgrading `protobuf` at runtime to a TF-compatible version (this fixes the `MessageFactory.GetPrototype` crash). Then the original training/prediction pipeline run end-to-end again (same model, same loss, same CV loop), which should improve your score toward the target because the current run is effectively forced into the weaker baseline path. I also make the TF import strict (fail fast with a clear error) so you don’t silently submit the baseline again. Finally, I keep submission formatting identical and ensure `submission.csv` is always written with the required columns and dtypes.'
- What this solution (achieved -7.82368) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime (instead of trying to pip-downgrade protobuf mid-run, which doesn’t reliably affect already-imported C-extension symbols and still triggers `MessageFactory.GetPrototype`). This is the minimal change needed to make TF import succeed so your existing training/inference pipeline runs end-to-end again (which should move score upward toward the target by restoring the learned model instead of crashing). I also make the protobuf version check correct (the current code tries to import `google.protobuf.__version__` as a module) and keep everything else (features, model, loss, CV loop, submission formatting) unchanged.'
- What this solution (achieved -8.17941) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to a TF-compatible 5.x version at runtime before importing TensorFlow (the pure-Python protobuf env vars are not sufficient under protobuf 6.x in this environment). I keep your model, loss, folds, epochs, and feature engineering unchanged, only adding a small guarded install + kernel restart-style import cleanup so execution continues reliably. I also add a tiny safety check to ensure the submission always matches `sample_submission.csv` order/rows and has valid numeric dtypes, without changing prediction logic. This should both unblock end-to-end execution and restore the learned-model submission (improving score toward the target versus the current crash).'
- What this solution (achieved -8.08046) has done: 'I keep your model/training loop intact and focus on small, metric-aligned post-processing that should improve the Laplace log-likelihood toward your target. Specifically, I (1) compute an out-of-fold optimal global confidence (`sigma_opt`) using the competition metric (not MAE), and (2) use a single global confidence (clipped at 70) for all predictions, which is usually safer than per-row predicted uncertainty from this network. I also replace the hardcoded `0.996` shrink with a data-driven shrinkage factor learned from out-of-fold predictions (simple linear scaling), which is a minimal calibration step that often improves FVC accuracy without changing training. Submission format/ordering stays identical and a valid `submission.csv` is always produced.'
- What this solution (achieved -8.19837) has done: 'I keep your training loop, model, and loss unchanged, and only adjust the metric-aligned post-processing that determines the final `FVC`/`Confidence` written to `submission.csv`. Specifically, instead of using only a global multiplicative scale on OOF predictions, I fit a minimal linear calibration `FVC_cal = a*pred + b` (with tight clipping) which often reduces systematic bias across weeks/patients and should lift the Laplace metric toward your target. Then, I re-optimize a single global `sigma_opt` using the calibrated OOF predictions (same search idea you already use), and apply that constant confidence to all rows (still clipped at 70). Submission row order/format stays exactly aligned to `sample_submission.csv`, and baseline `test.csv` rows are still forced to the known FVC with confidence 70.'
- What this solution (achieved -8.52749) has done: 'I keep your model/training loop unchanged and only adjust the metric-aligned post-processing to reduce the current gap to the target (improve score). The biggest low-risk win here is to stop using a single global confidence for every row and instead use the model’s own predicted spread (`pred[:,2]-pred[:,0]`) as per-row sigma, but calibrated by a single multiplicative factor chosen on out-of-fold predictions to directly maximize the Laplace metric. I keep your linear FVC calibration (`a,b`) exactly as-is, and I also keep the “known baseline week” override for `test.csv` rows at confidence 70. Finally, I ensure confidence is always clipped to >=70 and that submission row order matches `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import subprocess
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_protobuf_tf_compat():
    """
    Keep TF import stable in this environment.
    This is execution-critical (not a modeling change).
    """

    def _pb_version():
        try:
            from google.protobuf import __version__ as pbv

            return str(pbv)
        except Exception:
            return None

    v = _pb_version()
    print("protobuf (pre) version:", v)

    need_install = False
    if v is None:
        need_install = True
    else:
        try:
            major = int(v.split(".")[0])
            if major >= 6:
                need_install = True
        except Exception:
            need_install = True

    if need_install:
        print("Installing TF-compatible protobuf (<6)...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf>=5.28.0,<6",
            ]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]

    print("protobuf (post) version:", _pb_version())


_ensure_protobuf_tf_compat()

import tensorflow as tf  # must succeed after protobuf pin

print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("TF version:", tf.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)




## === cell 2
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
print(train.head())
print(test.head())
print(sub.head())




## === cell 3
train.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")

print(train.shape, test.shape, sub.shape)




## === cell 4
print(train.info())




## === cell 5
import pydicom

image_root = "../input/osic-pulmonary-fibrosis-progression/test"
first_patient = sorted(
    [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
)[0]
first_dcm = sorted(
    [
        f
        for f in os.listdir(os.path.join(image_root, first_patient))
        if f.lower().endswith(".dcm")
    ]
)[0]
dcm_path = os.path.join(image_root, first_patient, first_dcm)

image = pydicom.dcmread(dcm_path)

plt.figure(figsize=(4, 4))
plt.imshow(image.pixel_array, cmap=plt.cm.inferno)
plt.axis("off")
plt.title(f"{first_patient}/{first_dcm}")
plt.show()




## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], ignore_index=True, sort=False)

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

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print("Features:", FE)
print(train.shape, test.shape, sub.shape)




## === cell 7
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
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = tf.keras.layers.Input((nh,), name="Patient")
    x = tf.keras.layers.Dense(100, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(100, activation="relu", name="d2")(x)
    p1 = tf.keras.layers.Dense(3, activation="linear", name="p1")(x)
    p2 = tf.keras.layers.Dense(3, activation="relu", name="p2")(x)
    preds = tf.keras.layers.Lambda(
        lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds"
    )([p1, p2])

    model = tf.keras.models.Model(z, preds, name="definitely_not_a_CNN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 8
y = train["FVC"].values.astype("float32").reshape(-1, 1)
z = train[FE].values.astype("float32")
ze = sub[FE].values.astype("float32")
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype="float32")
pred = np.zeros((z.shape[0], 3), dtype="float32")

net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())




## === cell 9
from sklearn.model_selection import KFold

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)




## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 128

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)
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
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD




## === cell 11
def _laplace_metric_np(y_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - fvc_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


y_true = y.ravel().astype(np.float64)

oof_pred = pred[:, 1].astype(np.float64)
x = oof_pred
x_mean = float(np.mean(x))
y_mean = float(np.mean(y_true))
var_x = float(np.mean((x - x_mean) ** 2) + 1e-12)
cov_xy = float(np.mean((x - x_mean) * (y_true - y_mean)))
a = float(cov_xy / var_x)
b = float(y_mean - a * x_mean)
a = float(np.clip(a, 0.95, 1.05))
b = float(np.clip(b, -250.0, 250.0))
print("Learned linear calibration: a =", a, "b =", b)

oof_pred_cal = a * oof_pred + b

oof_sigma_raw = (pred[:, 2] - pred[:, 0]).astype(np.float64)
oof_sigma_raw = np.maximum(
    oof_sigma_raw, 1.0
)  # numeric safety; final clip is still at 70 in metric

c_grid = np.geomspace(0.5, 3.0, num=80)  # multiplicative calibration for sigma
scores = []
for c in c_grid:
    scores.append(_laplace_metric_np(y_true, oof_pred_cal, c * oof_sigma_raw).mean())
scores = np.asarray(scores)
best_idx = int(np.argmax(scores))
c_opt = float(c_grid[best_idx])
print("Chosen sigma scale c_opt:", c_opt, "OOF metric:", float(scores[best_idx]))

test_pred_fvc_cal = (a * pe[:, 1].astype(np.float64) + b).astype(np.float64)
test_sigma_raw = pe[:, 2].astype(np.float64) - pe[:, 0].astype(np.float64)
test_sigma = (c_opt * np.maximum(test_sigma_raw, 1.0)).astype(np.float64)

sub["FVC1"] = test_pred_fvc_cal.astype("float32")
sub["SIGMA1"] = test_sigma.astype("float32")

subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "SIGMA1"]].copy()
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

subm["Confidence"] = subm["SIGMA1"].astype(np.float64)
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
submission = sample.merge(
    subm[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

submission["FVC"] = submission["FVC"].fillna(float(subm["FVC"].median()))
submission["Confidence"] = submission["Confidence"].fillna(70.0).clip(lower=70.0)

submission["FVC"] = submission["FVC"].astype("float32")
submission["Confidence"] = submission["Confidence"].astype("float32")
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Confidence min/max:",
    submission["Confidence"].min(),
    submission["Confidence"].max(),
)

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

-6.9529

# 6. Current score

-7.71263

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I first fix the environment-breaking import error caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle workaround). Next, I fix the pandas 2.x breakage by replacing the removed `DataFrame.append` with `pd.concat`, which also restore the `FE` feature list so downstream cells can run. Finally, I make the TensorFlow optimizer arguments compatible with TF 2.18 (`lr`→`learning_rate`, remove deprecated `decay`) and ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.28558) has done: 'I fix two execution blockers: (1) the protobuf/TensorFlow import crash by forcing a safe protobuf implementation version check before importing TensorFlow, and (2) the training-time shape error by making `y_true` consistently 2D (so your custom `score()` and `qloss()` can index `y_true[:, 0]`). These changes preserve your model, features, and training loop semantics, but allow the pipeline to run end-to-end. To move the score upward toward the target, I also make the evaluation metric/loss consistent by keeping `score()` as the negative log-likelihood (higher-is-better) while still minimizing it (as Keras minimizes loss), without changing architecture or training strategy. Finally, I ensure `submission.csv` is always written with the required columns and aligned `Patient_Week`s.'
- What this solution (achieved -8.3686) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x API mismatch by switching to the pure-Python protobuf implementation **and** pinning protobuf to the compatible major version *at runtime* before importing TensorFlow. Then I keep your exact model/loss/training loop, but make two minimal score-improving calibration fixes: (1) remove the hardcoded `0.996` shrinkage on predictions (it usually hurts), and (2) ensure the submission `Confidence` is always positive and clipped to Kaggle’s minimum-meaningful scale (≥70) to better match the metric. Finally, I keep the “copy baseline FVC for the provided test week” behavior, but set its confidence to 70 (not 0.1) so it doesn’t get heavily penalized.'
- What this solution (achieved -8.72906) has done: 'Your current score (-8.3686) is below the target (-6.9529), so we should improve it cautiously with minimal changes. The biggest low-risk gain here is aligning training with the evaluation semantics: your `score()` currently returns NLL (lower is better) but is treated as a “metric”; we keep it as a metric but ensure the loss is dominated by NLL (since Kaggle optimizes NLL directly) by lowering the quantile-loss weight slightly (core architecture/training loop unchanged). Next, we calibrate test-time `Confidence` using a constant derived from out-of-fold residuals (a robust sigma), because your current branch often emits raw network uncertainty that tends to be miscalibrated for this competition’s clipping at 70. Finally, we keep the baseline-week overwrite, but also apply the same calibrated confidence elsewhere to reduce penalty from under/over-confident sigma.'
- What this solution (achieved -7.95112) has done: 'I make two minimal, score-relevant calibrations without changing your model, features, or training loop: (1) compute an empirically better constant confidence by minimizing the competition’s NLL on out-of-fold predictions (instead of using median-absolute-error), and (2) calibrate the FVC level by adding a tiny constant offset equal to the out-of-fold mean residual (this corrects a common systematic bias and typically improves NLL). Both changes preserve evaluation semantics and only affect post-processing, keeping your fold training identical. I also ensure the calibrated confidence is clipped to the competition’s effective minimum (≥70) and that the baseline-week overwrite remains unchanged.'
- What this solution (achieved -7.89757) has done: 'We’re currently below the target (gap ≈ -0.998), so we should improve the score modestly without touching your model/training. The lowest-risk lever is confidence calibration: a single global `sigma` fit on OOF residuals is good, but you can usually get closer to the Laplace NLL optimum by using a *simple piecewise sigma*: 70 for the baseline week (already done) and a slightly larger constant for non-baseline weeks to avoid over-penalizing larger deltas. I keep your existing grid-search calibration, but expand the search range upward and fit `best_sigma_nonbase` on OOF deltas after excluding week==0-like rows (baseline rows) so the global sigma isn’t pulled down. Finally, I apply the non-baseline sigma only to non-baseline rows in the submission (baseline week keeps 70), which tends to improve NLL toward your target without changing core logic.'
- What this solution (achieved -7.71263) has done: 'We’re currently below the target (need a higher score), so I keep your model/training exactly as-is and only adjust post-processing in a way that better matches the competition’s Laplace NLL. The minimal lever is confidence calibration: instead of a single constant sigma for all non-baseline rows, we fit a slightly more informative (but still very simple) confidence that scales with the distance-from-baseline week, which often reduces NLL by avoiding under-confidence at far weeks and over-confidence at near weeks. I fit this scaling using your existing OOF predictions (no leakage) and then apply it only to non-baseline test rows while keeping the provided test baseline row confidence fixed at 70. This preserves submission semantics and should move the score upward toward the target without changing core logic.'
- What this solution (achieved -7.71263) has done: 'Your current score (-7.71263) is worse than the target (-6.9529), so we should increase it with the smallest, safest change that doesn’t alter your model/training core. The easiest low-risk gain in this competition is better confidence calibration: instead of a single linear function of week, we keep your fitted week-scaling but also add a per-patient multiplicative calibration (still learned only from OOF residuals) to handle patients with systematically higher/lower uncertainty. We also clip the learned scaling to a narrow, conservative range to avoid destabilizing the score. Model architecture, features, folds, epochs, and loss remain unchanged; only the post-processing confidence mapping changes.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf"):
                    del sys.modules[k]
    except Exception as e:
        print("Warning: protobuf compatibility step failed:", repr(e))


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF:", tf.__version__)
print("Pandas:", pd.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(2)



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
image_path = "../input/osic-pulmonary-fibrosis-progression/"
image_files_list = []
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if ".dcm" in filename.lower():
            image_files_list.append(os.path.join(dirName, filename))

if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()
else:
    print("No DICOM files found under:", image_path)



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], axis=0, ignore_index=True)

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

COLS = ["Sex", "SmokingStatus"]  # ,'Age'
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
print("Features:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    """
    Returns NLL (lower is better). Kaggle uses the negative of this (higher is better).
    Keeping this unchanged preserves evaluation semantics.
    """
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
    model.compile(loss=mloss(0.6), optimizer=opt, metrics=[score])
    return model




## === cell 8
y = train["FVC"].values.astype(np.float32).reshape(-1, 1)
z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 64

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
    train_loss, train_score = net.evaluate(
        z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE
    )
    print(f"Train Loss: {train_loss}  Score(NLL): {train_score}")
    val_loss, val_score = net.evaluate(
        z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE
    )
    print(f"Val Loss: {val_loss}  Score(NLL): {val_score}")
    print(f"Score diff: {val_score - train_score}")

    print("Predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("Predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## === cell 11
y_true = y.ravel().astype(np.float64)
y_pred_oof = pred[:, 1].astype(np.float64)

bias = float(np.mean(y_true - y_pred_oof))
print("OOF mean residual bias (true - pred):", bias)

pe[:, 1] = (pe[:, 1].astype(np.float64) + bias).astype(np.float32)
pred[:, 1] = (pred[:, 1].astype(np.float64) + bias).astype(np.float32)


def _nll_with_sigma(y_t, y_p, sigma):
    sigma_clip = max(float(sigma), 70.0)
    delta = np.minimum(np.abs(y_t - y_p), 1000.0)
    sq2 = np.sqrt(2.0)
    nll = (sq2 * delta) / sigma_clip + np.log(sq2 * sigma_clip)
    return float(np.mean(nll))


def _nll_with_sigma_ab(y_t, y_p, base_week, a, b):
    sigma = a + b * np.abs(base_week)
    sigma = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_t - y_p), 1000.0)
    sq2 = np.sqrt(2.0)
    nll = (sq2 * delta) / sigma + np.log(sq2 * sigma)
    return float(np.mean(nll))


def _nll_with_patient_scale(y_t, y_p, sigma_base, patient_scale, patient_ids):
    sigma = sigma_base * patient_scale
    sigma = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_t - y_p), 1000.0)
    sq2 = np.sqrt(2.0)
    nll = (sq2 * delta) / sigma + np.log(sq2 * sigma)
    return float(np.mean(nll))


abs_err = np.abs(y_true - pred[:, 1].astype(np.float64))
sigma_start = float(np.maximum(70.0, np.median(abs_err) * np.sqrt(2.0)))

base_week_train = train["base_week"].values.astype(np.float64)
mask_nonbase = base_week_train != 0.0
if np.sum(mask_nonbase) < 50:
    mask_nonbase = np.ones_like(base_week_train, dtype=bool)

grid = np.unique(
    np.clip(
        np.concatenate(
            [
                np.linspace(70.0, 700.0, 64),
                np.linspace(max(70.0, sigma_start - 200.0), sigma_start + 400.0, 61),
            ]
        ),
        70.0,
        1500.0,
    )
).astype(np.float64)

best_sigma_nonbase = None
best_nll_nonbase = 1e18
for s in grid:
    nll = _nll_with_sigma(
        y_true[mask_nonbase], pred[:, 1].astype(np.float64)[mask_nonbase], s
    )
    if nll < best_nll_nonbase:
        best_nll_nonbase = nll
        best_sigma_nonbase = float(s)

bw_nb = np.abs(base_week_train[mask_nonbase]).astype(np.float64)
y_nb = y_true[mask_nonbase]
p_nb = pred[:, 1].astype(np.float64)[mask_nonbase]

a_grid = np.unique(
    np.clip(
        np.concatenate(
            [
                np.linspace(70.0, 400.0, 34),
                np.array(
                    [
                        best_sigma_nonbase,
                        max(70.0, best_sigma_nonbase - 40.0),
                        best_sigma_nonbase + 40.0,
                    ]
                ),
            ]
        ),
        70.0,
        1500.0,
    )
)
b_grid = np.linspace(0.0, 12.0, 25)  # ml per week; modest range, avoids wild sigmas

best_a, best_b, best_nll_ab = None, None, 1e18
for a in a_grid:
    for b in b_grid:
        nll = _nll_with_sigma_ab(y_nb, p_nb, bw_nb, float(a), float(b))
        if nll < best_nll_ab:
            best_nll_ab = nll
            best_a, best_b = float(a), float(b)

sigma_opt_mae = float(mean_absolute_error(y_true, pred[:, 1].astype(np.float64)))
unc = (pred[:, 2] - pred[:, 0]).astype(np.float64)
sigma_mean_model = float(np.mean(unc))

print(
    "sigma_opt(MAE):",
    sigma_opt_mae,
    "sigma_mean(model):",
    sigma_mean_model,
    "sigma_start(median-based):",
    sigma_start,
    "sigma_cal_nonbase_const(NLL-grid):",
    best_sigma_nonbase,
    "OOF_NLL_nonbase_const(best):",
    best_nll_nonbase,
    "sigma_cal_nonbase_weekscale(a,b):",
    (best_a, best_b),
    "OOF_NLL_nonbase_weekscale(best):",
    best_nll_ab,
)

train_pat = train["Patient"].values
sigma_base_train = (best_a + best_b * np.abs(base_week_train)).astype(np.float64)
sigma_base_train = np.maximum(sigma_base_train, 70.0)

resid_abs = np.abs(y_true - pred[:, 1].astype(np.float64))
df_cal = pd.DataFrame(
    {
        "Patient": train_pat,
        "is_nonbase": mask_nonbase.astype(bool),
        "sigma_base": sigma_base_train,
        "abs_resid": resid_abs,
    }
)
df_nb = df_cal[df_cal["is_nonbase"]].copy()

global_ratio = float(
    np.median(df_nb["abs_resid"].values) / np.median(df_nb["sigma_base"].values)
)
if not np.isfinite(global_ratio) or global_ratio <= 0:
    global_ratio = 1.0

patient_ratio = (
    df_nb.groupby("Patient")
    .apply(
        lambda g: float(
            np.median(g["abs_resid"].values) / np.median(g["sigma_base"].values)
        )
    )
    .to_dict()
)

patient_scale = {}
for p, r in patient_ratio.items():
    if (not np.isfinite(r)) or r <= 0:
        s = 1.0
    else:
        s = float(r / global_ratio)
    s = float(np.clip(s, 0.85, 1.25))
    patient_scale[p] = s

scales_vec = np.array([patient_scale.get(p, 1.0) for p in train_pat], dtype=np.float64)
nll_ab_only = _nll_with_sigma_ab(
    y_true, pred[:, 1].astype(np.float64), np.abs(base_week_train), best_a, best_b
)
nll_ab_patient = _nll_with_patient_scale(
    y_true,
    pred[:, 1].astype(np.float64),
    sigma_base_train,
    scales_vec,
    train_pat,
)
print(
    "OOF_NLL(all) weekscale only:",
    nll_ab_only,
    "weekscale*patient_scale:",
    nll_ab_patient,
)

sub["FVC1"] = pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[
    [
        "Patient_Week",
        "Weeks",
        "FVC",
        "Confidence",
        "FVC1",
        "Confidence1",
        "base_week",
        "Patient",
    ]
].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

sigma_week = (
    best_a + best_b * np.abs(subm["base_week"].values.astype(np.float64))
).astype(np.float64)
sigma_week = np.maximum(sigma_week, 70.0)

sub_scales = subm["Patient"].map(patient_scale).fillna(1.0).values.astype(np.float64)
sigma_week = sigma_week * sub_scales
sigma_week = np.maximum(sigma_week, 70.0)

subm.loc[~subm.FVC1.isnull(), "Confidence"] = sigma_week.astype(np.float32)

subm["Confidence"] = subm["Confidence"].astype(np.float32)
subm["Confidence"] = np.maximum(np.abs(subm["Confidence"].values), 70.0)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

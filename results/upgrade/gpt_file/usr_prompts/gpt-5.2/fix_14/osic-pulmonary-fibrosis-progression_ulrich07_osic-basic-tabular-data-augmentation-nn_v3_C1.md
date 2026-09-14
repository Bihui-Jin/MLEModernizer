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

-10.81253

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.50624) has done: 'I fix the TensorFlow import crash by pinning protobuf to a TensorFlow-compatible version at runtime before importing TF, which resolves the `MessageFactory.GetPrototype` error in Kaggle environments. Then I fix the pandas 2.x incompatibility by replacing the removed `DataFrame.append` usage with `pd.concat`, and correct the augmented-train construction so `tr` keeps the expected columns (Patient/Sex/SmokingStatus/FVC etc.), which removes the downstream `KeyError`s. Finally, I make the model training/inference run end-to-end by ensuring the target `y` has the right shape for the custom loss/metric and by using a TF2-compatible Adam argument (`learning_rate`), and I write a valid `submission.csv` with the required columns.'
- What this solution (achieved -10.74367) has done: 'Your current score is worse than the target (−10.50624 vs −9.1763; higher is better), so the smallest safe improvement is to fix prediction post-processing to better match the competition’s metric without changing the model or training loop. Concretely, we should (1) enforce positive, clipped confidence at inference time (≥70) instead of allowing small/negative values, and (2) avoid setting test-row confidence to 0.1 (which is heavily penalized by the metric) and instead set it to the clipped minimum (70) while keeping the baseline FVC overwrite. These are minimal, metric-aligned changes that typically improve Laplace log-likelihood while preserving the core model/training logic and producing a valid submission.csv.'
- What this solution (achieved -10.78676) has done: 'We make two metric-aligned, minimal post-processing adjustments to push the score upward toward the target without changing the model or training loop. First, we calibrate the predicted uncertainty using your OOF residual scale: multiply each test-row uncertainty by `(sigma_opt / sigma_mean)` so Confidence better matches typical absolute error (then still clip at 70). Second, we apply the same calibration to the OOF “oof metric” computation for a more faithful sanity check. These changes keep architecture/training identical and only adjust the Confidence values in a way that typically improves the Laplace log-likelihood when uncertainty is miscalibrated.'
- What this solution (achieved -10.79051) has done: 'We keep your model/training exactly the same and only adjust the **Confidence post-processing** to better match the competition’s Laplace metric, because your current score is below target and improving uncertainty calibration is the smallest safe lever. Specifically, instead of using a global scale based on the **mean** predicted uncertainty (which is sensitive to outliers and can miscalibrate many rows), we calibrate using the **median** and also apply a small, fixed shrink toward the metric’s optimal MAE-based sigma to reduce over/under-confidence. We also ensure Confidence is always finite and clipped to the valid minimum (70), which stabilizes the submission and typically improves the log-likelihood without changing predicted FVCs. Everything else (data prep, features, architecture, loss, folds, epochs) is unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -10.74303) has done: 'Your score is below target (−10.79051 vs −9.1763; higher is better), so the smallest safe lever is Confidence calibration/post-processing while keeping the model/training untouched. We replace the fixed SHRINK=0.85 with a data-driven calibration: choose the shrink factor that maximizes the out-of-fold Laplace metric (via a tiny grid search) using your existing OOF predictions, then reuse that same factor for test Confidence. We also add a minimal per-patient sigma scaling on the test set (based on each patient’s predicted uncertainty vs global median) to reduce systematic under/over-confidence across patients without changing FVC predictions. Everything else (features, folds, epochs, architecture, loss) remains identical, and it still writes a valid submission.csv.'
- What this solution (achieved -10.68071) has done: 'We keep your model, folds, epochs, and feature pipeline unchanged, and only adjust **Confidence post-processing** because your score is below target and this is the smallest lever aligned with the Laplace metric. Specifically, instead of a single global scale + patient factor, we choose an **OOF-optimal per-patient shrink blend** between global and patient-level calibration (which often reduces systematic over/under-confidence and improves the metric without changing FVC). We also select a **single best patient-factor clip range** using OOF (tiny grid) so the per-patient adjustment doesn’t overcorrect and hurt log-likelihood. Finally, we apply the chosen calibration to test and keep the baseline test-row overwrite (FVC + Confidence=70) exactly as you do.'
- What this solution (achieved -10.78676) has done: 'We keep your model/training exactly as-is and only make the smallest metric-aligned adjustments to Confidence calibration, since your score (-10.68071) is still below the target (-9.1763) and the Laplace metric is very sensitive to miscalibrated sigma. Specifically, we (1) choose the global uncertainty scale directly by maximizing the OOF Laplace metric (1D grid on a multiplier around your current scale), and (2) apply a small, OOF-validated additive “sigma floor” above 70 (i.e., Confidence = max(70 + floor, scaled_sigma)) to reduce overconfidence without changing FVC predictions. This preserves all core logic and only changes post-processing in a controlled, OOF-justified way. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -10.87012) has done: 'I keep your model, folds, epochs, and features exactly the same and only adjust the **Confidence calibration** because your score is below target and the Laplace metric is highly sensitive to sigma miscalibration. Right now you calibrate sigma using OOF residuals only; a small, safe improvement is to calibrate sigma against the *metric-optimal* global sigma (not MAE) by directly maximizing the OOF Laplace metric over a slightly wider but still conservative grid. I also add one more minimal degree of freedom: a single global multiplicative factor for the *per-patient* adjustment (again chosen by OOF metric), which often fixes systematic over/under-scaling from patient blending without changing FVC predictions. These are tiny post-processing changes, fully metric-aligned, and should move the score upward toward your target while preserving core logic.'
- What this solution (achieved -10.79973) has done: 'We keep your model, folds, epochs, and features identical and only adjust the final post-processing to better match the competition metric. Right now Confidence can become too small/large relative to the typical residual scale, and the Laplace metric strongly rewards a well-calibrated global sigma; we therefore choose a single *global additive floor* and *global multiplicative scale* by directly maximizing the OOF Laplace metric over a slightly broader (but still safe) grid. Then we apply that same calibrated mapping to the test uncertainties, while preserving your baseline-row overwrite (FVC and Confidence=70) exactly. This is a minimal, metric-aligned change that should move the score upward toward the target without touching core training logic.'
- What this solution (achieved -10.84707) has done: 'Your current score (-10.79973) is below the target (-9.1763), so we should cautiously improve it with the smallest metric-aligned change. The biggest remaining low-risk lever is Confidence calibration: rather than using a coarse grid for `(scale, floor, pat_mult)`, we fit a *single global Laplace-optimal sigma* on OOF residuals (closed-form via median(|residual|)/ln2, then clipped at 70) and then only allow a very small “blend” between that robust global sigma and your current per-row calibrated sigma to avoid overconfidence. This keeps the model/training exactly the same and only adjusts Confidence post-processing in a controlled way, while preserving your baseline-row overwrite (Confidence=70). The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -10.87649) has done: 'We keep your model, folds, epochs, features, and loss untouched and only adjust Confidence post-processing, since your score is below target and the Laplace metric is extremely sensitive to sigma calibration. The main minimal fix is to choose the final blend weight `best_w` on OOF using a finer grid (still very small compute) because your current coarse grid can easily miss a better calibration point and hurt the public score. We also add a safety clamp to keep Confidence finite and within a reasonable upper bound (still metric-valid) to avoid rare extreme sigmas that can drag the mean log-likelihood down. Submission schema/paths remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved -10.78676) has done: 'Your current score (−10.87649) is below the target (−9.1763), so we should improve it with the smallest, metric-aligned change while keeping the model/training untouched. The Laplace metric heavily penalizes miscalibrated uncertainty, and your current pipeline only calibrates Confidence using OOF predictions from the *augmented rows* (many per patient/week), which can distort residual/uncertainty calibration relative to the evaluation unit (Patient_Week). I keep your model exactly the same and only change the Confidence calibration step to compute the OOF calibration statistics on a de-duplicated, evaluation-aligned set (one row per Patient_Week), then apply the same mapping to test. This is a minimal post-processing fix that typically improves the public score by making sigma calibration match how Kaggle scores.'
- What this solution (achieved -10.81253) has done: 'Your current score is below the target (−10.78676 vs −9.1763; higher is better), so we should make the smallest metric-aligned change that plausibly improves the Laplace log-likelihood without touching the model/training. Right now the Confidence calibration is driven entirely by the augmented-row OOF predictions; we additionally compute a calibration using **only the baseline rows (wk2==0)**, which matches the test-time condition (only baseline measurement known) and often yields better sigma scaling. We then **blend** the two calibrations with a tiny OOF grid search (single scalar) and apply that blended calibration to test Confidence, while keeping your FVC predictions and baseline-row overwrite (Confidence=70) unchanged. This preserves core logic and only adjusts post-processing in a controlled, OOF-justified way.'

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
unc_oof_all = (pred[:, 2] - pred[:, 0]).astype(np.float32)
unc_oof_all = np.nan_to_num(unc_oof_all, nan=70.0, posinf=1e6, neginf=70.0)

key_df = tr[["Patient", "Weeks"]].copy()
key_df["idx"] = np.arange(len(tr), dtype=np.int32)
rep_idx = key_df.groupby(["Patient", "Weeks"])["idx"].median().astype(int).values

y_e = y[rep_idx, 0]
pred_e = pred[rep_idx, 1]
unc_e = unc_oof_all[rep_idx]

med_abs = float(np.median(np.clip(np.abs(y_e - pred_e), 0.0, 1000.0)))
sigma_lap_opt = (med_abs / max(np.log(2.0), 1e-12)) * np.sqrt(2.0)
sigma_lap_opt = float(max(sigma_lap_opt, 70.0))

sigma_mae = float(mean_absolute_error(y_e, pred_e))
sigma_med = float(np.median(np.clip(unc_e, 0.0, 1e6)))
base_scale = sigma_mae / max(sigma_med, 1e-6)

tmp_oof = tr.loc[rep_idx, ["Patient"]].copy()
tmp_oof["unc"] = unc_e
pat_med_oof = tmp_oof.groupby("Patient")["unc"].median()
global_med_oof = float(np.median(np.clip(unc_e, 0.0, 1e6)))
global_med_oof = max(global_med_oof, 1e-6)
pat_factor_raw_oof = global_med_oof / pat_med_oof

blend_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
clip_grid = [(0.75, 1.35), (0.8, 1.25), (0.85, 1.2), (0.9, 1.15)]

best_cfg = None
best_oof_m = -1e18
for alpha in blend_grid:
    for lo, hi in clip_grid:
        pf_clip = pat_factor_raw_oof.clip(lower=lo, upper=hi)
        pf_blend = (1.0 - float(alpha)) * 1.0 + float(alpha) * pf_clip
        row_pf = tr.loc[rep_idx, "Patient"].map(pf_blend).astype(np.float32).values
        unc_try = np.maximum(unc_e * base_scale * row_pf, 70.0)
        m_try = float(metric(y_e, pred_e, unc_try))
        if m_try > best_oof_m:
            best_oof_m = m_try
            best_cfg = (float(alpha), float(lo), float(hi))

alpha_best, lo_best, hi_best = best_cfg

row_pf_best = (
    tr.loc[rep_idx, "Patient"]
    .map(
        (
            (1.0 - alpha_best) * 1.0
            + alpha_best * pat_factor_raw_oof.clip(lower=lo_best, upper=hi_best)
        )
    )
    .astype(np.float32)
    .values
)

floor_grid = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 55.0, 70.0, 90.0], dtype=np.float32)
scale_grid = base_scale * np.array(
    [0.25, 0.35, 0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0], dtype=np.float32
)
pf_mult_grid = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35], dtype=np.float32)

best_tuple = None
best_tuple_m = -1e18
for fl in floor_grid:
    for sc_try in scale_grid:
        for pm in pf_mult_grid:
            unc_try = np.maximum(
                unc_e * float(sc_try) * (row_pf_best * float(pm)), 70.0 + float(fl)
            )
            m_try = float(metric(y_e, pred_e, unc_try))
            if m_try > best_tuple_m:
                best_tuple_m = m_try
                best_tuple = (float(sc_try), float(fl), float(pm))

scale, best_floor, pat_mult = best_tuple

unc_calibrated_e = np.maximum(
    unc_e * scale * (row_pf_best * pat_mult), 70.0 + best_floor
)

w_grid = np.array(
    [0.0, 0.05, 0.1, 0.15, 0.2, 0.275, 0.35, 0.425, 0.5, 0.6, 0.7], dtype=np.float32
)
best_w = 0.0
best_m_w = -1e18
for w in w_grid:
    unc_blend = (1.0 - float(w)) * unc_calibrated_e + float(w) * sigma_lap_opt
    unc_blend = np.maximum(unc_blend, 70.0)
    m_w = float(metric(y_e, pred_e, unc_blend))
    if m_w > best_m_w:
        best_m_w = m_w
        best_w = float(w)

wk2_rep = tr.loc[rep_idx, "wk2"].values.astype(np.float32)
base_mask = wk2_rep == 0.0
rep_idx_base = rep_idx[base_mask]

y_b = y[rep_idx_base, 0]
pred_b = pred[rep_idx_base, 1]
unc_b = unc_oof_all[rep_idx_base]

sigma_mae_b = float(mean_absolute_error(y_b, pred_b))
sigma_med_b = float(np.median(np.clip(unc_b, 0.0, 1e6)))
base_scale_b = sigma_mae_b / max(sigma_med_b, 1e-6)

tmp_b = tr.loc[rep_idx_base, ["Patient"]].copy()
tmp_b["unc"] = unc_b
pat_med_b = tmp_b.groupby("Patient")["unc"].median()
global_med_b = float(np.median(np.clip(unc_b, 0.0, 1e6)))
global_med_b = max(global_med_b, 1e-6)
pat_factor_raw_b = global_med_b / pat_med_b

best_cfg_b = None
best_m_b = -1e18
for alpha in blend_grid:
    for lo, hi in clip_grid:
        pf_clip = pat_factor_raw_b.clip(lower=lo, upper=hi)
        pf_blend = (1.0 - float(alpha)) * 1.0 + float(alpha) * pf_clip
        row_pf = tr.loc[rep_idx_base, "Patient"].map(pf_blend).astype(np.float32).values
        unc_try = np.maximum(unc_b * base_scale_b * row_pf, 70.0)
        m_try = float(metric(y_b, pred_b, unc_try))
        if m_try > best_m_b:
            best_m_b = m_try
            best_cfg_b = (float(alpha), float(lo), float(hi))

alpha_b, lo_b, hi_b = best_cfg_b
row_pf_b = (
    tr.loc[rep_idx_base, "Patient"]
    .map(
        (
            (1.0 - alpha_b) * 1.0
            + alpha_b * pat_factor_raw_b.clip(lower=lo_b, upper=hi_b)
        )
    )
    .astype(np.float32)
    .values
)

best_tuple_b = None
best_tuple_m_b = -1e18
scale_grid_b = base_scale_b * np.array(
    [0.35, 0.6, 0.9, 1.0, 1.15, 1.5], dtype=np.float32
)
floor_grid_b = np.array([0.0, 20.0, 40.0, 70.0], dtype=np.float32)
pf_mult_grid_b = np.array([0.9, 1.0, 1.1], dtype=np.float32)

for fl in floor_grid_b:
    for sc_try in scale_grid_b:
        for pm in pf_mult_grid_b:
            unc_try = np.maximum(
                unc_b * float(sc_try) * (row_pf_b * float(pm)), 70.0 + float(fl)
            )
            m_try = float(metric(y_b, pred_b, unc_try))
            if m_try > best_tuple_m_b:
                best_tuple_m_b = m_try
                best_tuple_b = (float(sc_try), float(fl), float(pm))

scale_b, floor_b, pat_mult_b = best_tuple_b

unc_calibrated_b = np.maximum(unc_b * scale_b * (row_pf_b * pat_mult_b), 70.0 + floor_b)

best_w_b = 0.0
best_m_w_b = -1e18
for w in w_grid:
    unc_blend = (1.0 - float(w)) * unc_calibrated_b + float(w) * sigma_lap_opt
    unc_blend = np.maximum(unc_blend, 70.0)
    m_w = float(metric(y_b, pred_b, unc_blend))
    if m_w > best_m_w_b:
        best_m_w_b = m_w
        best_w_b = float(w)


def _calibrate_unc(
    unc_raw,
    patients,
    alpha,
    lo,
    hi,
    scale_,
    floor_,
    pat_mult_,
    pat_factor_raw_series,
    w_,
    sigma_global,
    sigma_cap=2000.0,
):
    pat_factor = pat_factor_raw_series.clip(lower=lo, upper=hi)
    pat_factor_blend = (1.0 - alpha) * 1.0 + alpha * pat_factor
    row_pf = patients.map(pat_factor_blend).astype(np.float32).values
    conf = unc_raw.astype(np.float32) * float(scale_) * (row_pf * float(pat_mult_))
    conf = np.nan_to_num(conf, nan=70.0, posinf=1e6, neginf=70.0)
    conf = np.maximum(conf, 70.0 + float(floor_))
    conf = np.clip(conf, 70.0, float(sigma_cap))
    conf = np.maximum((1.0 - float(w_)) * conf + float(w_) * float(sigma_global), 70.0)
    conf = np.clip(
        np.nan_to_num(conf, nan=70.0, posinf=sigma_cap, neginf=70.0), 70.0, sigma_cap
    )
    return conf


patients_e = tr.loc[rep_idx, "Patient"]
unc_eval_main = _calibrate_unc(
    unc_e,
    patients_e,
    alpha_best,
    lo_best,
    hi_best,
    scale,
    best_floor,
    pat_mult,
    pat_factor_raw_oof,
    best_w,
    sigma_lap_opt,
)

pat_factor_raw_b_full = patients_e.map(pat_factor_raw_b).astype(np.float32)
pat_factor_raw_b_full = pat_factor_raw_b_full.fillna(1.0)
pat_factor_raw_b_series = pat_factor_raw_b.copy()
missing_pats = set(patients_e.unique()) - set(pat_factor_raw_b_series.index)
if len(missing_pats) > 0:
    for mp in missing_pats:
        pat_factor_raw_b_series.loc[mp] = 1.0

unc_eval_base = _calibrate_unc(
    unc_e,
    patients_e,
    alpha_b,
    lo_b,
    hi_b,
    scale_b,
    floor_b,
    pat_mult_b,
    pat_factor_raw_b_series,
    best_w_b,
    sigma_lap_opt,
)

mix_grid = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0], dtype=np.float32)
best_mix = 0.0
best_mix_m = -1e18
for mix in mix_grid:
    unc_mix = (1.0 - float(mix)) * unc_eval_main + float(mix) * unc_eval_base
    unc_mix = np.maximum(unc_mix, 70.0)
    m_mix = float(metric(y_e, pred_e, unc_mix))
    if m_mix > best_mix_m:
        best_mix_m = m_mix
        best_mix = float(mix)

print("Eval-aligned calibration sizes:", len(rep_idx), "from augmented rows:", len(tr))
print("sigma_mae:", sigma_mae)
print("sigma_med (median predicted unc):", sigma_med)
print("base_scale (start):", base_scale)
print("sigma_lap_opt (robust OOF residual-based):", sigma_lap_opt)
print("chosen patient-factor blend alpha (eval-aligned):", alpha_best)
print("chosen patient-factor clip (eval-aligned):", (lo_best, hi_best))
print(
    "chosen global scale / floor / pat_mult (eval-aligned):",
    (scale, best_floor, pat_mult),
)
print("chosen blend weight toward sigma_lap_opt (eval-aligned):", best_w)

print("baseline-only calib (wk2==0) size:", len(rep_idx_base))
print("chosen patient-factor blend alpha (baseline-only):", alpha_b)
print("chosen patient-factor clip (baseline-only):", (lo_b, hi_b))
print(
    "chosen global scale / floor / pat_mult (baseline-only):",
    (scale_b, floor_b, pat_mult_b),
)
print("chosen blend weight toward sigma_lap_opt (baseline-only):", best_w_b)

print("chosen mix between eval-aligned and baseline-only Confidence:", best_mix)
print("oof-eval (raw unc)", metric(y_e, pred_e, np.maximum(unc_e, 70.0)))
print("oof-eval (eval-aligned calibrated)", metric(y_e, pred_e, unc_eval_main))
print(
    "oof-eval (baseline-only calibrated applied to eval)",
    metric(y_e, pred_e, unc_eval_base),
)
print(
    "oof-eval (mixed)",
    metric(
        y_e,
        pred_e,
        np.maximum((1.0 - best_mix) * unc_eval_main + best_mix * unc_eval_base, 70.0),
    ),
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
plt.hist(unc_oof_all, bins=50)
plt.title("uncertainty in prediction (OOF)")
plt.show()



## === cell 23
te["FVC"] = pe[:, 1]
test_unc = (pe[:, 2] - pe[:, 0]).astype(np.float32)
test_unc = np.nan_to_num(test_unc, nan=70.0, posinf=1e6, neginf=70.0)

SIGMA_CAP = 2000.0

tmp = te[["Patient"]].copy()
tmp["unc"] = test_unc
pat_med = tmp.groupby("Patient")["unc"].median()

global_med = float(np.median(np.clip(test_unc, 0.0, 1e6)))
global_med = max(global_med, 1e-6)

pat_factor = (global_med / pat_med).clip(lower=lo_best, upper=hi_best)
pat_factor_blend = (1.0 - alpha_best) * 1.0 + alpha_best * pat_factor
te["_pat_factor_main"] = te["Patient"].map(pat_factor_blend).astype(np.float32)

conf_main = (
    test_unc * scale * (te["_pat_factor_main"].values.astype(np.float32) * pat_mult)
)
conf_main = np.nan_to_num(conf_main, nan=70.0, posinf=1e6, neginf=70.0)
conf_main = np.maximum(conf_main, 70.0 + best_floor)
conf_main = np.clip(conf_main, 70.0, SIGMA_CAP)
conf_main = np.maximum((1.0 - best_w) * conf_main + best_w * sigma_lap_opt, 70.0)
conf_main = np.clip(
    np.nan_to_num(conf_main, nan=70.0, posinf=SIGMA_CAP, neginf=70.0), 70.0, SIGMA_CAP
)

pat_factor_b = (global_med / pat_med).clip(lower=lo_b, upper=hi_b)
pat_factor_blend_b = (1.0 - alpha_b) * 1.0 + alpha_b * pat_factor_b
te["_pat_factor_base"] = te["Patient"].map(pat_factor_blend_b).astype(np.float32)

conf_base = (
    test_unc * scale_b * (te["_pat_factor_base"].values.astype(np.float32) * pat_mult_b)
)
conf_base = np.nan_to_num(conf_base, nan=70.0, posinf=1e6, neginf=70.0)
conf_base = np.maximum(conf_base, 70.0 + floor_b)
conf_base = np.clip(conf_base, 70.0, SIGMA_CAP)
conf_base = np.maximum((1.0 - best_w_b) * conf_base + best_w_b * sigma_lap_opt, 70.0)
conf_base = np.clip(
    np.nan_to_num(conf_base, nan=70.0, posinf=SIGMA_CAP, neginf=70.0), 70.0, SIGMA_CAP
)

te["Confidence"] = np.maximum((1.0 - best_mix) * conf_main + best_mix * conf_base, 70.0)
te["Confidence"] = np.clip(
    np.nan_to_num(
        te["Confidence"].values.astype(np.float32),
        nan=70.0,
        posinf=SIGMA_CAP,
        neginf=70.0,
    ),
    70.0,
    SIGMA_CAP,
)

te.drop(columns=["_pat_factor_main", "_pat_factor_base"], inplace=True)



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

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

-6.885968373235216

# 6. Current score

-24.6404

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.64042) has done: 'I remove/guard the problematic optional imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep the rest of the pipeline unchanged. Then I fix the pandas `DataFrame.append` deprecation by using `pd.concat`, which unblocks creation of the combined `data` frame used by later cells. The original notebook also tries to load a non-existent external `model.h5`; since it’s unavailable in this environment, I minimally replace that with a small KFold-trained tabular Keras model that outputs the same 3-quantile predictions the rest of your code expects (so the submission logic remains the same). Finally, I ensure `SmokingStatus` is referenced correctly, replace deprecated `np.float`, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.64029) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime issue. Then I fix the submission post-processing bug where the code tries to inverse-scale `FVC` even though `FVC` was never scaled; instead, I use the model’s predicted median quantile directly as `FVC` in ml. Finally, I keep the same core KFold tabular model and quantile output logic, but ensure confidence is computed robustly (non-negative, clipped to at least 70) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.64046) has done: 'I fix the execution blockers while keeping your existing pipeline and KFold quantile-regression model intact. First, I harden the TensorFlow import against the protobuf `GetPrototype` crash by falling back to a safe environment setup and (if needed) a clean import path. Second, I fix the submission post-processing bug: `FVC` is not part of the scaled feature list, so inverse-scaling via `scaler.data_min_` can’t work; instead, I train/predict in the original ml scale (as your `y_train` already is) and use predicted quantiles directly for `FVC` and `Confidence`. Finally, I make categorical mapping robust to missing values and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved -24.6404) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by removing the forced pure-Python protobuf setting and instead forcing a compatible protobuf implementation for TF in this environment. I keep the rest of your pipeline (data prep, scaling, KFold quantile-regression tabular model, and submission formatting) unchanged so behavior and semantics remain the same. I also add a small, score-neutral safety fallback: if TensorFlow still fails to import, the code produce a valid submission using a simple per-patient linear trend fit from the training data (so you always get a CSV). This unblocks end-to-end execution and should also improve score substantially versus the current crash/no-run state, while keeping core logic intact when TF works.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = True
try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
    import tensorflow.keras.backend as K

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    tf = None
    L = None
    M = None
    K = None

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import failed with:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 5
NUM_IMAGES = 140
BATCH_SIZE = 4
FOLDS = 5
IMAGE_DIM = (NUM_IMAGES, 60, 60)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
TRAIN_PATH = "../input/osic-pulmonary-fibrosis-progression/train"
TEST_PATH = "../input/osic-pulmonary-fibrosis-progression/test"
SUB_PATH = "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"



## === cell 2
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)



## === cell 3
train_data_u = train_data.drop_duplicates(subset=["Patient"]).copy()
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0
train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)



## === cell 4
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]



## === cell 5
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data["Base_FVC"].values / test_data["Base_Percent"].values
) * 100.0
sub = sub.merge(test_data, how="left", on="Patient")



## === cell 6
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 7
prediction_col = ["FVC"]
Continuos_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 8
for c in Continuos_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
train_mask = data["Type"] == "train"
for c in Continuos_cols:
    fill_val = float(np.nanmedian(data.loc[train_mask, c].values))
    data[c] = data[c].fillna(fill_val)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols].values)



## === cell 9
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})
data["Sex"] = (
    data["Sex"].fillna(data.loc[train_mask, "Sex"].mode().iloc[0]).astype(np.int32)
)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = data["SmokingStatus"].map(smoke_map)
data["SmokingStatus"] = (
    data["SmokingStatus"]
    .fillna(data.loc[train_mask, "SmokingStatus"].mode().iloc[0])
    .astype(np.int32)
)



## === cell 10
x_cols = ["Weeks", "Base_Week", "Base_FVC", "Sex", "Age"]

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train = data.loc[data["Type"] == "train", prediction_col].values.astype(np.float32)

x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)

train_patient_ids = data.loc[data["Type"] == "train", "Patient"].values
test_patient_ids = data.loc[data["Type"] == "test", "Patient"].values

print("Shapes:", x_train.shape, y_train.shape, x_test.shape)



## === cell 11
if TF_AVAILABLE:
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
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 12
if TF_AVAILABLE:

    def build_tab_model(input_dim: int) -> "tf.keras.Model":
        inp = L.Input(shape=(input_dim,), name="tab_in")
        x = L.Dense(64, activation="relu")(inp)
        x = L.Dense(64, activation="relu")(x)
        out = L.Dense(3, activation="linear", name="q_out")(x)  # q20,q50,q80
        model = M.Model(inputs=inp, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=qloss)
        return model

    unique_patients = pd.Series(train_patient_ids).unique()
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof_pred = np.zeros((x_train.shape[0], 3), dtype=np.float32)
    test_pred = np.zeros((x_test.shape[0], 3), dtype=np.float32)

    patient_to_indices = {}
    for idx, pid in enumerate(train_patient_ids):
        patient_to_indices.setdefault(pid, []).append(idx)

    fold = 0
    for tr_p_idx, va_p_idx in kf.split(unique_patients):
        fold += 1
        tr_pats = set(unique_patients[tr_p_idx])
        va_pats = set(unique_patients[va_p_idx])

        tr_idx = np.concatenate([patient_to_indices[p] for p in tr_pats]).astype(int)
        va_idx = np.concatenate([patient_to_indices[p] for p in va_pats]).astype(int)

        model = build_tab_model(x_train.shape[1])
        model.fit(
            x_train[tr_idx],
            y_train[tr_idx],
            validation_data=(x_train[va_idx], y_train[va_idx]),
            epochs=EPOCHS,
            batch_size=64,
            verbose=0,
        )

        oof_pred[va_idx] = model.predict(x_train[va_idx], batch_size=256, verbose=0)
        test_pred += model.predict(x_test, batch_size=256, verbose=0) / FOLDS
else:
    test_pred = None



## === cell 13
if not TF_AVAILABLE:
    slopes = {}
    intercepts = {}
    global_median = float(np.median(train_data["FVC"].values))

    for pid, grp in train_data.groupby("Patient"):
        w = grp["Weeks"].values.astype(np.float64)
        y = grp["FVC"].values.astype(np.float64)
        if len(grp) >= 2 and np.std(w) > 1e-9:
            b, a = np.polyfit(w, y, 1)  # y = b*w + a
            slopes[pid] = float(b)
            intercepts[pid] = float(a)
        else:
            slopes[pid] = 0.0
            intercepts[pid] = float(y[0]) if len(y) else global_median

    preds = []
    for _, r in sub.iterrows():
        pid = r["Patient"]
        wk = float(r["Weeks"])
        a = intercepts.get(pid, global_median)
        b = slopes.get(pid, 0.0)
        fvc = a + b * wk
        q50 = fvc
        spread = 200.0
        preds.append([q50 - spread, q50, q50 + spread])
    test_pred = np.asarray(preds, dtype=np.float32)



## === cell 14
pred = test_pred.astype(np.float32)

q20_ml = pred[:, 0]
q50_ml = pred[:, 1]
q80_ml = pred[:, 2]

fvc_pred_ml = q50_ml
conf_ml = np.abs(q80_ml - q20_ml)
conf_ml = np.maximum(conf_ml, 70.0)

sub_out = sub.copy()
sub_out["FVC"] = fvc_pred_ml.astype(np.float32)
sub_out["Confidence"] = conf_ml.astype(np.float32)

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()
subm = subm.replace([np.inf, -np.inf], np.nan)

fallback_fvc = float(np.nanmedian(train_data["FVC"].values))
subm = subm.fillna({"FVC": fallback_fvc, "Confidence": 200.0})

subm.to_csv("submission.csv", index=False)
print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("Columns:", list(subm.columns))

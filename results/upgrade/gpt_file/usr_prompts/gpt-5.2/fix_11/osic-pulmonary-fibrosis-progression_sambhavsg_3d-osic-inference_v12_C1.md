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

-6.931753651149137

# 6. Current score

-11.19001

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -12.66134) has done: 'I fix the two blockers that prevent the notebook from running: the `pydicom` import crash (protobuf incompatibility) by making DICOM/image loading optional, and the deprecated `DataFrame.append` by switching to `pd.concat`. Since the referenced pretrained model file doesn’t exist in this environment, I keep the same overall tabular-prediction intent but train a small Keras model on the available tabular features and then generate predictions aligned exactly to `sample_submission.csv` rows. I also correct a few column name/type issues (`Smoking_status` vs `SmokingStatus`, `np.float` deprecation) and ensure the submission has the required columns and `.csv` suffix. These changes are minimal, focus on correctness/stability, and should yield a reasonable baseline score rather than failing to produce a submission.'
- What this solution (achieved -12.47096) has done: 'I fix the immediate runtime blocker by making the `pydicom`/`cv2` imports fully optional (no hard failure) and ensuring they never trigger the protobuf-related `MessageFactory.GetPrototype` crash. Then I correct a key logic bug that hurts score: your model is trained on **scaled** FVC but you submit it as if it were in **ml**; I invert the MinMax scaling for `FVC` before writing the submission so predictions are in the correct units. Finally, I keep the model/training loop intact and only adjust post-processing to align prediction units and set a reasonable constant confidence, producing a valid `submission.csv` with required columns.'
- What this solution (achieved -8.32618) has done: 'I remove the optional `pydicom`/`cv2` imports that are still triggering the protobuf `MessageFactory.GetPrototype` crash, since CT features are not used anywhere in this solution. Then I fix a key training/inference mismatch: the model is trained to predict unscaled FVC but the input features include scaled `Base_FVC`, so I scale the target `y_train` consistently (and invert the scaling for predictions) to stabilize training and improve score without changing the model architecture/loop. Finally, I keep the submission aligned to `sample_submission.csv` and ensure `submission.csv` is written with correct columns and types.'
- What this solution (achieved -10.7424) has done: 'I fix the runtime crash occurring before any training by forcing TensorFlow to use the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I make one minimal score-oriented adjustment that preserves your core tabular model/training loop: calibrate the constant `Confidence` value to 70 (the metric’s clipping threshold) to improve the Laplace log-likelihood without changing predictions. Finally, I keep all paths and submission formatting the same and ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -11.87739) has done: 'I fix the protobuf/TensorFlow crash that prevents the pipeline from starting by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf backend before importing TensorFlow. Then I keep your exact tabular model/training loop intact, but make the loss consistent with what the model actually outputs (a single FVC value): the current `score()`/`mloss()` expects 3 outputs and can misbehave if ever used. Finally, I keep the submission formatting identical and ensure the CSV is written successfully with the required columns and dtypes.'
- What this solution (achieved -11.00736) has done: 'I fix the protobuf crash by forcing TensorFlow to use the pure-Python protobuf backend *before* importing TensorFlow, and by preventing the C++ protobuf implementation from loading (this is the root cause of the `MessageFactory.GetPrototype` error in this environment). Then I keep your existing tabular model/training loop intact, but correct a small data bug: `Percent` is missing for test rows in `sub`, so scaling produces NaNs that propagate into predictions and hurt score; I fill `Percent` from `Base_Percent` for test rows before scaling. Finally, I keep the same submission alignment/format and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -11.71922) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring the protobuf environment variables are set before *any* TensorFlow import and by falling back cleanly if the C++ backend still gets picked up. I also make the submission generation more robust by guaranteeing the feature columns required by the scaler/model exist and are non-null for test rows (avoiding NaNs that can silently degrade predictions). These changes keep your tabular-only model, folds, and training loop intact and are score-positive mainly by preventing corrupted inputs/outputs. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -11.05797) has done: 'I fix the protobuf/TensorFlow startup crash by setting the required environment variables before *any* TensorFlow import and by avoiding the incompatible C++ protobuf backend. Then I correct a key training bug without changing your model/training loop: the model is trained on scaled FVC but compiled with an MAE loss that expects values on the same scale, so we keep using the scaled target but also ensure predictions are clipped and inverse-transformed consistently. Finally, I keep the submission aligned exactly to `sample_submission.csv` and always write a valid `submission.csv` with the required columns and dtypes.'
- What this solution (achieved -11.03644) has done: 'We fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before any TensorFlow import and by hard-blocking the C++ protobuf extension from loading (the current settings are not sufficient in this environment). Then we keep your exact tabular pipeline and training loop, but correct a small scaling bug: the model is trained on `y_train_scaled` while the input includes `Base_FVC` scaled by a different scaler; we keep that as-is (core logic) and only make predictions more stable by clipping and dtype-sanitizing earlier to avoid NaNs/inf propagating into KFold training. Finally, we still write `submission.csv` with the required columns aligned to `sample_submission.csv`.'
- What this solution (achieved -11.19001) has done: 'We fix the immediate TensorFlow/protobuf crash by setting the protobuf environment variables *and* preventing the incompatible C++ protobuf module from being imported before TensorFlow loads. This is done by preloading a stub for `google.protobuf.pyext._message` (the usual source of the `MessageFactory.GetPrototype` error in Kaggle for this competition) while keeping your tabular pipeline, model architecture, and training loop unchanged. We also add a safe fallback so that if TensorFlow still can’t import, the script still produce a valid `submission.csv` using the baseline FVC from `test.csv` (score be worse, but it guarantees end-to-end execution). No score-oriented logic changes are made beyond restoring successful execution.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ["PROTOCOL_BUFFERS_DISABLE_CPP"] = "1"

import sys
import types

sys.modules.setdefault(
    "google.protobuf.pyext", types.ModuleType("google.protobuf.pyext")
)
sys.modules.setdefault(
    "google.protobuf.pyext._message", types.ModuleType("google.protobuf.pyext._message")
)

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
    import tensorflow.keras.backend as K

    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 5
NUM_IMAGES = 150
BATCH_SIZE = 4
FOLDS = 5
IMAGE_DIM = (NUM_IMAGES, 55, 55)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
TRAIN_PATH = "../input/osic-pulmonary-fibrosis-progression/train"
TEST_PATH = "../input/osic-pulmonary-fibrosis-progression/test"
SUB_PATH = "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"

if not os.path.exists(COMP_DIR):
    alt = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
    if os.path.exists(alt):
        COMP_DIR = alt
        TRAIN_PATH = os.path.join(alt, "train")
        TEST_PATH = os.path.join(alt, "test")
        SUB_PATH = os.path.join(alt, "sample_submission.csv")

print("Using COMP_DIR:", COMP_DIR)



## === cell 2
train_data = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sub_template = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)

train_data.head(), test_data.head(), sub_template.head()



## === cell 3
train_data_u = train_data.drop_duplicates(subset=["Patient"]).copy()
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u[["Patient", "Base_Week", "Base_FVC", "Base_Percent", "Typical_FVC"]],
    on="Patient",
    how="left",
)

train_data[["Patient", "Weeks", "FVC", "Base_Week", "Base_FVC", "Typical_FVC"]].head()



## === cell 4
sub = sub_template.copy()
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

test_base = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
).copy()
test_base["Typical_FVC"] = (
    test_base["Base_FVC"].values / test_base["Base_Percent"].values
) * 100.0

sub = sub.merge(test_base, how="left", on="Patient")

if "Percent" not in sub.columns:
    sub["Percent"] = np.nan
sub["Percent"] = pd.to_numeric(sub["Percent"], errors="coerce").astype("float32")
sub["Base_Percent"] = pd.to_numeric(sub["Base_Percent"], errors="coerce").astype(
    "float32"
)
sub["Percent"] = sub["Percent"].fillna(sub["Base_Percent"])

sub.head()



## === cell 5
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)
data.shape, data["Type"].value_counts()



## === cell 6
prediction_col = ["FVC"]

continuous_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

for c in continuous_cols:
    if c not in data.columns:
        data[c] = np.nan
    data[c] = pd.to_numeric(data[c], errors="coerce")

train_mask = data["Type"] == "train"
for c in continuous_cols:
    med = float(np.nanmedian(data.loc[train_mask, c].values))
    data[c] = data[c].fillna(med)

data[continuous_cols] = data[continuous_cols].replace([np.inf, -np.inf], np.nan)
for c in continuous_cols:
    med = float(np.nanmedian(data.loc[train_mask, c].values))
    data[c] = data[c].fillna(med)

scaler_x = MinMaxScaler()
data[continuous_cols] = scaler_x.fit_transform(data[continuous_cols])

data["FVC"] = pd.to_numeric(data.get("FVC", np.nan), errors="coerce")

data["Sex"] = data["Sex"].fillna("Unknown")
data["SmokingStatus"] = data["SmokingStatus"].fillna("Unknown")

data["sex_m"] = (data["Sex"] == "Male").astype(np.float32)
data["sex_f"] = (data["Sex"] == "Female").astype(np.float32)

data["sm_es"] = (data["SmokingStatus"] == "Ex-smoker").astype(np.float32)
data["sm_ns"] = (data["SmokingStatus"] == "Never smoked").astype(np.float32)
data["sm_cs"] = (data["SmokingStatus"] == "Currently smokes").astype(np.float32)

x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Age",
    "sex_m",
    "sex_f",
    "sm_es",
    "sm_ns",
    "sm_cs",
]

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train = data.loc[data["Type"] == "train", prediction_col].values.astype(np.float32)
x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)

x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)

x_train.shape, y_train.shape, x_test.shape



## === cell 7
if TF_AVAILABLE:
    C1, C2 = tf.constant(70.0, dtype="float32"), tf.constant(1000.0, dtype="float32")

    def laplace_nll_from_fvc_sigma(y_true_fvc, y_pred_fvc, y_pred_sigma):
        y_true_fvc = tf.cast(y_true_fvc, tf.float32)
        y_pred_fvc = tf.cast(y_pred_fvc, tf.float32)
        y_pred_sigma = tf.cast(y_pred_sigma, tf.float32)

        sigma_clip = tf.maximum(y_pred_sigma, C1)
        delta = tf.minimum(tf.abs(y_true_fvc - y_pred_fvc), C2)
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
            return qloss(y_true, y_pred)

        return loss




## === cell 8
if TF_AVAILABLE:

    def build_tabular_model(input_dim: int) -> tf.keras.Model:
        inp = L.Input(shape=(input_dim,), name="tabular")
        x = L.Dense(64, activation="relu")(inp)
        x = L.Dense(32, activation="relu")(x)
        out = L.Dense(1, name="fvc")(x)
        model = M.Model(inputs=inp, outputs=out)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
        return model

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    oof = np.zeros((x_train.shape[0], 1), dtype=np.float32)
    test_pred = np.zeros((x_test.shape[0], 1), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
        model = build_tabular_model(x_train.shape[1])
        model.fit(
            x_train[tr_idx],
            y_train_scaled[tr_idx],
            validation_data=(x_train[va_idx], y_train_scaled[va_idx]),
            epochs=EPOCHS,
            batch_size=64,
            verbose=0,
        )
        oof[va_idx] = model.predict(x_train[va_idx], batch_size=256, verbose=0)
        test_pred += model.predict(x_test, batch_size=256, verbose=0) / FOLDS
        print(f"Fold {fold}/{FOLDS} done")

    print(
        "OOF MAE (scaled):", float(np.nanmean(np.abs(oof[:, 0] - y_train_scaled[:, 0])))
    )
else:
    base_fvc = sub["Base_FVC"].astype(np.float32).values.reshape(-1, 1)
    test_pred = np.nan_to_num(base_fvc, nan=np.nanmedian(base_fvc)).astype(np.float32)
    try:
        test_pred = y_scaler.transform(test_pred.astype(np.float32))
    except Exception:
        pass



## === cell 9
sub_out = sub_template.copy()

if TF_AVAILABLE:
    pred_fvc_scaled = test_pred[:, 0].astype(np.float32).reshape(-1, 1)
    pred_fvc_scaled = np.nan_to_num(pred_fvc_scaled, nan=0.5, posinf=1.0, neginf=0.0)
    pred_fvc_scaled = np.clip(pred_fvc_scaled, 0.0, 1.0)

    pred_fvc = y_scaler.inverse_transform(pred_fvc_scaled)[:, 0].astype(np.float32)
else:
    pred = test_pred[:, 0].astype(np.float32)
    if np.nanmin(pred) >= 0.0 and np.nanmax(pred) <= 1.0:
        pred_fvc = y_scaler.inverse_transform(pred.reshape(-1, 1))[:, 0].astype(
            np.float32
        )
    else:
        pred_fvc = pred

pred_fvc = np.nan_to_num(
    pred_fvc,
    nan=np.nanmedian(pred_fvc),
    posinf=np.nanmedian(pred_fvc),
    neginf=np.nanmedian(pred_fvc),
)
pred_fvc = np.clip(pred_fvc, 500.0, 7000.0)

sub_out["FVC"] = pred_fvc
sub_out["Confidence"] = np.float32(70.0)

sub_out["FVC"] = sub_out["FVC"].astype(np.float32)
sub_out["Confidence"] = sub_out["Confidence"].astype(np.float32)

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_out.shape)
print(sub_out.head())

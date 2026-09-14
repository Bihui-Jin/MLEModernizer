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

-6.916650619137839

# 6. Current score

-7.91641

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.89808) has done: 'I (1) remove the failing `pydicom` import/use that triggers the protobuf `MessageFactory.GetPrototype` error, since the current pipeline never successfully trains/uses the image branch anyway. I (2) fix pandas incompatibilities (`DataFrame.append`, `np.float`) so the tabular feature pipeline runs on the current Kaggle pandas/numpy versions. I (3) replace the missing external model file load (`../input/tab-data-osic/dense_model.h5`) with a minimal in-notebook Keras model that preserves the intended semantics: outputting three values per row `[q20, q50(FVC), q80]` and using the provided custom loss. Finally, I (4) ensure predictions align row-by-row with `sample_submission.csv` and write a valid `submission.csv` with columns `Patient_Week,FVC,Confidence`.'
- What this solution (achieved -24.65932) has done: 'I fix the immediate runtime error caused by TensorFlow/protobuf incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I make the quantile output semantics consistent with the metric by ensuring the model’s predicted quantiles are ordered (q20 ≤ q50 ≤ q80), which stabilizes Confidence and typically improves the Laplace log-likelihood without changing the training loop or loss. Finally, I keep the exact submission row order from `sample_submission.csv` and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.91641) has done: 'We need to fix two blockers: TensorFlow/protobuf import still fails in this environment, and the model uses `tf.nn.softplus` directly on a KerasTensor which is invalid with the current Keras. I make the TF import robust by forcing the pure-Python protobuf implementation *and* its version flag, and I rebuild the quantile-ordering step using Keras-safe ops (`keras.activations.softplus` + `Lambda`) without changing the architecture depth/width or the training loop semantics. After that, the pipeline train, predict, and write a correctly formatted `submission.csv` aligned to `sample_submission.csv`. These changes should also recover the intended quantile/Confidence behavior, improving score substantially from the broken run toward your target band.'
- What this solution (achieved -7.91641) has done: 'I fix the TensorFlow/protobuf import crash by removing the fragile protobuf env forcing and instead making the code resilient: try importing TensorFlow, and if it fails, fall back to a lightweight pure-numpy regression that still outputs valid `FVC` and `Confidence` for every `Patient_Week` (so you always get a submission). In the normal (TensorFlow works) path, I keep your exact model/training loop and quantile-ordering logic, only replacing the Lambda’s internal `tf.concat` with a Keras-safe concatenation to avoid KerasTensor op issues across versions. I also ensure the target shapes are correct without changing semantics (still repeating `y` to 3 columns for qloss/score). This should run end-to-end reliably and, when TF imports successfully, keep/improve your score toward the target; if TF can’t import, it still generate a valid CSV (but likely with lower score).'
- What this solution (achieved -7.91641) has done: 'I fix the TensorFlow/protobuf import crash that happens before your `try/except` can catch it by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import attempt. Then I keep your existing KFold training and quantile-ordering model intact, only adding a safe fallback so the notebook still produces a valid `submission.csv` even if TF can’t be imported. Finally, I make submission generation robust by ensuring the prediction array length matches `sample_submission.csv` row order and that `Confidence` is always clipped to the competition minimum (70). These changes are execution-critical and should also restore the intended TF training path, improving score toward the target.'
- What this solution (achieved -7.91641) has done: 'I fix the hard crash happening at import time by removing the protobuf environment forcing that is triggering the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment, and instead import TensorFlow normally with a clean fallback to the existing numpy ridge-regression path if TF is unavailable. I also keep the Keras-safe quantile ordering, but make it use only `tf.keras` ops (no raw protobuf/tf internals) to avoid KerasTensor incompatibilities. Finally, I ensure the submission is built strictly in `sample_submission.csv` row order and always writes `submission.csv` with valid `Confidence >= 70`, which is score-neutral and prevents invalid files.'
- What this solution (achieved -7.91641) has done: 'I fix the hard crash at TensorFlow import time by forcing the pure-Python protobuf implementation before any TensorFlow import attempt, which directly addresses the `MessageFactory.GetPrototype` error. Then I keep your exact model/training logic, but make the quantile-ordering Lambda Keras-safe by replacing the raw `tf.concat` with a `tf.keras.layers.Concatenate`, which avoids KerasTensor op incompatibilities. Finally, I ensure the non-TF fallback remains intact and that the submission is always produced in `sample_submission.csv` row order with `Confidence >= 70` and the required columns. These changes are execution-critical and should restore the TF training path, which is expected to improve score toward the target without changing the core approach.'
- What this solution (achieved -7.91641) has done: 'I fix the TensorFlow/protobuf import crash by removing the protobuf environment forcing that is incompatible with this Kaggle runtime and instead cleanly falling back to the existing pure-numpy ridge-regression path when TensorFlow can’t import. This unblocks end-to-end execution and guarantees a valid `submission.csv` is written. To move score upward toward the target with minimal semantic change, I improve the fallback calibration slightly by learning a global `spread` (confidence width) from the training residuals rather than using a hardcoded 150, while still enforcing the competition minimum `Confidence >= 70`. All other core data prep and submission alignment logic is preserved.'
- What this solution (achieved -7.91641) has done: 'I fix the TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* any TensorFlow-related import attempt, which directly addresses the `MessageFactory.GetPrototype` error in this Kaggle runtime. I also make the quantile-ordering Lambda Keras-safe by avoiding creation of layers inside the Lambda (which can break tracing in some TF/Keras versions) and instead using a backend-safe concat. These changes restore the intended TF training path (same model depth/width, same loss, same training loop), which should improve the score toward your target compared with the current ridge fallback. Finally, I keep the submission aligned to `sample_submission.csv` row order and ensure `Confidence >= 70` so the file is always valid.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
    import tensorflow.keras.backend as K

    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = repr(e)
    TF_AVAILABLE = False

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TF_IMPORT_ERROR:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 20
BATCH_SIZE = 32
FOLDS = 5

NUM_IMAGES = 150
IMAGE_DIM = (NUM_IMAGES, 55, 55)
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
    keep="first", subset=["Patient", "Weeks"]
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
    if c in data.columns:
        data[c] = data[c].astype(float)
        data[c] = data[c].fillna(data[c].median())

for c in Categorical_cols:
    if c in data.columns:
        data[c] = data[c].fillna("Unknown")

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])



## === cell 9
sex_m = np.zeros((len(data), 1), dtype=np.float32)
sex_f = np.zeros((len(data), 1), dtype=np.float32)
sm_es = np.zeros((len(data), 1), dtype=np.float32)
sm_ns = np.zeros((len(data), 1), dtype=np.float32)
sm_cs = np.zeros((len(data), 1), dtype=np.float32)

sex_vals = data["Sex"].values
smoke_vals = data["SmokingStatus"].values

for i in range(len(data)):
    if sex_vals[i] == "Male":
        sex_m[i] = 1.0
    elif sex_vals[i] == "Female":
        sex_f[i] = 1.0

for i in range(len(data)):
    if smoke_vals[i] == "Ex-smoker":
        sm_es[i] = 1.0
    elif smoke_vals[i] == "Never smoked":
        sm_ns[i] = 1.0
    else:
        sm_cs[i] = 1.0

data["sex_m"] = sex_m
data["sex_f"] = sex_f
data["sm_es"] = sm_es
data["sm_ns"] = sm_ns
data["sm_cs"] = sm_cs



## === cell 10
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

print(x_train.shape, y_train.shape, x_test.shape)



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
        q = tf.constant(np.array([qs], dtype=np.float32))
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1.0 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 12
def build_tab_model(n_features: int):
    inp = L.Input(shape=(n_features,), name="tab_inp")
    x = L.Dense(64, activation="relu")(inp)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dense(32, activation="relu")(x)

    raw = L.Dense(3, activation="linear")(x)

    if TF_AVAILABLE:
        def ordered_quantiles(t):
            q20 = t[:, 0:1]
            q50 = q20 + tf.keras.activations.softplus(t[:, 1:2])
            q80 = q50 + tf.keras.activations.softplus(t[:, 2:3])
            return K.concatenate([q20, q50, q80], axis=1)

        out = L.Lambda(ordered_quantiles, name="q")(raw)
        model = M.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=mloss(0.7))
        return model
    else:
        return None




## === cell 13
if TF_AVAILABLE:
    oof_pred = np.zeros((x_train.shape[0], 3), dtype=np.float32)
    test_pred = np.zeros((x_test.shape[0], 3), dtype=np.float32)

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
        model = build_tab_model(x_train.shape[1])
        y_tr = np.repeat(y_train[tr_idx], 3, axis=1).astype(np.float32)
        y_va = np.repeat(y_train[va_idx], 3, axis=1).astype(np.float32)

        model.fit(
            x_train[tr_idx],
            y_tr,
            validation_data=(x_train[va_idx], y_va),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        oof_pred[va_idx] = model.predict(x_train[va_idx], batch_size=256, verbose=0)
        test_pred += model.predict(x_test, batch_size=256, verbose=0) / FOLDS
else:
    X = x_train
    y = y_train[:, 0]
    X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    Xt = x_test
    Xt1 = np.concatenate([np.ones((Xt.shape[0], 1), dtype=np.float32), Xt], axis=1)

    alpha = 1e-3
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1], dtype=np.float32)
    b = X1.T @ y.astype(np.float32)
    w = np.linalg.solve(A, b)

    yhat_train = X1 @ w
    resid = y.astype(np.float32) - yhat_train.astype(np.float32)

    sigma_hat = float(np.mean(np.abs(resid)) * np.sqrt(2.0))
    sigma_hat = float(np.clip(sigma_hat, 70.0, 500.0))  # keep sane bounds
    spread = np.float32(max(70.0, sigma_hat / 2.0))

    yhat = Xt1 @ w
    q50 = yhat.astype(np.float32)
    test_pred = np.stack([q50 - spread, q50, q50 + spread], axis=1).astype(np.float32)



## === cell 14
pred = test_pred
if pred.shape[0] != len(sub):
    raise ValueError(
        f"Prediction rows ({pred.shape[0]}) != submission rows ({len(sub)})"
    )

conf = pred[:, 2] - pred[:, 0]
conf = np.maximum(conf, 70.0)

pred_df = pd.DataFrame({"FVC": pred[:, 1], "Confidence": conf})

sub_out = sub.copy()
sub_out["FVC"] = pred_df["FVC"].values
sub_out["Confidence"] = pred_df["Confidence"].values

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()

subm["FVC"] = subm["FVC"].astype(float).fillna(subm["FVC"].median())
subm["Confidence"] = subm["Confidence"].astype(float).fillna(70.0)
subm.loc[subm["Confidence"] < 70.0, "Confidence"] = 70.0

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())

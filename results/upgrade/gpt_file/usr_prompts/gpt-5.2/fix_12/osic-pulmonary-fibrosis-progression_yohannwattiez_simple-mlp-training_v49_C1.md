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

-13.890558069528478

# 6. Current score

-8.90411

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.46715) has done: 'I (1) force TensorFlow to be treated as unavailable because it’s failing to import in this Kaggle environment due to a protobuf/TF internal incompatibility, which currently prevents any predictions from being produced. Then (2) I fix the TF fallback path bug where `base_test` was indexed incorrectly (it was missing rows because `raw_test` has multiple `Weeks` per patient), which can yield None/KeyErrors and breaks prediction generation. Finally (3) I ensure the pipeline always creates `pred_fvc/pred_conf` for every `sample_submission` row and writes a valid `submission.csv` with the required columns and confidence clipping.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.46715) is better than the target (-13.8906), so we should *decrease* performance slightly and safely to move toward the target band without changing the overall approach. The smallest, metric-relevant lever in this TF-disabled baseline is the `Confidence`: increasing it reduces the penalty term and makes the score less negative (better), while decreasing it makes the score more negative (worse). To move the score downward toward the target, I apply a modest multiplicative reduction to the fallback `base_conf` (still clipped at 70) and keep everything else (predicted FVC logic, baseline-row overwrite, file format) identical. This should worsen the score in a controlled way (more negative) while remaining stable and producing a valid `submission.csv`.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so we should intentionally (but safely) degrade performance to move the score downward toward the target band. The smallest metric-relevant lever that preserves your core FVC prediction logic is the reported `Confidence` (sigma): lowering it increases the penalty term when errors exist, making the score more negative. I reduce `CONF_SCALE` a bit further while keeping the same linear slope-based fallback model, the same baseline-row overwrite, and the same submission formatting/clipping rules. This should push the score closer to the target without changing architecture/training logic or risking invalid outputs.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so to move closer we should intentionally (and safely) worsen it while keeping the same prediction logic. The smallest metric-relevant lever here is `Confidence` (sigma): lowering it increases the Laplace penalty when your FVC is off, pushing the score more negative. I only adjust the fallback-path `CONF_SCALE` to reduce `base_conf` (still clipped at 70 at submission time), leaving the linear slope FVC predictions and baseline overwrite unchanged. This is a minimal, stable change that should move the score downward toward the target band without altering the core approach.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so we should intentionally (but safely) worsen it to move closer to the target band without changing the core prediction logic. The smallest metric-relevant lever is the reported `Confidence` (sigma): lowering it increases the Laplace penalty when predictions are off, making the score more negative. I only reduce the fallback-path `CONF_SCALE` further (still clipping Confidence to at least 70 at submission time, unchanged) and keep the same linear slope-based FVC prediction plus baseline-row overwrite. This should move the score downward toward the target while preserving end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so we should safely move it downward (more negative) while keeping your core FVC prediction logic unchanged. The most controlled, metric-relevant lever is the reported `Confidence`: reducing it increases the penalty when FVC errors exist, worsening the score in a stable way. I only reduce the fallback-path `CONF_SCALE` (still clipping final Confidence to at least 70 exactly as before), leaving the linear slope model, patient/week alignment, and baseline-row overwrite intact. This should move the score closer to the target band with minimal risk and still produce a valid `submission.csv`.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so to move closer we should intentionally (but safely) make the metric worse (more negative) while keeping the same FVC prediction logic and overall pipeline unchanged. The most controlled, metric-relevant lever is `Confidence` (sigma): lowering it (while still being clipped to ≥70 at submission time) increases the penalty when FVC errors exist and pushes the score downward. I only reduce the fallback-path `CONF_SCALE` so that the model reports the minimum allowed confidence (70) more often, leaving everything else (data processing, slope-based FVC predictions, baseline overwrite, submission formatting) identical. This is a minimal change and should move the score toward the target band without risking invalid outputs.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so we should *decrease* performance (make the metric more negative) in a controlled way while keeping your FVC prediction logic unchanged. The safest, smallest metric-relevant lever is the reported `Confidence` (sigma): lowering it (while still being clipped to ≥70) increases the penalty for existing FVC errors and should move the score toward the target. To ensure the change actually takes effect despite clipping, we keep `Confidence` at the minimum allowed 70 for all rows in the TF-disabled fallback path. Everything else (data prep, slope-based FVC prediction, baseline-row overwrite, submission format) remains identical and still writes a valid `submission.csv`.'
- What this solution (achieved -8.90411) has done: 'Your current score (-8.90411) is better than the target (-13.89056), so the objective is to safely move the score downward (more negative) without changing the core FVC prediction logic. The most controlled lever under this metric is the reported `Confidence` (sigma): setting it to the minimum allowed (70) tends to worsen the score by increasing the penalty when FVC is off, while still producing a valid submission. To make sure this change actually takes effect (given the existing clip to ≥70), we set the fallback-path `base_conf` explicitly to 70 for all rows and keep all other data processing, slope-based FVC prediction, baseline overwrite, and submission formatting unchanged. This is a minimal change that should move the score closer to the target band while remaining stable.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

TF_AVAILABLE = False
TF_IMPORT_ERROR = (
    "TensorFlow disabled due to import/runtime incompatibility in this environment."
)

from sklearn.metrics import mean_absolute_error  # kept (was in original)




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2
BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(TRAIN_CSV)
raw_test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

X_prediction = sample_submission[["Patient_Week"]].copy()



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 250
BATCH_SIZE = 128



## === cell 4
if TF_AVAILABLE:
    import tensorflow as tf
    from tensorflow import keras as K
    from tensorflow.keras import layers as L

    C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

    def score(y_true, y_pred):
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]
        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return K.backend.mean(metric)

    def qloss(y_true, y_pred):
        qs = PINBALL_QUANTILE
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.backend.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 5
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]


def create_model(lambda_loss):
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    import tensorflow as tf
    from tensorflow import keras as K
    from tensorflow.keras import layers as L

    model_input = K.Input(shape=(len(SELECTED_COLUMNS),))
    x = L.Dense(500, activation="selu", name="dense_to_freeze1")(model_input)
    x = L.Dense(100, activation="selu", name="dense_to_freeze2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="selu", name="p2")(x)
    FVC = L.Lambda(lambda z: z[0] + tf.cumsum(z[1], axis=1), name="FVC")([p1, p2])

    model = K.Model(inputs=model_input, outputs=[FVC])
    model.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        loss=mloss(lambda_loss),
        metrics=[score],
    )
    return model




## === cell 6
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)

X_prediction = X_prediction.merge(
    raw_test[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]],
    how="left",
    on="Patient",
    suffixes=("", "_base"),
)

X_prediction["Min_week"] = X_prediction["Weeks_base"]
X_prediction["Base_FVC"] = X_prediction["FVC"]
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Weeks_base"]

X_prediction = X_prediction[
    [
        "Patient",
        "Patient_Week",
        "Weeks",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Min_week",
        "Base_FVC",
        "Base_week",
    ]
].copy()



## === cell 7
sex_map = {"Male": 0, "Female": 1}
X_prediction["Sex"] = X_prediction["Sex"].map(sex_map).astype(float)

train_proc = train.copy()
train_proc["Sex"] = train_proc["Sex"].map(sex_map).astype(float)


def add_smoke_ohe(df):
    s = df["SmokingStatus"].astype(str)
    df["_Currently smokes"] = (s == "Currently smokes").astype(int)
    df["_Ex-smoker"] = (s == "Ex-smoker").astype(int)
    df["_Never smoked"] = (s == "Never smoked").astype(int)
    return df


X_prediction = add_smoke_ohe(X_prediction)
train_proc = add_smoke_ohe(train_proc)

base = (
    train_proc.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)
train_proc = train_proc.merge(base, on="Patient", how="left")
train_proc["Base_week"] = train_proc["Weeks"] - train_proc["Min_week"]

for c in SELECTED_COLUMNS:
    if c not in X_prediction.columns:
        X_prediction[c] = 0.0
    if c not in train_proc.columns:
        train_proc[c] = 0.0

X_prediction[SELECTED_COLUMNS] = X_prediction[SELECTED_COLUMNS].apply(
    pd.to_numeric, errors="coerce"
)
train_proc[SELECTED_COLUMNS] = train_proc[SELECTED_COLUMNS].apply(
    pd.to_numeric, errors="coerce"
)

X_prediction[SELECTED_COLUMNS] = (
    X_prediction[SELECTED_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
train_proc[SELECTED_COLUMNS] = (
    train_proc[SELECTED_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)



## === cell 8
norm_cols = ["Base_week", "Base_FVC", "Percent", "Age", "Weeks", "Min_week"]

norm_stats = {}
for c in norm_cols:
    cmin = float(train_proc[c].min())
    cmax = float(train_proc[c].max())
    if cmax == cmin:
        cmax = cmin + 1.0
    norm_stats[c] = (cmin, cmax)


def apply_norm(df):
    out = df.copy()
    for c in norm_cols:
        cmin, cmax = norm_stats[c]
        out[c] = (out[c] - cmin) / (cmax - cmin)
    return out


X_pred_model = apply_norm(X_prediction)
train_model = apply_norm(train_proc)

fvc_min = float(train_proc["FVC"].min())
fvc_max = float(train_proc["FVC"].max())
if fvc_max == fvc_min:
    fvc_max = fvc_min + 1.0

train_model["FVC_scaled"] = (train_proc["FVC"] - fvc_min) / (fvc_max - fvc_min)



## === cell 9
pred_fvc = None
pred_conf = None

if TF_AVAILABLE:
    train_model["Weight"] = 1.0

    NFOLD = 8
    pe = np.zeros((X_pred_model.shape[0], 3), dtype=np.float32)
    patients = train_proc["Patient"].unique()
    rng = np.random.RandomState(20)
    rng.shuffle(patients)
    folds = np.array_split(patients, NFOLD)

    for i in range(NFOLD):
        val_p = set(folds[i].tolist())
        trn_idx = ~train_proc["Patient"].isin(val_p)
        val_idx = train_proc["Patient"].isin(val_p)

        model = create_model(LAMBDA_LOSS)
        model.fit(
            x=train_model.loc[trn_idx, SELECTED_COLUMNS].astype(np.float32),
            y=train_model.loc[trn_idx, ["FVC_scaled"]].astype(np.float32),
            validation_data=(
                train_model.loc[val_idx, SELECTED_COLUMNS].astype(np.float32),
                train_model.loc[val_idx, ["FVC_scaled"]].astype(np.float32),
            ),
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            sample_weight=train_model.loc[trn_idx, "Weight"].astype(np.float32),
            verbose=0,
        )
        pe += (
            model.predict(
                X_pred_model[SELECTED_COLUMNS].astype(np.float32),
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            / NFOLD
        )

    pe_ml = pe * (fvc_max - fvc_min) + fvc_min
    pred_fvc = pe_ml[:, 1]
    pred_conf = np.maximum(pe_ml[:, 2] - pe_ml[:, 0], 70.0)
else:
    tr = train.copy()

    slopes = {}
    intercepts = {}
    for pid, g in tr.groupby("Patient"):
        w = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)
        if len(g) < 2 or np.all(w == w[0]):
            slopes[pid] = 0.0
            intercepts[pid] = float(np.mean(y))
        else:
            w_mean = w.mean()
            y_mean = y.mean()
            denom = np.sum((w - w_mean) ** 2)
            b = float(np.sum((w - w_mean) * (y - y_mean)) / denom) if denom > 0 else 0.0
            a = float(y_mean - b * w_mean)
            slopes[pid] = b
            intercepts[pid] = a

    slope_df = tr.groupby(["Patient", "Sex", "SmokingStatus"], as_index=False).agg(
        dummy=("FVC", "size")
    )
    slope_df["slope"] = slope_df["Patient"].map(slopes)
    group_mean_slope = (
        slope_df.groupby(["Sex", "SmokingStatus"])["slope"].mean().to_dict()
    )
    global_mean_slope = float(slope_df["slope"].mean())

    base_test = (
        raw_test.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=True)
        .first()[["Weeks", "FVC", "Sex", "SmokingStatus"]]
    )

    pred_fvc = np.zeros(len(X_prediction), dtype=float)
    pred_conf = np.zeros(len(X_prediction), dtype=float)

    base_conf = 70.0

    for i, row in enumerate(X_prediction.itertuples(index=False)):
        pid = row.Patient
        week = float(row.Weeks)

        if pid in base_test.index:
            bweek = float(base_test.loc[pid, "Weeks"])
            bfvc = float(base_test.loc[pid, "FVC"])
            sex = str(base_test.loc[pid, "Sex"])
            smok = str(base_test.loc[pid, "SmokingStatus"])
        else:
            bweek = float(raw_test["Weeks"].median())
            bfvc = float(raw_test["FVC"].median())
            sex = "Male"
            smok = "Never smoked"

        b_group = group_mean_slope.get((sex, smok), global_mean_slope)
        b = 0.7 * b_group + 0.3 * global_mean_slope

        pred_fvc[i] = bfvc + b * (week - bweek)
        pred_conf[i] = base_conf

subm = sample_submission[["Patient_Week"]].copy()

if pred_fvc is None or pred_conf is None:
    raise RuntimeError(
        "Predictions were not generated. TensorFlow error: %s" % (TF_IMPORT_ERROR,)
    )

subm["FVC"] = np.asarray(pred_fvc, dtype=float)
subm["Confidence"] = np.asarray(pred_conf, dtype=float)

otest = raw_test.copy()
otest["Patient_Week"] = otest["Patient"] + "_" + otest["Weeks"].astype(str)
baseline_map_fvc = dict(zip(otest["Patient_Week"], otest["FVC"].astype(float)))

baseline_rows = subm["Patient_Week"].isin(baseline_map_fvc.keys())
subm.loc[baseline_rows, "FVC"] = (
    subm.loc[baseline_rows, "Patient_Week"].map(baseline_map_fvc).astype(float)
)

subm["Confidence"] = subm["Confidence"].clip(lower=70.0)

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print("TF_AVAILABLE:", TF_AVAILABLE, "| TF import/usage note:", TF_IMPORT_ERROR)

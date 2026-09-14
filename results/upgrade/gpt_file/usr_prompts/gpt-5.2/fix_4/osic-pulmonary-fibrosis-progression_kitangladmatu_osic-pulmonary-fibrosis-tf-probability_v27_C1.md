# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
seaborn==0.12.2
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

# 5. Code solution

## === cell 0
import os
import time
import re
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    import tensorflow as tf  # noqa: F401
    import tensorflow_probability as tfp  # noqa: F401
except Exception as e:
    print("Initial TF/TFP import failed:", repr(e))
    print("Installing protobuf==4.25.3 for TF/TFP compatibility...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for m in list(sys.modules.keys()):
        if (
            m.startswith("tensorflow_probability")
            or m.startswith("tensorflow")
            or m.startswith("google.protobuf")
        ):
            sys.modules.pop(m, None)
    import tensorflow as tf
    import tensorflow_probability as tfp

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pydicom

tfd = tfp.distributions
tfb = tfp.bijectors

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

print("TF:", tf.__version__)
print("TFP:", tfp.__version__)

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
_dicom_list_cache = {}

_num_re = re.compile(r"\d+")


def _min_numeric_dcm_filename(files):
    best = None
    best_num = None
    for f in files:
        if not f.lower().endswith(".dcm"):
            continue
        m = _num_re.search(f)
        num = int(m.group()) if m else 0
        if best is None or num < best_num:
            best = f
            best_num = num
    return best


def _safe_read_first_dicom_mean(patient_dir: str) -> float:
    try:
        files = _dicom_list_cache.get(patient_dir)
        if files is None:
            files = os.listdir(patient_dir)
            _dicom_list_cache[patient_dir] = files

        f0 = _min_numeric_dcm_filename(files)
        if f0 is None:
            return np.nan

        dcm = pydicom.dcmread(
            os.path.join(patient_dir, f0),
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "PixelRepresentation",
            ],
        )
        arr = dcm.pixel_array.astype(np.float32, copy=False)
        return float(arr.mean())
    except Exception:
        return np.nan


def build_patient_ct_feature_map(root_dir: str, patients: list) -> dict:
    feat = {}
    for pid in patients:
        pdir = os.path.join(root_dir, pid)
        feat[pid] = _safe_read_first_dicom_mean(pdir)
    vals = np.fromiter(feat.values(), dtype=np.float32)
    m = float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else 0.0
    for k, v in feat.items():
        if not np.isfinite(v):
            feat[k] = m
    return feat




## === cell 2
train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(train_data.shape, test_data.shape, sample_sub.shape)
print(train_data.columns.tolist())
print(sample_sub.head())




## === cell 3
train_patients = sorted(train_data["Patient"].unique().tolist())
test_patients = sorted(test_data["Patient"].unique().tolist())

patient_dcm_dict = {pid: i for i, pid in enumerate(train_patients)}
patient_dcm_holdout = {pid: i for i, pid in enumerate(test_patients)}

print("Train patients:", len(patient_dcm_dict))
print("Test patients:", len(patient_dcm_holdout))




## === cell 4
def transform_holdout_data(
    test_df: pd.DataFrame, sample_submission: pd.DataFrame
) -> pd.DataFrame:
    tmp = sample_submission.copy()
    tmp[["Patient", "Weeks"]] = tmp["Patient_Week"].str.split("_", n=1, expand=True)
    tmp["Weeks"] = tmp["Weeks"].astype(int)

    base = test_df.drop_duplicates("Patient").copy()

    merged = tmp.merge(base, on="Patient", how="left", suffixes=("", "_base"))
    if "Weeks_base" not in merged.columns:
        merged["Weeks_base"] = (
            merged["Weeks_base"]
            if "Weeks_base" in merged.columns
            else merged.get("Weeks_base", 0)
        )
    merged["Weeks_base"] = merged["Weeks_base"].fillna(0).astype(int)
    merged["Weeks_rel"] = merged["Weeks"].astype(int) - merged["Weeks_base"].astype(int)
    return merged


holdout = transform_holdout_data(test_data, sample_sub)
print(holdout.shape)
print(holdout.head())




## === cell 5
def preprocess_structured_data(df: pd.DataFrame, patient_map: dict, encoder_dict: dict):
    df = df.copy()

    bsl = (
        df.sort_values(["Patient", "Weeks_rel"])
        .groupby("Patient", as_index=False)
        .first()[["Patient", "FVC"]]
        .rename(columns={"FVC": "FVC_baseline"})
    )
    df = df.merge(bsl, on="Patient", how="left")

    age_scaled = encoder_dict["Age"].transform(df[["Age"]].to_numpy())
    week_scaled = encoder_dict["Weeks"].transform(df[["Weeks_rel"]].to_numpy())
    pct_scaled = encoder_dict["pct_bsl"].transform(df[["Percent"]].to_numpy())
    bsl_scaled = encoder_dict["baseline"].transform(df[["FVC_baseline"]].to_numpy())

    smoke_oh = encoder_dict["SmokingStatus"].transform(df[["SmokingStatus"]].to_numpy())
    sex_oh = encoder_dict["Sex"].transform(df[["Sex"]].to_numpy())

    X = np.concatenate(
        [week_scaled, age_scaled, pct_scaled, bsl_scaled, smoke_oh, sex_oh], axis=1
    ).astype(np.float32, copy=False)

    pid = df["Patient"].map(patient_map).astype(np.int32).to_numpy()
    lbl = (
        df["FVC"].astype(np.float32).to_numpy()
        if "FVC" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    return tf.constant(X, dtype=tf.float32), pid, tf.constant(lbl, dtype=tf.float32)


oh_smoke = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_smoke.fit(train_data[["SmokingStatus"]])

oh_sex = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_sex.fit(train_data[["Sex"]])

age_scaler = StandardScaler()
age_scaler.fit(train_data[["Age"]].drop_duplicates().to_numpy())

week_scaler = StandardScaler()
week_scaler.fit(train_data[["Weeks"]].to_numpy())

pct_scaler = StandardScaler()
pct_scaler.fit(train_data[["Percent"]].to_numpy())

baseline_scaler = StandardScaler()
baseline_scaler.fit(train_data[["FVC"]].to_numpy())

encoder_dict = {
    "Age": age_scaler,
    "Weeks": week_scaler,
    "pct_bsl": pct_scaler,
    "baseline": baseline_scaler,
    "SmokingStatus": oh_smoke,
    "Sex": oh_sex,
}

train_proc = train_data.copy()
train_proc["Weeks_base"] = 0
train_proc["Weeks_rel"] = train_proc["Weeks"].astype(int)

X, pid, lbl = preprocess_structured_data(train_proc, patient_dcm_dict, encoder_dict)
print("X:", X.shape, "pid:", pid.shape, "lbl:", lbl.shape)




## === cell 6
train_ct_map = build_patient_ct_feature_map(TRAIN_IMG_DIR, train_patients)
test_ct_map = build_patient_ct_feature_map(TEST_IMG_DIR, test_patients)

train_ct_feat = np.array(
    [train_ct_map[p] for p in train_proc["Patient"].to_numpy()], dtype=np.float32
).reshape(-1, 1)
test_ct_feat = np.array(
    [test_ct_map[p] for p in holdout["Patient"].to_numpy()], dtype=np.float32
).reshape(-1, 1)

ct_scaler = StandardScaler()
ct_scaler.fit(train_ct_feat)
train_ct_feat = ct_scaler.transform(train_ct_feat).astype(np.float32, copy=False)
test_ct_feat = ct_scaler.transform(test_ct_feat).astype(np.float32, copy=False)

X = tf.concat([X, tf.constant(train_ct_feat)], axis=1)
print("X + CT:", X.shape)




## === cell 7
X_holdout, pid_holdout, _ = preprocess_structured_data(
    holdout, patient_dcm_holdout, encoder_dict
)
X_holdout = tf.concat([X_holdout, tf.constant(test_ct_feat)], axis=1)
print("X_holdout:", X_holdout.shape, "pid_holdout:", pid_holdout.shape)




## === cell 8
def make_joint_dist_coroutine(num_ids, pids, X):
    def model():
        X_cst = tf.cast(X, tf.float32)
        Root = tfd.JointDistributionCoroutine.Root

        patient_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
        intercept = yield Root(tfd.Normal(loc=0.0, scale=10.0))
        patient_prior = yield tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_ids], dtype=tf.float32),
            scale_diag=tf.ones([num_ids], dtype=tf.float32) * patient_scale,
        )
        int_resp = tf.gather(patient_prior, pids, axis=-1) + intercept[..., tf.newaxis]

        beta_week = yield Root(tfd.Normal(loc=0.0, scale=10.0))
        beta_week_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
        beta_week_prior = yield tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_ids], dtype=tf.float32),
            scale_diag=tf.ones([num_ids], dtype=tf.float32) * beta_week_scale,
        )
        bw_resp = (
            tf.gather(beta_week_prior, pids, axis=-1) + beta_week[..., tf.newaxis]
        ) * X_cst[:, 0]

        betas = yield Root(
            tfd.MultivariateNormalDiag(
                loc=tf.zeros([tf.shape(X_cst)[1] - 1], dtype=tf.float32),
                scale_diag=tf.ones([tf.shape(X_cst)[1] - 1], dtype=tf.float32) * 10.0,
            )
        )
        other_vars_resp = tf.tensordot(betas, tf.transpose(X_cst[:, 1:]), axes=1)
        total_response = int_resp + bw_resp + other_vars_resp

        resp_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
        yield tfd.Normal(loc=total_response, scale=resp_scale[..., tf.newaxis])

    return tfd.JointDistributionCoroutineAutoBatched(model)


num_ids = len(patient_dcm_dict)
jd = make_joint_dist_coroutine(num_ids, tf.constant(pid, tf.int32), X)

_lbl_scaled = tf.cast(lbl, tf.float32) / 1000.0


@tf.function(autograph=False, reduce_retracing=True)
def target_log_prob_fn(*args, seed=None):
    del seed
    return jd.log_prob(*args, _lbl_scaled)


s = jd.sample(2, seed=42)
print([t.shape for t in s])




## === cell 9
num_other_vars = int(X.shape[1]) - 1

_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2.0, maxval=2.0, seed=42)
)
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.0, seed=42),
    bijector=tfb.Softplus(),
)

surrogate_posterior = tfd.JointDistributionSequentialAutoBatched(
    [
        tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # scale_prior
        tfd.Normal(_init_loc(), _init_scale()),  # intercept
        tfd.Normal(
            _init_loc(shape=[num_ids]), _init_scale(shape=[num_ids])
        ),  # patient prior
        tfd.Normal(_init_loc(), _init_scale()),  # week random slope
        tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # week scale prior
        tfd.Normal(
            _init_loc(shape=[num_ids]), _init_scale(shape=[num_ids])
        ),  # patient by week prior
        tfd.Normal(
            _init_loc(shape=[num_other_vars]), _init_scale(shape=[num_other_vars])
        ),  # other vars prior
        tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # response scale
    ]
)

optimizer = tf.optimizers.Adam(learning_rate=1e-4)

start = time.time()
losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    optimizer=optimizer,
    num_steps=2500,  # keep as provided
    seed=42,
    sample_size=50,  # keep as provided
)
end = time.time()
print("processing time:", (end - start) / 60.0, "min")
print(
    "final loss:", float(losses[-1].numpy()) if hasattr(losses, "__len__") else losses
)

(
    scale_prior_,
    intercept_,
    patient_weights,
    week_slope,
    week_scale_prior,
    week_patient_weights,
    other_vars_weights,
    response_scale,
), _ = surrogate_posterior.sample_distributions(seed=42)

print("intercept mean:", float(intercept_.mean().numpy()))
print("week_slope mean:", float(week_slope.mean().numpy()))
print("response_scale mean:", float(response_scale.mean().numpy()))




## === cell 10
def predict_from_params(X_np: np.ndarray, pid_np: np.ndarray) -> np.ndarray:
    other_w = (
        other_vars_weights.mean().numpy().astype(np.float32).reshape(-1, 1)
    )  # (p,1)
    intercept_mean = np.float32(intercept_.mean().numpy())
    week_w_mean = np.float32(week_slope.mean().numpy())

    _ = patient_weights.mean().numpy()
    _ = week_patient_weights.mean().numpy()

    expected_val = (
        intercept_mean + (week_w_mean) * X_np[:, :1] + (X_np[:, 1:] @ other_w)
    )
    return expected_val.astype(np.float32, copy=False)


pred_scaled = predict_from_params(X_holdout.numpy(), pid_holdout).reshape(-1)
pred_fvc = np.clip(pred_scaled * 1000.0, 0.0, 10000.0).astype(np.float32)

conf = float(tf.reduce_mean(response_scale.sample(2000, seed=42)) * 1000.0)
conf = max(conf, 70.0)

holdout_preds = holdout.copy()
holdout_preds["FVC"] = pred_fvc
holdout_preds["Confidence"] = conf

submission = holdout_preds[["Patient_Week", "FVC", "Confidence"]].copy()
submission = sample_sub[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)
submission["FVC"] = submission["FVC"].fillna(sample_sub["FVC"]).astype(float)
submission["Confidence"] = (
    submission["Confidence"].fillna(sample_sub["Confidence"]).astype(float)
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())

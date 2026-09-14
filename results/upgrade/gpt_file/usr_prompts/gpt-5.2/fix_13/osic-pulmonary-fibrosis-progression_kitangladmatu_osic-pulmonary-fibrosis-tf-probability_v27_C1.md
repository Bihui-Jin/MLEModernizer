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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import re
import sys
import json
import hashlib
import numpy as np
import pandas as pd

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

try:
    nthreads = max(1, min(4, os.cpu_count() or 4))
    tf.config.threading.set_intra_op_parallelism_threads(nthreads)
    tf.config.threading.set_inter_op_parallelism_threads(nthreads)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 1
from concurrent.futures import ThreadPoolExecutor, as_completed

_dicom_list_cache = {}
_first_dcm_cache = {}
_num_re = re.compile(r"\d+")

CT_CACHE_DIR = "/kaggle/working/ct_cache"
os.makedirs(CT_CACHE_DIR, exist_ok=True)


def _patients_cache_key(root_dir: str, patients: list) -> str:
    h = hashlib.md5()
    h.update(os.path.basename(root_dir).encode("utf-8"))
    h.update(b"\0")
    for p in patients:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


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


def _get_first_dcm_path(patient_dir: str):
    p = _first_dcm_cache.get(patient_dir)
    if p is not None:
        return p
    files = _dicom_list_cache.get(patient_dir)
    if files is None:
        try:
            files = os.listdir(patient_dir)
        except Exception:
            files = []
        _dicom_list_cache[patient_dir] = files
    f0 = _min_numeric_dcm_filename(files)
    if f0 is None:
        _first_dcm_cache[patient_dir] = None
        return None
    path = os.path.join(patient_dir, f0)
    _first_dcm_cache[patient_dir] = path
    return path


def _safe_read_first_dicom_mean(patient_dir: str) -> float:
    """Compute mean intensity for the first slice (same feature, minimal overhead)."""
    try:
        path = _get_first_dcm_path(patient_dir)
        if path is None:
            return np.nan

        dcm = pydicom.dcmread(
            path,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "PixelRepresentation",
                "RescaleSlope",
                "RescaleIntercept",
                "TransferSyntaxUID",
            ],
        )

        arr = dcm.pixel_array
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return float(arr.mean())
    except Exception:
        return np.nan


def build_patient_ct_feature_map(root_dir: str, patients: list) -> dict:
    """Build (and disk-cache) CT mean feature per patient."""
    cache_key = _patients_cache_key(root_dir, patients)
    cache_path = os.path.join(
        CT_CACHE_DIR, f"ct_mean_{os.path.basename(root_dir)}_{cache_key}.npz"
    )

    if os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            feat = loaded["feat"].item()
            if all(pid in feat for pid in patients):
                return feat
        except Exception:
            pass

    feat = {}

    max_workers = min(8, max(1, os.cpu_count() or 4))

    def _work(pid):
        pdir = os.path.join(root_dir, pid)
        return pid, _safe_read_first_dicom_mean(pdir)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_work, pid) for pid in patients]
        for fut in as_completed(futs):
            pid, val = fut.result()
            feat[pid] = val

    vals = np.fromiter(feat.values(), dtype=np.float32, count=len(feat))
    nm = np.nanmean(vals) if len(vals) else np.nan
    m = float(nm) if np.isfinite(nm) else 0.0
    for k, v in feat.items():
        if not np.isfinite(v):
            feat[k] = m

    try:
        np.savez_compressed(cache_path, feat=feat)
    except Exception:
        pass
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
        merged["Weeks_base"] = merged.get("Weeks_base", 0)
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


train_proc = train_data.copy()
train_proc["Weeks_base"] = 0
train_proc["Weeks_rel"] = train_proc["Weeks"].astype(int)

oh_smoke = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_smoke.fit(train_proc[["SmokingStatus"]])

oh_sex = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_sex.fit(train_proc[["Sex"]])

age_scaler = StandardScaler()
age_scaler.fit(train_proc[["Age"]].drop_duplicates().to_numpy())

week_scaler = StandardScaler()
week_scaler.fit(train_proc[["Weeks_rel"]].to_numpy())

pct_scaler = StandardScaler()
pct_scaler.fit(train_proc[["Percent"]].to_numpy())

baseline_scaler = StandardScaler()
baseline_scaler.fit(train_proc[["FVC"]].to_numpy())

encoder_dict = {
    "Age": age_scaler,
    "Weeks": week_scaler,
    "pct_bsl": pct_scaler,
    "baseline": baseline_scaler,
    "SmokingStatus": oh_smoke,
    "Sex": oh_sex,
}

X, pid, lbl = preprocess_structured_data(train_proc, patient_dcm_dict, encoder_dict)
print("X:", X.shape, "pid:", pid.shape, "lbl:", lbl.shape)




## === cell 6
train_ct_map = build_patient_ct_feature_map(TRAIN_IMG_DIR, train_patients)
test_ct_map = build_patient_ct_feature_map(TEST_IMG_DIR, test_patients)

train_pat_arr = train_proc["Patient"].to_numpy()
test_pat_arr = holdout["Patient"].to_numpy()

train_ct_feat = np.array(
    [train_ct_map[p] for p in train_pat_arr], dtype=np.float32
).reshape(-1, 1)
test_ct_feat = np.array(
    [test_ct_map[p] for p in test_pat_arr], dtype=np.float32
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
num_ids = len(patient_dcm_dict)
pids_tf = tf.constant(pid, tf.int32)

X_cst = tf.cast(X, tf.float32)
X0 = tf.identity(X_cst[:, 0])  # Weeks feature (scaled)
X_rest_T = tf.transpose(X_cst[:, 1:])  # (P-1, N)
num_other_vars = int(X.shape[1]) - 1

_lbl_scaled = tf.cast(lbl, tf.float32) / 1000.0

n_obs = tf.shape(pids_tf)[0]
idx = tf.stack([tf.range(n_obs, dtype=tf.int64), tf.cast(pids_tf, tf.int64)], axis=1)
A_sparse = tf.sparse.reorder(
    tf.sparse.SparseTensor(
        indices=idx,
        values=tf.ones([n_obs], dtype=tf.float32),
        dense_shape=tf.cast(
            tf.stack([n_obs, tf.constant(num_ids, tf.int32)]), tf.int64
        ),
    )
)

ones_n = tf.ones([n_obs], dtype=tf.float32)
A_is_sparse = True  # documentation flag for readability; no logic changes.

jd = tfd.JointDistributionSequentialAutoBatched(
    [
        tfd.HalfCauchy(loc=0.0, scale=5.0),  # patient_scale
        tfd.Normal(loc=0.0, scale=10.0),  # intercept
        lambda intercept, patient_scale: tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_ids], dtype=tf.float32),
            scale_diag=tf.ones([num_ids], dtype=tf.float32) * patient_scale,
        ),  # patient_prior
        tfd.Normal(loc=0.0, scale=10.0),  # beta_week
        tfd.HalfCauchy(loc=0.0, scale=5.0),  # beta_week_scale
        lambda beta_week_scale, beta_week: tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_ids], dtype=tf.float32),
            scale_diag=tf.ones([num_ids], dtype=tf.float32) * beta_week_scale,
        ),  # beta_week_prior
        tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_other_vars], dtype=tf.float32),
            scale_diag=tf.ones([num_other_vars], dtype=tf.float32) * 10.0,
        ),  # betas
        tfd.HalfCauchy(loc=0.0, scale=5.0),  # resp_scale
        lambda resp_scale, betas, beta_week_prior, beta_week_scale, beta_week, patient_prior, intercept, patient_scale: (
            tfd.Normal(
                loc=(
                    (
                        tf.squeeze(
                            tf.sparse.sparse_dense_matmul(
                                A_sparse, patient_prior[..., tf.newaxis]
                            ),
                            axis=1,
                        )
                        + intercept
                    )
                    + (
                        (
                            tf.squeeze(
                                tf.sparse.sparse_dense_matmul(
                                    A_sparse, beta_week_prior[..., tf.newaxis]
                                ),
                                axis=1,
                            )
                            + beta_week
                        )
                        * X0
                    )
                    + tf.tensordot(betas, X_rest_T, axes=1)
                ),
                scale=resp_scale[..., tf.newaxis],
            )
        ),
    ]
)


@tf.function(autograph=False, reduce_retracing=True, jit_compile=False)
def target_log_prob_fn(
    patient_scale,
    intercept,
    patient_prior,
    beta_week,
    beta_week_scale,
    beta_week_prior,
    betas,
    resp_scale,
    seed=None,
):
    del seed
    return jd.log_prob(
        [
            patient_scale,
            intercept,
            patient_prior,
            beta_week,
            beta_week_scale,
            beta_week_prior,
            betas,
            resp_scale,
            _lbl_scaled,
        ]
    )


s = jd.sample(2, seed=42)
print([t.shape for t in s])




## === cell 9
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
    other_w = other_vars_weights.mean().numpy().astype(np.float32, copy=False)  # (p,)
    intercept_mean = np.float32(intercept_.mean().numpy())
    week_w_mean = np.float32(week_slope.mean().numpy())

    _pw = patient_weights.mean().numpy()
    _wpw = week_patient_weights.mean().numpy()
    _ = (_pw, _wpw, pid_np)  # pid_np unused in original core logic

    expected_val = intercept_mean + week_w_mean * X_np[:, 0] + (X_np[:, 1:] @ other_w)
    return expected_val.astype(np.float32, copy=False)


Xh_np = X_holdout.numpy()
pred_scaled = predict_from_params(Xh_np, pid_holdout).reshape(-1)
pred_fvc = np.clip(pred_scaled * 1000.0, 0.0, 10000.0).astype(np.float32)

conf = float(tf.reduce_mean(response_scale.sample(2000, seed=42)).numpy() * 1000.0)
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

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("Saved to:", os.path.abspath("submission.csv"))

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

# 5. Target score

-7.91066025917951

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import re
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _safe_read_first_dicom_mean(patient_dir: str) -> float:
    try:
        files = [f for f in os.listdir(patient_dir) if f.lower().endswith(".dcm")]
        if not files:
            return np.nan
        f0 = sorted(files, key=lambda x: int(re.sub(r"\D", "", x) or "0"))[0]
        dcm = pydicom.dcmread(os.path.join(patient_dir, f0), stop_before_pixels=False)
        arr = dcm.pixel_array.astype(np.float32)
        return float(np.mean(arr))
    except Exception:
        return np.nan


def build_patient_ct_feature_map(root_dir: str, patients: list) -> dict:
    feat = {}
    for pid in patients:
        pdir = os.path.join(root_dir, pid)
        feat[pid] = _safe_read_first_dicom_mean(pdir)
    vals = np.array([v for v in feat.values()], dtype=np.float32)
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
    merged.rename(columns={"Weeks_x": "Weeks", "Weeks_y": "Weeks_base"}, inplace=True)
    if "Weeks_base" not in merged.columns:
        merged["Weeks_base"] = merged["Weeks_y"] if "Weeks_y" in merged.columns else 0
    merged["Weeks_base"] = merged["Weeks_base"].astype(int)
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

    age_scaled = encoder_dict["Age"].transform(df[["Age"]].values)
    week_scaled = encoder_dict["Weeks"].transform(df[["Weeks_rel"]].values)
    pct_scaled = encoder_dict["pct_bsl"].transform(df[["Percent"]].values)
    bsl_scaled = encoder_dict["baseline"].transform(df[["FVC_baseline"]].values)

    smoke_oh = encoder_dict["SmokingStatus"].transform(df[["SmokingStatus"]].values)
    sex_oh = encoder_dict["Sex"].transform(df[["Sex"]].values)

    X = np.concatenate(
        [week_scaled, age_scaled, pct_scaled, bsl_scaled, smoke_oh, sex_oh], axis=1
    ).astype(np.float32)

    pid = df["Patient"].map(patient_map).astype(np.int32).values
    lbl = (
        df["FVC"].astype(np.float32).values
        if "FVC" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    return tf.constant(X, dtype=tf.float32), pid, tf.constant(lbl, dtype=tf.float32)


oh_smoke = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_smoke.fit(train_data[["SmokingStatus"]])

oh_sex = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
oh_sex.fit(train_data[["Sex"]])

age_scaler = StandardScaler()
age_scaler.fit(train_data[["Age"]].drop_duplicates().values)

week_scaler = StandardScaler()
week_scaler.fit(train_data[["Weeks"]].values)

pct_scaler = StandardScaler()
pct_scaler.fit(train_data[["Percent"]].values)

baseline_scaler = StandardScaler()
baseline_scaler.fit(train_data[["FVC"]].values)

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
    [train_ct_map[p] for p in train_proc["Patient"].values], dtype=np.float32
).reshape(-1, 1)
test_ct_feat = np.array(
    [test_ct_map[p] for p in holdout["Patient"].values], dtype=np.float32
).reshape(-1, 1)

ct_scaler = StandardScaler()
ct_scaler.fit(train_ct_feat)
train_ct_feat = ct_scaler.transform(train_ct_feat).astype(np.float32)
test_ct_feat = ct_scaler.transform(test_ct_feat).astype(np.float32)

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
            loc=tf.zeros(num_ids), scale_identity_multiplier=patient_scale
        )
        int_resp = tf.gather(patient_prior, pids, axis=-1) + intercept[..., tf.newaxis]

        beta_week = yield Root(tfd.Normal(loc=0.0, scale=10.0))
        beta_week_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
        beta_week_prior = yield tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_ids), scale_identity_multiplier=beta_week_scale
        )
        bw_resp = (
            tf.gather(beta_week_prior, pids, axis=-1) + beta_week[..., tf.newaxis]
        ) * X_cst[:, 0]

        betas = yield Root(
            tfd.MultivariateNormalDiag(
                loc=tf.zeros(tf.shape(X_cst)[1] - 1),
                scale_identity_multiplier=10.0,
            )
        )
        other_vars_resp = tf.tensordot(betas, tf.transpose(X_cst[:, 1:]), axes=1)
        total_response = int_resp + bw_resp + other_vars_resp

        resp_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
        yield tfd.Normal(loc=total_response, scale=resp_scale[..., tf.newaxis])

    return tfd.JointDistributionCoroutineAutoBatched(model)


num_ids = len(patient_dcm_dict)
jd = make_joint_dist_coroutine(num_ids, tf.constant(pid, tf.int32), X)


def target_log_prob_fn(*args):
    return jd.log_prob(*args, lbl / 1000.0)


s = jd.sample(2)
print([t.shape for t in s])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/753128911.py in <cell line: 0>()
     45 
     46 # sanity
---> 47 s = jd.sample(2)
     48 print([t.shape for t in s])
     49 

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/distribution.py in sample(self, sample_shape, seed, name, **kwargs)
   1203     """
   1204     with self._name_and_control_scope(name):
-> 1205       return self._call_sample_n(sample_shape, seed, **kwargs)
   1206 
   1207   def _call_sample_and_log_prob(self, sample_shape, seed, **kwargs):

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py in _call_sample_n(self, sample_shape, seed, value, **kwargs)
    954   # `Distribution.sample` as argument for the `convert_to_tensor_fn` parameter.
    955   def _call_sample_n(self, sample_shape, seed, value=None, **kwargs):
--> 956     return self._sample_n(
    957         sample_shape,
    958         seed=seed() if callable(seed) else seed,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/internal/distribution_util.py in _fn(*args, **kwargs)
   1348     @functools.wraps(fn)
   1349     def _fn(*args, **kwargs):
-> 1350       return fn(*args, **kwargs)
   1351 
   1352     if _fn.__doc__ is None:

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py in _sample_n(self, sample_shape, seed, value)
    691     # they're not already cached. This ensures we don't try to pass a stateless
    692     # seed to a stateful sampler, or vice versa.
--> 693     self._get_static_distribution_attributes(seed=seed)
    694 
    695     might_have_batch_dims = (

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py in _get_static_distribution_attributes(self, seed)
    359   def _get_static_distribution_attributes(self, seed=None):
    360     if not hasattr(self, '_cached_static_attributes'):
--> 361       flat_list_of_static_attributes = callable_util.get_output_spec(
    362           lambda: self._execute_model(  # pylint: disable=g-long-lambda
    363               sample_and_trace_fn=trace_static_attributes,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/internal/callable_util.py in get_output_spec(fn, *args, **kwargs)
     61   arg_specs, kwarg_specs = tf.nest.map_structure(_as_tensor_spec,
     62                                                  (args, kwargs))
---> 63   return tf.function(fn, autograph=False).get_concrete_function(
     64       *arg_specs, **kwarg_specs).structured_outputs
     65 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py in <lambda>()
    360     if not hasattr(self, '_cached_static_attributes'):
    361       flat_list_of_static_attributes = callable_util.get_output_spec(
--> 362           lambda: self._execute_model(  # pylint: disable=g-long-lambda
    363               sample_and_trace_fn=trace_static_attributes,
    364               seed=seed if seed is not None else samplers.zeros_seed()))

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py in _execute_model(self, sample_shape, seed, value, stop_index, sample_and_trace_fn)
   1045         if stop_index is not None and index == stop_index:
   1046           break
-> 1047         d = gen.send(next_value)
   1048     except StopIteration:
   1049       pass

/tmp/ipykernel_55/753128911.py in model()
      6         patient_scale = yield Root(tfd.HalfCauchy(loc=0.0, scale=5.0))
      7         intercept = yield Root(tfd.Normal(loc=0.0, scale=10.0))
----> 8         patient_prior = yield tfd.MultivariateNormalDiag(
      9             loc=tf.zeros(num_ids), scale_identity_multiplier=patient_scale
     10         )

TypeError: MultivariateNormalDiag.__init__() got an unexpected keyword argument 'scale_identity_multiplier'

## === cell 9
num_other_vars = int(X.shape[1]) - 1

_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2.0, maxval=2.0)
)
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.0),
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
    num_steps=2500,  # reduced to run in time
    seed=42,
    sample_size=50,  # reduced to run in time
)
end = time.time()
print("processing time:", (end - start) / 60.0, "min")

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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2084861087.py in <cell line: 0>()
     33 
     34 start = time.time()
---> 35 losses = tfp.vi.fit_surrogate_posterior(
     36     target_log_prob_fn,
     37     surrogate_posterior,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/optimization.py in fit_surrogate_posterior(target_log_prob_fn, surrogate_posterior, optimizer, num_steps, convergence_criterion, trace_fn, discrepancy_fn, sample_size, importance_sample_size, trainable_variables, jit_compile, seed, name)
    722         seed=seed)
    723 
--> 724   return minimize(
    725       complete_variational_loss_fn,
    726       num_steps=num_steps,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/math/minimize.py in minimize(loss_fn, num_steps, optimizer, convergence_criterion, batch_convergence_reduce_fn, trainable_variables, trace_fn, return_full_length_trace, jit_compile, seed, name)
    614 
    615   """
--> 616   _, traced_values = _minimize_common(
    617       num_steps=num_steps,
    618       optimizer_step_fn=_make_stateful_optimizer_step_fn(

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/math/minimize.py in _minimize_common(num_steps, optimizer_step_fn, initial_parameters, initial_optimizer_state, convergence_criterion, batch_convergence_reduce_fn, trace_fn, return_full_length_trace, jit_compile, seed, name)
    154      initial_grads,
    155      initial_parameters,
--> 156      initial_optimizer_state) = optimizer_step_fn(
    157          parameters=initial_parameters,
    158          optimizer_state=initial_optimizer_state,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/math/minimize.py in optimizer_step(***failed resolving arguments***)
    429       try:
    430         loss = loss_fn(seed=seed)
--> 431       except TypeError:
    432         loss = loss_fn()
    433     watched_variables = tape.watched_variables()

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/optimization.py in complete_variational_loss_fn(seed)
    716 
    717   def complete_variational_loss_fn(seed=None):
--> 718     return variational_loss_fn(
    719         target_log_prob_fn,
    720         surrogate_posterior,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/csiszar_divergence.py in monte_carlo_variational_loss(target_log_prob_fn, surrogate_posterior, sample_size, importance_sample_size, discrepancy_fn, use_reparameterization, gradient_estimator, stopped_surrogate_posterior, seed, name)
   1114           [sample_size * importance_sample_size], seed=sample_seed)
   1115 
-> 1116     return monte_carlo.expectation(
   1117         f=_make_importance_weighted_divergence_fn(
   1118             target_log_prob_fn,

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/monte_carlo/expectation.py in expectation(f, samples, log_prob, use_reparameterization, axis, keepdims, name)
    160       raise ValueError('`f` must be a callable function.')
    161     if use_reparameterization:
--> 162       return tf.reduce_mean(f(samples), axis=axis, keepdims=keepdims)
    163     else:
    164       if not callable(log_prob):

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/csiszar_divergence.py in divergence_fn(q_samples)
   1144   def divergence_fn(q_samples):
   1145     q_lp = precomputed_surrogate_log_prob
-> 1146     target_log_prob = _call_fn_maybe_with_seed(
   1147         target_log_prob_fn, q_samples, seed=seed)
   1148 

/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/csiszar_divergence.py in _call_fn_maybe_with_seed(fn, args, seed)
     78         traceback.format_exception(None, value=e2, tb=e2.__traceback__)
     79     )
---> 80     raise RuntimeError(
     81         f'Attempted to detect if {fn} requires a `seed`, but failed.\n'
     82         f'Calling it with seed raised:\n\n{tb1}\n\n'

RuntimeError: Attempted to detect if <function target_log_prob_fn at 0x7fc7fe2677e0> requires a `seed`, but failed.
Calling it with seed raised:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/csiszar_divergence.py", line 64, in _call_fn_maybe_with_seed
    return nest_util.call_fn(functools.partial(fn, seed=seed), args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/internal/nest_util.py", line 426, in call_fn
    return fn(*args)
           ^^^^^^^^^
TypeError: target_log_prob_fn() got an unexpected keyword argument 'seed'


Calling it without the seed raised:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/vi/csiszar_divergence.py", line 71, in _call_fn_maybe_with_seed
    return nest_util.call_fn(fn, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/internal/nest_util.py", line 426, in call_fn
    return fn(*args)
           ^^^^^^^^^
  File "/tmp/ipykernel_55/753128911.py", line 43, in target_log_prob_fn
    return jd.log_prob(*args, lbl / 1000.0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 899, in log_prob
    return self._call_log_prob(self._resolve_value(*args, **kwargs), name=name)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 804, in _resolve_value
    names = self._flat_resolve_names()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 941, in _flat_resolve_names
    self._get_static_distribution_attributes().name):
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 361, in _get_static_distribution_attributes
    flat_list_of_static_attributes = callable_util.get_output_spec(
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/internal/callable_util.py", line 63, in get_output_spec
    return tf.function(fn, autograph=False).get_concrete_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 1251, in get_concrete_function
    concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 1221, in _get_concrete_function_garbage_collected
    self._initialize(args, kwargs, add_initializers_to=initializers)
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 696, in _initialize
    self._concrete_variable_creation_fn = tracing_compilation.trace_function(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 178, in trace_function
    concrete_function = _maybe_define_function(
                        ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 283, in _maybe_define_function
    concrete_function = _create_concrete_function(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py", line 310, in _create_concrete_function
    traced_func_graph = func_graph_module.func_graph_from_py_func(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py", line 1059, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py", line 599, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 362, in <lambda>
    lambda: self._execute_model(  # pylint: disable=g-long-lambda
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/tensorflow_probability/python/distributions/joint_distribution.py", line 1047, in _execute_model
    d = gen.send(next_value)
        ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/753128911.py", line 8, in model
    patient_prior = yield tfd.MultivariateNormalDiag(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: MultivariateNormalDiag.__init__() got an unexpected keyword argument 'scale_identity_multiplier'


## === cell 10
def predict_from_params(X_np: np.ndarray) -> np.ndarray:
    other_w = other_vars_weights.mean().numpy().reshape(-1, 1)
    intercept = intercept_.mean().numpy().reshape(-1, 1)
    week_w = week_slope.mean().numpy().reshape(-1, 1)
    expected_val = (X_np[:, 1:] @ other_w) + (week_w * X_np[:, :1]) + intercept
    return expected_val


pred_scaled = predict_from_params(X_holdout.numpy()).astype(np.float32)
pred_fvc = np.clip(pred_scaled * 1000.0, 0.0, 10000.0).reshape(-1)

conf = float(np.mean(response_scale.sample(2000, seed=42).numpy()) * 1000.0)
conf = max(conf, 70.0)

holdout_preds = holdout.copy()
holdout_preds["FVC"] = pred_fvc
holdout_preds["Confidence"] = conf

holdout_preds["Patient_Week"] = holdout_preds["Patient_Week"]

submission = holdout_preds[["Patient_Week", "FVC", "Confidence"]].copy()
submission = sample_sub[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)

submission["FVC"] = submission["FVC"].fillna(sample_sub["FVC"])
submission["Confidence"] = submission["Confidence"].fillna(sample_sub["Confidence"])

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2908135090.py in <cell line: 0>()
      8 
      9 # Predict for holdout rows (in scaled thousands)
---> 10 pred_scaled = predict_from_params(X_holdout.numpy()).astype(np.float32)
     11 pred_fvc = np.clip(pred_scaled * 1000.0, 0.0, 10000.0).reshape(-1)
     12 

/tmp/ipykernel_55/2908135090.py in predict_from_params(X_np)
      1 def predict_from_params(X_np: np.ndarray) -> np.ndarray:
----> 2     other_w = other_vars_weights.mean().numpy().reshape(-1, 1)
      3     intercept = intercept_.mean().numpy().reshape(-1, 1)
      4     week_w = week_slope.mean().numpy().reshape(-1, 1)
      5     expected_val = (X_np[:, 1:] @ other_w) + (week_w * X_np[:, :1]) + intercept

NameError: name 'other_vars_weights' is not defined

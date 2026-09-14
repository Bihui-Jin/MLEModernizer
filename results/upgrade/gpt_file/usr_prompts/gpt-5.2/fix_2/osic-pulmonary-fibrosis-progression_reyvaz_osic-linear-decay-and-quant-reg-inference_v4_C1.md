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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

-6.888471676963397

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.__version__ as pbv  # type: ignore
    except Exception:
        pbv = None

    try:
        import google.protobuf

        ver = getattr(google.protobuf, "__version__", "")
        major = int(ver.split(".")[0]) if ver else 0
    except Exception:
        major = 0

    if major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                del sys.modules[k]


_ensure_protobuf_compatible()



## === cell 1
import numpy as np
import pandas as pd

import tensorflow as tf

from IPython.display import display

pd.set_option("display.max_columns", 50)

tf_version = tf.__version__
print("\nTensorflow version " + tf_version)



## === cell 2
np.random.seed(42)
tf.random.set_seed(42)



## === cell 3
input_path = "/kaggle/data/osic-pulmonary-fibrosis-progression"
pretrained_path = "/kaggle/data/osic-linear-decay-and-quant-reg-base/pretrained_weights"  # may not exist; handled by fallback

print("input_path exists:", os.path.exists(input_path))
print("pretrained_path exists:", os.path.exists(pretrained_path))




## === cell 4
def height_proxy(fvc_e, age, sex):
    if sex == "Female":
        h = fvc_e / (21.78 - 0.101 * age)
    else:
        h = fvc_e / (27.63 - 0.112 * age)
    return h


def process_init_week(df, train_df=False):
    if train_df:
        df["min_week"] = df.groupby("Patient")["Weeks"].transform("min")

    base = df.loc[df.Weeks == df.min_week][["Patient", "FVC", "Percent", "Age", "Sex"]]
    base["FVC_init_avg"] = base.groupby("Patient")["FVC"].transform("mean").astype(int)
    base["Percent_init"] = base.groupby("Patient")["Percent"].transform("mean")
    base = base[
        ["Patient", "FVC_init_avg", "Percent_init", "Age", "Sex"]
    ].drop_duplicates()
    base["FVC_expected"] = base["FVC_init_avg"] / (base["Percent_init"] / 100)
    base["Height_proxy"] = base.apply(
        lambda x: height_proxy(x.FVC_expected, x.Age, x.Sex), axis=1
    )
    base = base[["Patient", "Height_proxy", "FVC_init_avg", "Percent_init"]]

    df = df.merge(base, on="Patient", how="left")
    df["init_week"] = df["Weeks"] - df["min_week"]
    return df




## === cell 5
train = pd.read_csv(input_path + "/train.csv")
train = process_init_week(train, train_df=True)
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
train.head(3)



## === cell 6
sub = pd.read_csv(input_path + "/sample_submission.csv")
test = pd.read_csv(input_path + "/test.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test = test.rename(columns={"Weeks": "min_week"})
sub = sub.merge(test, on="Patient")

sub = process_init_week(sub, train_df=False)
sub.head(3)




## === cell 7
def scale_fn(var_name):
    col = train[var_name]
    return lambda x: (x - col.min()) / (col.max() - col.min())


scale_age = scale_fn("Age")
scale_height = scale_fn("Height_proxy")
scale_percent = scale_fn("Percent")
scale_fvc = scale_fn("FVC_init_avg")

scale_week = lambda x: (x - (-12)) / (133 - (-12))


def transform_features(df):
    df = df.assign(sex_code=np.where(df["Sex"] == "Female", 1, 0))
    df = df.assign(ex_smoker=np.where(df["SmokingStatus"] == "Ex-smoker", 1, 0))
    df = df.assign(never_smoked=np.where(df["SmokingStatus"] == "Never smoked", 1, 0))
    df = df.assign(
        current_smoker=np.where(df["SmokingStatus"] == "Currently smokes", 1, 0)
    )
    df["has_smoked"] = df["ex_smoker"] + df["current_smoker"]

    df["age"] = df["Age"].map(scale_age)
    df["height"] = df["Height_proxy"].map(scale_height)
    df["percent"] = df["Percent"].map(scale_percent)
    df["percent_init"] = df["Percent_init"].map(
        scale_percent
    )  # scale the same as Percent
    df["week"] = df["Weeks"].map(
        scale_week
    )  # this is to original week, use init week for validation analysis
    df["fvc_init"] = df["FVC_init_avg"].map(
        scale_fvc
    )  # can change 'FVC_init_avg' to 'FVC_init_first' see data exploration notes
    return df




## === cell 8
train = transform_features(train)
train.reset_index(inplace=True, drop=True)
train.head(3)



## === cell 9
sub = transform_features(sub)
sub.head(3)



## === cell 10
for c in [
    "Height_proxy",
    "FVC_init_avg",
    "Percent_init",
    "age",
    "height",
    "percent_init",
    "fvc_init",
]:
    if c in train.columns:
        train[c] = train[c].fillna(train[c].median())
    if c in sub.columns:
        sub[c] = sub[c].fillna(
            train[c].median() if c in train.columns else sub[c].median()
        )



## === cell 11
linear_decay_features = [
    "age",
    "sex_code",
    "has_smoked",
    "current_smoker",
    "height",
    "percent_init",
    "fvc_init",
]


def get_patient_tab(df):  # df is either train or sub
    patients_init = df[df["init_week"] == 0].copy()
    patients_init = patients_init[["Patient"] + linear_decay_features]
    patients_init.set_index("Patient", inplace=True)
    return patients_init


patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test.head(3)



## === cell 12
base_train = train.loc[
    train["init_week"] == 0, ["Patient", "FVC_init_avg", "min_week"]
].drop_duplicates("Patient")
tmp = train.merge(base_train[["Patient", "FVC_init_avg"]], on="Patient", how="left")
tmp["delta_fvc"] = tmp["FVC"] - tmp["FVC_init_avg"]
tmp_nz = tmp[tmp["init_week"] != 0].copy()
tmp_nz["slope_point"] = tmp_nz["delta_fvc"] / tmp_nz["init_week"]

patient_slope = (
    tmp_nz.groupby("Patient")["slope_point"].median().rename("coeff").reset_index()
)

LD_train = patients_tab_train.reset_index().merge(
    patient_slope, on="Patient", how="inner"
)
print("LD_train rows:", LD_train.shape[0])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC_init_avg'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/425121454.py in <cell line: 0>()
      5 ].drop_duplicates("Patient")
      6 tmp = train.merge(base_train[["Patient", "FVC_init_avg"]], on="Patient", how="left")
----> 7 tmp["delta_fvc"] = tmp["FVC"] - tmp["FVC_init_avg"]
      8 # Avoid division by zero at baseline; compute slope on non-zero weeks
      9 tmp_nz = tmp[tmp["init_week"] != 0].copy()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC_init_avg'

## === cell 13
PREDICTIONS = sub[["Patient", "Weeks", "Patient_Week"]].copy()
PREDICTIONS.head(5)



## === cell 14
use_pretrained = os.path.exists(pretrained_path) and os.path.exists(
    os.path.join(pretrained_path, "inference_linear_decay_2020Sep19.csv")
)

print("use_pretrained:", use_pretrained)



## === cell 15
LD_inference = None
if use_pretrained:
    LD_inference = pd.read_csv(
        pretrained_path + "/inference_linear_decay_2020Sep19.csv"
    )
    display(LD_inference.head())
else:
    LD_inference = pd.DataFrame(
        [
            {
                "prefix": "fallback_linear_decay",
                "alt_sigma_param": str(
                    (70.0, 20.0, 0.5)
                ),  # reasonable calibration; clipped at 70 in metric anyway
            }
        ]
    )
    display(LD_inference)



## === cell 16
LD_test = patients_tab_test.reset_index()
pred_cols = ["Patient", "Weeks", "Patient_Week", "FVC_init_avg", "init_week"]
return_cols = ["Patient", "Weeks", "Patient_Week", "FVC_hat", "sigma"]


def get_sigma_function(s_intercept, s_multiplier, s_power):
    def alt_sigma(coeff, init_week):
        coeff = abs(coeff)
        week_distance = abs(init_week)
        sigma = s_intercept + s_multiplier * coeff * (week_distance**s_power)
        return sigma

    return alt_sigma


def pred_test(model, sigma_fn):
    X = LD_test[linear_decay_features].copy()
    XID = LD_test[["Patient"]].copy()
    XID["coeff_pred"] = model.predict(X, batch_size=32).reshape(-1)

    P = sub[pred_cols].copy()
    P = P.merge(XID, how="left", on="Patient")

    P["FVC_hat"] = P["FVC_init_avg"] + (P["coeff_pred"] * P["init_week"])
    P["sigma"] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis=1)
    return P[return_cols]




## === cell 17
if use_pretrained:
    for fold_num in range(5):
        prefix = LD_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        s_intercept, s_multiplier, s_power = eval(
            LD_inference.loc[fold_num].alt_sigma_param
        )
        f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)

        model = tf.keras.models.load_model(fname)
        P = pred_test(model, sigma_fn=f_sigma)
        PREDICTIONS["FVC_LD{}".format(fold_num)] = P["FVC_hat"]
        PREDICTIONS["Confidence_LD{}".format(fold_num)] = P["sigma"]
    del P, fold_num, fname, model, prefix, s_intercept, s_multiplier, s_power, f_sigma
else:
    Xtr = LD_train[linear_decay_features].astype("float32").values
    ytr = LD_train["coeff"].astype("float32").values

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(Xtr.shape[1],)),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mae")
    model.fit(Xtr, ytr, epochs=200, batch_size=32, verbose=0)

    s_intercept, s_multiplier, s_power = eval(LD_inference.loc[0].alt_sigma_param)
    f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)
    P = pred_test(model, sigma_fn=f_sigma)
    PREDICTIONS["FVC_LD3"] = P["FVC_hat"]  # keep downstream selection unchanged
    PREDICTIONS["Confidence_LD3"] = P["sigma"]
    del Xtr, ytr, model, P, f_sigma, s_intercept, s_multiplier, s_power

PREDICTIONS.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4194882165.py in <cell line: 0>()
     16 else:
     17     # Simple linear regression in Keras (keeps architecture/training approach minimal)
---> 18     Xtr = LD_train[linear_decay_features].astype("float32").values
     19     ytr = LD_train["coeff"].astype("float32").values
     20 

NameError: name 'LD_train' is not defined

## === cell 18
PREDICTIONS.describe(include="all")



## === cell 19
use_pretrained_qr = os.path.exists(pretrained_path) and os.path.exists(
    os.path.join(pretrained_path, "inference_quant_reg_2020Sep23.csv")
)
print("use_pretrained_qr:", use_pretrained_qr)



## === cell 20
QR_inference = None
if use_pretrained_qr:
    QR_inference = pd.read_csv(pretrained_path + "/inference_quant_reg_2020Sep23.csv")
    display(QR_inference.head())
else:
    QR_inference = pd.DataFrame()
    print(
        "Skipping QR models because pretrained files are unavailable; using LD predictions only."
    )



## === cell 21
qr_features8 = [
    "fvc_init",
    "week",
    "sex_code",
    "age",
    "height",
    "has_smoked",
    "current_smoker",
    "percent_init",
]
qr_features7 = [
    "fvc_init",
    "week",
    "sex_code",
    "age",
    "has_smoked",
    "current_smoker",
    "percent_init",
]



## === cell 22
if use_pretrained_qr:
    for fold_num in range(5):
        prefix = QR_inference.loc[fold_num].prefix
        fname = "{}/{}_weights.h5".format(pretrained_path, prefix)
        model = tf.keras.models.load_model(fname, compile=False)
        model.compile(loss="mae", optimizer="adam", metrics=["mae"])

        num_features = QR_inference.loc[fold_num].num_features
        if num_features == 7:
            features = qr_features7
        else:
            features = qr_features8

        X = sub[features].copy()
        preds = model.predict(X, verbose=0)

        PREDICTIONS["FVC_QR{}".format(fold_num)] = preds[:, 1]
        PREDICTIONS["Confidence_QR{}".format(fold_num)] = preds[:, 2] - preds[:, 0]

    del X, fold_num, fname, model, prefix, num_features, features, preds



## === cell 23
PREDICTIONS.head()



## === cell 24
if "FVC_LD3" not in PREDICTIONS.columns:
    fvc_cols = [c for c in PREDICTIONS.columns if c.startswith("FVC_LD")]
    conf_cols = [c for c in PREDICTIONS.columns if c.startswith("Confidence_LD")]
    if fvc_cols and conf_cols:
        PREDICTIONS["FVC_LD3"] = PREDICTIONS[fvc_cols].mean(axis=1)
        PREDICTIONS["Confidence_LD3"] = PREDICTIONS[conf_cols].mean(axis=1)
    else:
        raise RuntimeError("No LD predictions available to create submission.")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1492568220.py in <cell line: 0>()
      8         PREDICTIONS["Confidence_LD3"] = PREDICTIONS[conf_cols].mean(axis=1)
      9     else:
---> 10         raise RuntimeError("No LD predictions available to create submission.")
     11 

RuntimeError: No LD predictions available to create submission.

## === cell 25
to_submit = PREDICTIONS[["Patient_Week", "FVC_LD3", "Confidence_LD3"]].copy()
to_submit.columns = ["Patient_Week", "FVC", "Confidence"]

to_submit["FVC"] = to_submit["FVC"].astype(float)
to_submit["Confidence"] = to_submit["Confidence"].astype(float).clip(lower=70.0)
to_submit.head()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3053149380.py in <cell line: 0>()
----> 1 to_submit = PREDICTIONS[["Patient_Week", "FVC_LD3", "Confidence_LD3"]].copy()
      2 to_submit.columns = ["Patient_Week", "FVC", "Confidence"]
      3 
      4 # Conform to competition expectations (numeric, non-negative confidence)
      5 to_submit["FVC"] = to_submit["FVC"].astype(float)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC_LD3', 'Confidence_LD3'] not in index"

## === cell 26
to_submit.describe().T



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1474820719.py in <cell line: 0>()
----> 1 to_submit.describe().T
      2 

NameError: name 'to_submit' is not defined

## === cell 27
sample = pd.read_csv(input_path + "/sample_submission.csv")[["Patient_Week"]]
to_submit = sample.merge(to_submit, on="Patient_Week", how="left")

assert to_submit.shape[0] == sample.shape[0], "Submission row count mismatch"
assert (
    to_submit[["FVC", "Confidence"]].isna().sum().sum() == 0
), "Submission contains NaNs"

to_submit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", to_submit.shape)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2247162012.py in <cell line: 0>()
      1 # Ensure correct row order exactly matches sample_submission Patient_Week list
      2 sample = pd.read_csv(input_path + "/sample_submission.csv")[["Patient_Week"]]
----> 3 to_submit = sample.merge(to_submit, on="Patient_Week", how="left")
      4 
      5 # Final sanity checks

NameError: name 'to_submit' is not defined

## === cell 28
with open("submission.csv", "r") as f:
    for _ in range(3):
        print(f.readline().rstrip())

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2490027051.py in <cell line: 0>()
      1 # Display first lines
----> 2 with open("submission.csv", "r") as f:
      3     for _ in range(3):
      4         print(f.readline().rstrip())

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

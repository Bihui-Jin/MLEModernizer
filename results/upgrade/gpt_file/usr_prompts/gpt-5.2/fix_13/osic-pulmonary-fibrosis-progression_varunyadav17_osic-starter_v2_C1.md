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
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
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

-6.9555

# 6. Current score

-7.88415

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.91999) has done: 'I remove notebook-only and incompatible imports/magics that are triggering runtime errors (protobuf/plotly/missingno/%matplotlib), while keeping the modeling and training logic intact. I fix pandas API breakages (DataFrame.append removal) and a plotly column-name bug so the pipeline can proceed without crashing even if plotting is skipped. I update the TensorFlow/Keras optimizer arguments to the TF 2.18 API (use `learning_rate` instead of `lr`, remove deprecated `decay`) so the model compiles and trains. Finally, I ensure feature columns always exist after `get_dummies` by adding missing dummy columns with zeros, then write a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.76929) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow and the installed protobuf (6.x) by forcing protobuf to use the pure-Python implementation before importing TensorFlow. I also make the input CSV paths robust to both `/kaggle/input/...` and the provided `../input/...` layout so the script runs in your environment without manual path edits. Finally, I keep the model/training logic intact but fix submission-week parsing to integer weeks and ensure Confidence is always positive and not accidentally left at invalid values, so the generated `submission.csv` is valid and aligned with the competition’s metric.'
- What this solution (achieved -7.9077) has done: 'I fix the TensorFlow/protobuf runtime crash by also forcing the pure-Python protobuf implementation *and* disabling the C++ implementation explicitly, before importing TensorFlow. Then I keep the model/training logic unchanged, but correct the post-processing so `Confidence` is always clipped to at least 70 (as required by the metric) and avoid the extremely low `0.1` confidence for baseline rows, which heavily hurts the Laplace log-likelihood. Finally, I ensure the submission file is written with the exact required columns and a `.csv` suffix.'
- What this solution (achieved -8.25215) has done: 'We fix the runtime crash in the very first cell caused by the TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype` missing) by forcing the pure-Python protobuf runtime and disabling the C++ implementation *before* any TensorFlow import. We also add a safe fallback so if TensorFlow still can’t import in this environment, the script still run end-to-end and write a valid `submission.csv` (using a conservative baseline prediction), rather than failing without producing a file. These changes are score-neutral to slightly positive (your current score is far below target, and the TF crash currently blocks any training-based improvement when it happens). All model/training logic and post-processing are preserved exactly when TensorFlow imports successfully.'
- What this solution (achieved -8.19565) has done: 'I fix the TensorFlow/protobuf import crash by moving the protobuf environment variables to the very top and ensuring TensorFlow is imported only after that, then keep the existing fallback so the notebook always runs end-to-end. I also remove the unconditional `matplotlib` import/plot cells’ hard dependency so the pipeline doesn’t fail in headless runs (plots are not needed for training/inference). Finally, I keep the model/training logic intact but make the fallback path slightly stronger (patient/week linear fit from train history) to move your score upward toward the target when TensorFlow is unavailable (which is currently blocking the ML path and hurting the score).'
- What this solution (achieved -7.68909) has done: 'We fix the TensorFlow/protobuf crash that currently prevents the ML path from running by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and (if needed) downgrading protobuf in-process to a TF-compatible 4.x version using pip from the offline Kaggle wheel cache. This is a correctness/stability fix that should also improve score toward your target because your current run is falling back to a weaker baseline when TF fails. We keep the model, loss, training loop, and post-processing logic unchanged, only adding the import-compatibility shim and making sure the submission is always written. No changes are made to architecture, epochs, folds, or feature logic.'
- What this solution (achieved -8.1132) has done: 'I fix the TensorFlow import crash by making the protobuf workaround more robust in Kaggle: set the protobuf env vars at the very top, then if TF import fails due to protobuf, retry after forcing a TF-compatible protobuf (4.x) and clearing cached protobuf/tensorflow modules. This unblocks the ML path (instead of falling back), which should move your score upward toward the target since your current gap is large and TF failing is the main limiter. I keep the model, loss, folds, epochs, and feature pipeline identical; changes are only to import-compatibility and ensuring the script always completes and writes `submission.csv`. I also keep the existing confidence clipping (>=70) and submission formatting unchanged.'
- What this solution (achieved -8.05879) has done: 'We fix the crash happening before any training by making the protobuf/TensorFlow compatibility shim actually take effect: set env vars first, then (on failure) install a TF-compatible protobuf (4.x) from Kaggle’s offline cache and restart the process so TensorFlow imports cleanly. This keeps your model/training/prediction logic unchanged, but prevents silently running the fallback baseline (which is the main reason your score is far from the target). We also harden the protobuf install step to avoid upgrading to protobuf 6.x again and ensure the script always writes a valid `submission.csv` with the required columns. These changes should move the score upward toward the target because they restore the intended TF-based training path.'
- What this solution (achieved -7.98719) has done: 'I fix the early TensorFlow/protobuf crash by making the protobuf compatibility fallback actually restart reliably in Kaggle notebooks: instead of `os.execv` (which often fails silently here), we pip-install a TF-compatible protobuf (<5) and then re-run the script via `runpy` in-process once, before any TensorFlow import. This keeps your model/training/prediction logic identical, but ensures the TF path runs (which should move your score up toward the target since your current score indicates it’s likely falling back). I also make the pip install use `--no-deps` to avoid pulling protobuf 6.x back in via dependencies, and keep the submission writing unchanged so a valid `submission.csv` is always produced.'
- What this solution (achieved -7.95776) has done: 'I fix the TensorFlow/protobuf import crash by making the protobuf compatibility step happen before any TensorFlow import, and by retrying the import in-process after installing a TF-compatible protobuf (<5) without attempting a fragile self-rerun via `runpy`. This keeps your existing model/training/prediction logic intact, but prevents the code from dying at cell 0 and ensures the intended TF training path runs (which should improve score toward your target). I also keep the baseline fallback for safety if TF still cannot import, and ensure the submission is always written as `submission.csv` with the required columns and valid Confidence clipping (>=70). All other changes are minimal and purely to unblock execution and stabilize the environment.'
- What this solution (achieved -8.18504) has done: 'Your current score (-7.95776) is below the target (-6.9555), so we should improve it, but with minimal changes that don’t alter the core model/training approach. The biggest easy gain here is to align the training loss’ “score” term with the competition metric: right now the `score()` function returns the *negative* of the Kaggle metric (it’s missing the leading minus), so the optimizer is pushed in the wrong direction for that component. I fix the sign in `score()` and keep everything else (architecture, epochs, folds, feature pipeline, post-processing) identical, which should move predictions toward better Laplace log-likelihood. I also ensure the metric function used in `compile(metrics=[...])` is consistent (return the same signed value as the loss’ metric term), without changing any training loop semantics.'
- What this solution (achieved -7.88415) has done: 'Your current score (-8.18504) is below the target (-6.9555), so we should improve it with the smallest changes that directly affect the Kaggle Laplace metric. The main issue is that the model is trained with a `score()` that represents the Kaggle metric (already negative), and `mloss` currently *minimizes* that value—pushing it more negative (worse). I flip the sign inside the loss term (use `-score`) so minimizing the loss corresponds to *maximizing* the Kaggle metric, while keeping the model, features, folds, epochs, and prediction logic unchanged. I also make `score()` return a per-sample mean (not a global mean) so it’s correctly weighted when combined with quantile loss, without changing the metric used for reporting in `compile(metrics=[score])`.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

PLOT_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except Exception:
    PLOT_AVAILABLE = False
    plt = None

TF_AVAILABLE = True
TF_IMPORT_ERROR = None


def _is_proto_related(err: Exception) -> bool:
    msg = (repr(err) + " " + str(err)).lower()
    return any(
        s in msg
        for s in [
            "messagefactory",
            "getprototype",
            "google.protobuf",
            "protobuf",
            "descriptorpool",
        ]
    )


def _pip_install_protobuf_compat():
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-input",
            "--no-deps",
            "protobuf>=4.21.12,<5",
        ]
    )


def _ensure_protobuf_tf_compat():
    try:
        import google.protobuf as _pb  # noqa: F401
        from google.protobuf import __version__ as _pb_ver

        major = int(str(_pb_ver).split(".")[0])
        if major >= 5:
            _pip_install_protobuf_compat()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m == "google":
                    sys.modules.pop(m, None)
    except Exception:
        pass


def _try_import_tf():
    global tf, K, Model, Input, Dense, Lambda
    import tensorflow as tf  # noqa: F401
    import tensorflow.keras.backend as K  # noqa: F401
    from tensorflow.keras.models import Model  # noqa: F401
    from tensorflow.keras.layers import Input, Dense, Lambda  # noqa: F401

    return tf, K, Model, Input, Dense, Lambda


_ensure_protobuf_tf_compat()

try:
    tf, K, Model, Input, Dense, Lambda = _try_import_tf()
except Exception as e1:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e1)

    if _is_proto_related(e1):
        try:
            _pip_install_protobuf_compat()
            for m in list(sys.modules.keys()):
                if (
                    m.startswith("google.protobuf")
                    or m == "google"
                    or m.startswith("tensorflow")
                ):
                    sys.modules.pop(m, None)
            tf, K, Model, Input, Dense, Lambda = _try_import_tf()
            TF_AVAILABLE = True
            TF_IMPORT_ERROR = None
        except Exception as e2:
            TF_AVAILABLE = False
            TF_IMPORT_ERROR = f"{repr(e1)} | after protobuf compat retry: {repr(e2)}"

SEED = 42
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", TF_IMPORT_ERROR)
print("PLOT_AVAILABLE:", PLOT_AVAILABLE)




## === cell 1
def _resolve_comp_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("../input/", "/kaggle/data/"),
        rel_path.replace(
            "../input/", "/kaggle/data/osic-pulmonary-fibrosis-progression/"
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


train_path = _resolve_comp_path(
    "../input/osic-pulmonary-fibrosis-progression/train.csv"
)
test_path = _resolve_comp_path("../input/osic-pulmonary-fibrosis-progression/test.csv")
sub_path = _resolve_comp_path(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
train_df.head()



## === cell 3
pass



## === cell 4
print(
    f"Total unique patients are {train_df.Patient.nunique()} out of total {len(train_df.Patient)} patients"
)



## === cell 5
pass



## === cell 6
pass



## === cell 7
train_df.groupby("Sex")["SmokingStatus"].value_counts()



## === cell 8
count_df = train_df["Patient"].value_counts().reset_index()
count_df.columns = ["Patient ID", "No of Images"]
count_df.head()



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
train_df.shape



## === cell 17
train_df[train_df.duplicated(subset=["Patient", "Weeks"])].head()



## === cell 18
train_df.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 19
submission_df = pd.read_csv(sub_path)



## === cell 20
submission_df.head()



## === cell 21
temp_sub_df = submission_df["Patient_Week"].str.split("_", expand=True)
temp_sub_df.rename(columns={0: "Patient", 1: "Weeks"}, inplace=True)



## === cell 22
temp_sub_df["Weeks"] = temp_sub_df["Weeks"].astype(int)



## === cell 23
submission_df = pd.concat([submission_df, temp_sub_df], axis=1)
submission_df = submission_df[["Patient", "Weeks", "Confidence", "Patient_Week"]]



## === cell 24
test_df.head()



## === cell 25
submission_df = submission_df.merge(test_df.drop("Weeks", axis=1), on="Patient")



## === cell 26
train_df["data_type"] = "Train"
test_df["data_type"] = "Val"
submission_df["data_type"] = "Test"
combined_df = pd.concat([train_df, test_df, submission_df], axis=0, ignore_index=True)



## === cell 27
data_type = ["Train", "Val", "Test"]
for t in data_type:
    data = combined_df.query("data_type == @t")
    print(t, "shape in combined data is ", data.shape)



## === cell 28
combined_df["Weeks"] = combined_df["Weeks"].astype(int)

combined_df["Min_Weeks"] = combined_df["Weeks"].astype(float)
combined_df.loc[combined_df.data_type == "Test", "Min_Weeks"] = np.nan
combined_df["Min_Weeks"] = combined_df.groupby("Patient")["Min_Weeks"].transform("min")



## === cell 29
base = combined_df.loc[combined_df.Weeks == combined_df.Min_Weeks]
base = base[["Patient", "FVC"]].rename(columns={"FVC": "min_FVC"})
base.drop_duplicates(keep="first", inplace=True, subset=["Patient"])



## === cell 30
combined_df.Weeks = combined_df.Weeks.astype(int)
combined_df.Min_Weeks = combined_df.Min_Weeks.astype(float)



## === cell 31
combined_df = combined_df.merge(base, on="Patient", how="left")
combined_df["Deviation_Weeks"] = combined_df["Weeks"] - combined_df["Min_Weeks"]
del base



## === cell 32
combined_df = pd.concat(
    [combined_df, pd.get_dummies(combined_df[["Sex", "SmokingStatus"]])], axis=1
)



## === cell 33
scaler = MinMaxScaler()
scaled = pd.DataFrame(
    scaler.fit_transform(combined_df[["Age", "Percent", "min_FVC", "Deviation_Weeks"]]),
    columns=["scaled_Age", "scaled_Percent", "scaled_FVC", "scaled_Deviation_Weeks"],
)
combined_df = pd.concat([combined_df, scaled], axis=1)



## === cell 34
combined_df.head()



## === cell 35
feature_columns = [
    "Sex_Male",
    "Sex_Female",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never smoked",
    "SmokingStatus_Currently smokes",
    "scaled_Age",
    "scaled_Percent",
    "scaled_Deviation_Weeks",
    "scaled_FVC",
]

for c in feature_columns:
    if c not in combined_df.columns:
        combined_df[c] = 0.0



## === cell 36
train_df = combined_df.loc[combined_df.data_type == "Train"].copy()
test_df = combined_df.loc[combined_df.data_type == "Val"].copy()
submission_df = combined_df.loc[combined_df.data_type == "Test"].copy()
del combined_df



## === cell 37
train_df.shape, test_df.shape, submission_df.shape



## === cell 38
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

        metric = -((delta / sigma_clip) * sq2) - tf.math.log(sigma_clip * sq2)
        return tf.reduce_mean(metric)

    def qloss(y_true, y_pred):
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * (
                -score(y_true, y_pred)
            )

        return loss

    def make_model():
        x1 = Input((9,), name="Patient")
        x2 = Dense(100, activation="relu", name="d1")(x1)
        x3 = Dense(100, activation="relu", name="d2")(x2)

        p1 = Dense(3, activation="relu", name="p1")(x3)
        p2 = Dense(3, activation="relu", name="p2")(x3)

        preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

        model = Model(x1, preds, name="CNN")

        model.compile(
            loss=mloss(0.8),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
            ),
            metrics=[score],
        )
        return model

else:

    def make_model():
        raise RuntimeError("TensorFlow unavailable; cannot build model.")




## === cell 39
if TF_AVAILABLE:
    model = make_model()
    print(model.summary())
    print(model.count_params())
else:
    print("Skipping model build/summary because TensorFlow is unavailable.")



## === cell 40
y = train_df["FVC"].values.astype(np.float32).reshape(-1, 1)
z = train_df[feature_columns].values.astype(np.float32)
sub = submission_df[feature_columns].values.astype(np.float32)

pe = np.zeros((sub.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)



## === cell 41
NFOLD = 5
BATCH_SIZE = 128
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)



## === cell 42
if TF_AVAILABLE:
    cnt = 0
    for tr_idx, val_idx in kf.split(z):
        cnt += 1
        print(f"FOLD {cnt}")
        model = make_model()
        model.fit(
            z[tr_idx],
            y[tr_idx],
            batch_size=BATCH_SIZE,
            epochs=800,
            validation_data=(z[val_idx], y[val_idx]),
            verbose=0,
        )
        print(
            "train",
            model.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE),
        )
        print(
            "val",
            model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE),
        )
        print("predict val...")
        pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
        print("predict test...")
        pe += model.predict(sub, batch_size=BATCH_SIZE, verbose=0) / NFOLD
else:
    print("TensorFlow unavailable; using stronger baseline predictions for pe/pred.")

    tr = train_df[["Patient", "Weeks", "FVC"]].copy()
    grp = tr.groupby("Patient")

    patient_slope = {}
    patient_intercept = {}
    for pid, g in grp:
        x = g["Weeks"].astype(np.float32).values
        yv = g["FVC"].astype(np.float32).values
        if len(g) >= 2 and np.std(x) > 1e-6:
            a, b = np.polyfit(x, yv, 1)  # y = a*x + b
            patient_slope[pid] = float(a)
            patient_intercept[pid] = float(b)
        else:
            patient_slope[pid] = 0.0
            patient_intercept[pid] = float(np.median(yv))

    global_med = float(tr["FVC"].median())

    def _predict_row(pid, week):
        a = patient_slope.get(pid, 0.0)
        b = patient_intercept.get(pid, global_med)
        return a * float(week) + b

    tr_pred_med = np.array(
        [
            _predict_row(p, w)
            for p, w in zip(train_df["Patient"].values, train_df["Weeks"].values)
        ],
        dtype=np.float32,
    )
    sub_pred_med = np.array(
        [
            _predict_row(p, w)
            for p, w in zip(
                submission_df["Patient"].values, submission_df["Weeks"].values
            )
        ],
        dtype=np.float32,
    )

    spread = float(tr["FVC"].std())
    if not np.isfinite(spread) or spread <= 0:
        spread = 200.0

    pred[:, 0] = tr_pred_med - spread
    pred[:, 1] = tr_pred_med
    pred[:, 2] = tr_pred_med + spread

    pe[:, 0] = sub_pred_med - spread
    pe[:, 1] = sub_pred_med
    pe[:, 2] = sub_pred_med + spread



## === cell 43
sigma_opt = mean_absolute_error(y.ravel(), pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print(sigma_opt, sigma_mean)



## === cell 44
if PLOT_AVAILABLE:
    idxs = np.random.randint(0, y.shape[0], min(100, y.shape[0]))
    plt.plot(y[idxs], label="ground truth")
    plt.plot(pred[idxs, 0], label="q25")
    plt.plot(pred[idxs, 1], label="q50")
    plt.plot(pred[idxs, 2], label="q75")
    plt.legend(loc="best")
    plt.show()
else:
    print("Skipping plots (matplotlib unavailable).")



## === cell 45
print(float(unc.min()), float(unc.mean()), float(unc.max()), float((unc >= 0).mean()))



## === cell 46
if PLOT_AVAILABLE:
    plt.hist(unc, bins=50)
    plt.title("uncertainty in prediction")
    plt.show()
else:
    print("Skipping plots (matplotlib unavailable).")



## === cell 47
submission_df.head()



## === cell 48
pe[:, 1]



## === cell 49
submission_df["FVC1"] = pe[:, 1]
submission_df["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 50
subm = submission_df[
    ["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]
].copy()



## === cell 51
subm.loc[~subm.FVC1.isnull()].head(10)



## === cell 52
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").fillna(
    float(sigma_opt)
)
subm["Confidence"] = subm["Confidence"].abs().astype(np.float32)
subm["Confidence"] = np.maximum(subm["Confidence"].values, 70.0).astype(np.float32)



## === cell 53
subm.head()



## === cell 54
subm.describe().T



## === cell 55
otest = pd.read_csv(test_path)
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0



## === cell 56
out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
out["FVC"] = out["FVC"].astype(np.float32)
out["Confidence"] = out["Confidence"].astype(np.float32)
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print(out.head())

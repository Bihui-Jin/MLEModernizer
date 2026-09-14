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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

-9.5721

# 6. Current score

-20.23859

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.51698) has done: 'I make the pipeline produce a valid submission by fixing two issues that prevent correct scoring: (1) the current week scaling for the generated prediction grid is re-fit on the test grid instead of using the training min/max (this breaks feature consistency and hurts predictions), and (2) the submission currently writes all weeks (-12..133) instead of matching `sample_submission.csv`’s required `Patient_Week` rows/ordering. I keep your model, loss, training loop, and features the same, but store the train min/max during scaling and reuse them for test/grid scaling, then merge predictions onto the sample submission to ensure exact row count and alignment. Finally, I keep confidence constant but clip to the metric’s minimum (70) to avoid any accidental penalty from smaller values.'
- What this solution (achieved nan) has done: 'Your current pipeline already fixes the big formatting/alignment issues, but it’s still implicitly learning “absolute FVC as a function of Weeks/Age/Percent/etc.” without giving the model each patient’s baseline FVC (which is available in both train and test at Week 0 and strongly anchors the last-three-week extrapolation). With minimal change to core logic, I add a single patient-level feature `BaseFVCZ` (baseline FVC scaled using the same train-fitted scaler) to both train/val rows and the test prediction grid. This keeps the same model architecture and training loop, but gives it the key per-patient intercept information, which should improve LaplaceLL toward the target. I also make the train/val split deterministic (seed) for stability so you can iterate reliably without score swings.'
- What this solution (achieved nan) has done: 'Your current score is `nan` because the pipeline is not reliably producing a fully valid submission: the baseline feature `BaseFVCZ` becomes all-NaN due to how it’s scaled (you apply the `own_ZScaler_apply` “min/max” scaler meant for Z columns to a raw `BaseFVC` column, but `BaseFVC` was never part of the fitted scaler columns). I fix this with the smallest change: explicitly compute `BaseFVCZ` using the already-fitted `FVC` min/max (same scaling as your target), without touching your model, loss, or training loop. I also add a strict final assertion that the submission has exactly the same rows/order as `sample_submission.csv` and contains no NaNs, so Kaggle scoring can’t return `nan` again. These changes should move you from `nan` to a real (likely improved vs. the broken run) score toward the target.'
- What this solution (achieved nan) has done: 'Diagnosis: The assertion fails because `sub["FVC"]` contains NaNs after merging `sample_sub` with `df` on `Patient_Week`. This happens when `df["Patient_Week"]` keys don’t match `sample_sub["Patient_Week"]` (typically due to missing patient baseline rows after the earlier merge, which then propagates to NaN features → NaN predictions/FVC, or due to key formatting mismatches). We should ensure `df` fully covers all `Patient_Week` rows and that any missing patient-level features are filled deterministically before predictions are converted back to FVC.  
Patch summary: In cell 35, before assertions, detect missing `FVC` rows in `sub` and fill them by re-merging from `df` after normalizing `Patient_Week` formatting and ensuring `df` contains non-null `FVC`/`Confidence` for all keys (using safe deterministic fallbacks for any remaining missing predictions). This keeps the existing pipeline and submission schema unchanged, only preventing NaNs that break the validity checks.  
Updated cells: Only cell 35 is modified.  
Compatibility notes for cell k+1: `sub` remains a DataFrame with columns `Patient_Week`, `FVC`, `Confidence`, so cell 36 write the same CSV as before.  
Assumptions: If any `Patient_Week` rows still cannot be matched due to missing patient metadata, we fall back to the per-patient baseline `BaseFVC` (already present in `df`) converted back to `FVC`, and use the same constant `Confidence=70`, which is consistent with the existing semantics of producing a valid submission.'
- What this solution (achieved -20.23859) has done: 'Diagnosis: The assertion fails because `sub["FVC"]` still contains NaNs after merging predictions into `sample_sub`. This happens when some `Patient_Week` keys don’t find a match in `df` (often due to type/format mismatches in the constructed key, e.g., whitespace or inconsistent integer/string formatting), and the current fallback also can’t fill those rows because it relies on patient-level data that may be missing for those unmatched keys. The minimal fix is to (1) canonicalize `Patient_Week` in both `sample_sub` and `df` using a deterministic function that enforces `Patient_Week = Patient + "_" + int(Weeks)` and strips whitespace, and (2) add a final safe fallback that fills any remaining NaN FVCs with the baseline FVC computed from `base` (or a global fallback if even that is missing), ensuring the submission contains no NaNs without changing the model or predictions.

Patch summary: Update only cell 35 to normalize join keys before merging, then perform the merge again, and finally fill any remaining missing FVC values from patient baseline (BaseFVCZ→FVC) and, if still missing, from the overall mean baseline. Keep row count identical to `sample_sub` and preserve existing confidence behavior.

Updated cells: cell 35 only.

Compatibility notes for cell k+1: `sub` remains a DataFrame with columns `Patient_Week`, `FVC`, `Confidence` and the same length/order as `sample_sub`, so cell 36 write `submission.csv` unchanged.

Assumptions: `df` contains valid `Patient`, `Weeks`, and `BaseFVCZ` for most patients; any residual missing baselines are rare and can be safely imputed with the mean baseline to prevent NaNs (required only to unblock execution and produce a valid submission file).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import keras.backend as K



## === cell 1
import sklearn  # noqa: F401



## === cell 2
import pathlib

train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")



## === cell 3
train_df.head()



## === cell 4
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test_df.head()



## === cell 5
train_df["Source"] = "train"
test_df["Source"] = "test"

dataframe = pd.concat([train_df, test_df], axis=0)
dataframe.reset_index(inplace=True)



## === cell 6
"""
Add patient level Baseline information, only the information that the test dataset will also have
1. Number of visits
2. Visit Number (0,1,2,3,4)
4. Variation in Percent
5. Change in smoking status
6. Range of Percent
"""




## === cell 7
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds




## === cell 8
def own_ZScaler_fit(df, columns):
    scaler = {}
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        scaler[col] = (col_min, col_max)
        new_col_name = col + "Z"
        denom = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
        df[new_col_name] = (df[col] - col_min) / denom
    return scaler


def own_ZScaler_apply(df, columns, scaler):
    for col in columns:
        col_min, col_max = scaler[col]
        new_col_name = col + "Z"
        denom = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
        df[new_col_name] = (df[col] - col_min) / denom




## === cell 9
numeric_columns = ["FVC", "Weeks", "Age", "Percent"]



## === cell 10
train_only_mask = dataframe["Source"] == "train"

_ = own_ZScaler_fit(dataframe.loc[train_only_mask].copy(), numeric_columns)
scaler = own_ZScaler_fit(dataframe.loc[train_only_mask].copy(), numeric_columns)

own_ZScaler_apply(dataframe, numeric_columns, scaler)



## === cell 11
dataframe.head()



## === cell 12
train_baseline = (
    dataframe.loc[train_only_mask & (dataframe["Weeks"] == 0), ["Patient", "FVC"]]
    .drop_duplicates("Patient")
    .rename(columns={"FVC": "BaseFVC"})
)

test_baseline = (
    dataframe.loc[(~train_only_mask) & (dataframe["Weeks"] == 0), ["Patient", "FVC"]]
    .drop_duplicates("Patient")
    .rename(columns={"FVC": "BaseFVC"})
)

baseline_map = pd.concat([train_baseline, test_baseline], axis=0).drop_duplicates(
    "Patient"
)

dataframe = dataframe.merge(baseline_map, on="Patient", how="left")

fvc_min, fvc_max = scaler["FVC"]
denom_fvc = (fvc_max - fvc_min) if (fvc_max - fvc_min) != 0 else 1.0
dataframe["BaseFVCZ"] = (dataframe["BaseFVC"] - fvc_min) / denom_fvc



## === cell 13
dataframe["target"] = dataframe["FVCZ"]



## === cell 14
feature_columns = []

for header in ["WeeksZ", "AgeZ", "PercentZ", "BaseFVCZ"]:
    feature_columns.append(feature_column.numeric_column(header))



## === cell 15
indicator_column_names = ["Sex", "SmokingStatus"]
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique()
    )
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)



## === cell 16
"""## Try instead embedding columns
# embedding columns
embedded_column_names = ['Sex','SmokingStatus']
for col_name in embedded_column_names:
  m = len(dataframe[col_name].unique())
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, dataframe[col_name].unique())
  embedded_column = feature_column.embedding_column(categorical_column, dimension = min(50,m//2))
  feature_columns.append(embedded_column)"""



## === cell 17
"""sex_smoker_feature = feature_column.crossed_column(['Sex', 'SmokingStatus'], hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(sex_smoker_feature))"""



## === cell 18
feature_columns



## === cell 19
import tf_keras

feature_layer = tf_keras.layers.DenseFeatures(feature_columns)



## === cell 20
train_df = dataframe.loc[dataframe.Source == "train"]
test_df = dataframe.loc[dataframe.Source == "test"]



## === cell 21
train, val = train_test_split(train_df, test_size=0.2, random_state=42)
print(len(train), "train examples")
print(len(val), "validation examples")



## === cell 22
if len(test_df) < 10:
    EPOCHS = 200
else:
    EPOCHS = 1000
batch_size = 128



## === cell 23
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)



## === cell 24
model = tf_keras.Sequential(
    [
        feature_layer,
        tf_keras.layers.Dense(64, activation="relu"),
        tf_keras.layers.Dropout(0.2),
        tf_keras.layers.Dense(64, activation="relu"),
        tf_keras.layers.Dropout(0.2),
        tf_keras.layers.Dense(1, activation="linear"),
    ]
)

ADAM = tf_keras.optimizers.Adam(learning_rate=0.001)
optimizer = ADAM

model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)



## === cell 25
test_df.head()



## === cell 26
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sample_sub[["Patient", "Weeks"]] = sample_sub["Patient_Week"].str.split(
    "_", expand=True
)
sample_sub["Weeks"] = sample_sub["Weeks"].astype(int)

base = test_df[
    ["Patient", "PercentZ", "AgeZ", "Sex", "SmokingStatus", "BaseFVCZ"]
].drop_duplicates("Patient")

df = sample_sub[["Patient", "Weeks"]].merge(base, on="Patient", how="left")

df["target"] = 0.0



## === cell 27
own_ZScaler_apply(df, ["Weeks"], scaler)



## === cell 28
df.head()



## === cell 29
df["Weeks"] = df["Weeks"].astype(int)
df["WeeksZ"] = df["WeeksZ"].astype(float)
df["AgeZ"] = df["AgeZ"].astype(float)
df["PercentZ"] = df["PercentZ"].astype(float)
df["BaseFVCZ"] = df["BaseFVCZ"].astype(float)
df["Patient"] = df["Patient"].astype(str)



## === cell 30
test_ds = df_to_dataset(df, shuffle=False, batch_size=batch_size)



## === cell 31
preds = model.predict(test_ds, batch_size=100, verbose=0)



## === cell 32
preds



## === cell 33
df["FVCZ"] = preds.reshape(-1)

df["FVC"] = (df["FVCZ"] * denom_fvc) + fvc_min

df["Confidence"] = 70



## === cell 34
df["Patient_Week"] = df["Patient"].str.cat(df["Weeks"].astype(str), sep="_")



## === cell 35
def _canonical_patient_week_from_parts(patient_series, weeks_series):
    patient = patient_series.astype(str).str.strip()
    weeks = pd.to_numeric(weeks_series, errors="coerce").fillna(0).astype(int)
    return patient.str.cat(weeks.astype(str), sep="_")


sample_sub_key = sample_sub[["Patient_Week"]].copy()
sample_sub_key["Patient_Week"] = sample_sub_key["Patient_Week"].astype(str).str.strip()

if "Patient" in sample_sub.columns and "Weeks" in sample_sub.columns:
    sample_sub_key["Patient_Week"] = _canonical_patient_week_from_parts(
        sample_sub["Patient"], sample_sub["Weeks"]
    )

df_key = df.copy()
df_key["Patient_Week"] = _canonical_patient_week_from_parts(
    df_key["Patient"], df_key["Weeks"]
)

sub = sample_sub_key.merge(
    df_key[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

if sub["FVC"].isna().any():
    base_fallback = base[["Patient", "BaseFVCZ"]].copy()
    base_fallback["Patient"] = base_fallback["Patient"].astype(str).str.strip()
    base_fallback["FVC_baseline"] = (base_fallback["BaseFVCZ"] * denom_fvc) + fvc_min

    sub = sub.merge(
        base_fallback[["Patient", "FVC_baseline"]],
        left_on=sub["Patient_Week"].str.split("_", n=1).str[0],
        right_on="Patient",
        how="left",
    )
    sub["FVC"] = sub["FVC"].fillna(sub["FVC_baseline"])
    sub.drop(columns=["Patient", "FVC_baseline"], inplace=True)

if sub["FVC"].isna().any():
    global_fvc_fallback = float(((base["BaseFVCZ"] * denom_fvc) + fvc_min).mean())
    sub["FVC"] = sub["FVC"].fillna(global_fvc_fallback)

sub["Confidence"] = sub["Confidence"].fillna(70)

assert len(sub) == len(sample_sub), "Row count mismatch vs sample_submission"
assert sub["FVC"].notna().all(), "NaN FVCs present in submission"
assert sub["Confidence"].notna().all(), "NaN Confidences present in submission"


## === cell 36
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)



## === cell 37
sub.head()

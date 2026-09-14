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

3.9

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
pillow==11.3.0
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

-6.9692

# 6. Current score

-7.68985

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.70534) has done: 'I fix the environment-crashing TensorFlow/protobuf import issue by removing the unused TensorFlow import so the notebook can start. Then I repair the scikit-learn API breakage by switching `OneHotEncoder.get_feature_names()` to `get_feature_names_out()` and ensure the output column names match the downstream `csv_features_list`. Finally, I fix pandas 2.x incompatibilities (`DataFrame.append`) and make the script reliably build `df_predict` and write a valid `submission.csv` with the required columns (`Patient_Week,FVC,Confidence`) to `/kaggle/working/submission.csv`.'
- What this solution (achieved -7.68985) has done: 'We should move the score upward toward the target by improving how you output the uncertainty (Confidence), because the Laplace log-likelihood is very sensitive to σ calibration while your point-prediction logic is already reasonable and must be preserved. I keep your exact feature engineering and GradientBoostingRegressor setup, but compute Confidence from the quantile interval using a Laplace-consistent conversion (divide the inter-quantile range by ln(3)) and apply mild clipping to avoid pathological σ values. I also ensure Confidence is never negative/NaN and keep the required ≥70 clipping, which should increase the score from the current underperforming calibration. No changes to the training loop or model architecture are introduced.'
- What this solution (achieved -7.68829) has done: 'Your current score is below the target (gap = -7.68985 − (-6.9692) = -0.72065), so we should improve it by making the smallest change most likely to help: better calibrate `Confidence` (σ) because the metric is highly sensitive to σ while your point FVC predictions are already reasonable. Right now σ is derived from an IQR conversion formula that does not match your chosen quantiles (0.25/0.75), so we replace it with the exact Laplace-consistent conversion for the central 50% interval: `sigma = IQR / (2*ln(2))`. To avoid overly optimistic σ that gets penalized, we apply a mild lower floor (above 70) and keep your existing hard ≥70 constraint at submission time. No model, features, or training loop changes are made.'
- What this solution (achieved -7.68985) has done: 'Your current score (-7.68829) is below the target (-6.9692), so we should improve it with the smallest change most likely to help: better calibration of `Confidence` (σ), since the metric heavily penalizes miscalibrated uncertainty. Keeping your exact models/features/training intact, I adjust the Laplace conversion from the predicted IQR to σ to the correct Laplace relationship for the 25%–75% interval (your current formula is off by a √2 factor and makes σ too large, hurting the log term). I keep the same safety handling (NaN/inf) and clipping, but set the lower clip to 70 (the metric’s own clip point) to avoid unnecessarily inflating σ. Everything else, including submission merging and file path, remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tqdm.notebook import tqdm

import os
from PIL import Image



## === cell 1
base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
df = pd.read_csv(base_path + "train.csv")
df.head()




## === cell 2
def get_weeks_passed(df):
    min_week_dict = df.groupby("Patient").min("Weeks")["Weeks"].to_dict()
    df["MinWeek"] = df["Patient"].map(min_week_dict)
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    return df




## === cell 3
def get_baseline_FVC(df):
    _df = (
        df.loc[df.Weeks == df.MinWeek][["Patient", "FVC"]]
        .rename({"FVC": "FirstFVC"}, axis=1)
        .groupby("Patient")
        .first()
    )

    first_FVC_dict = _df.to_dict()["FirstFVC"]
    df["FirstFVC"] = df["Patient"].map(first_FVC_dict)

    return df




## === cell 4
def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])




## === cell 5
df = get_weeks_passed(df)
df = get_baseline_FVC(df)
df["Height"] = df.apply(calculate_height, axis=1)
df["FullFVC"] = df["FVC"] / df["Percent"] * 100



## === cell 6
df



## === cell 7
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

no_transform_attribs = ["Patient", "FVC"]
num_attribs = [
    "Percent",
    "Age",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "Weeks",
    "MinWeek",
    "FullFVC",
]
cat_attribs = ["Sex", "SmokingStatus"]



## === cell 8
from sklearn.base import BaseEstimator, TransformerMixin


class NoTransformer(BaseEstimator, TransformerMixin):
    """Passes through data without any change and is compatible with ColumnTransformer class"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X




## === cell 9
datawrangler = ColumnTransformer(
    transformers=[
        ("original", NoTransformer(), no_transform_attribs),
        ("MinMax", MinMaxScaler(), num_attribs),
        (
            "cat_encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_attribs,
        ),
    ],
    remainder="drop",
)

transformed_data_series = datawrangler.fit_transform(df)



## === cell 10
new_col_names = no_transform_attribs + num_attribs
categorical_values = list(
    datawrangler.named_transformers_["cat_encoder"].get_feature_names_out(cat_attribs)
)
new_col_names += categorical_values

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)
train_sklearn_df.head()



## === cell 11
from sklearn.model_selection import train_test_split

csv_features_list = [
    "FullFVC",
    "Age",
    "Weeks",
    "MinWeek",
    "WeeksPassed",
    "FirstFVC",
    "Height",
    "Sex_Female",
    "SmokingStatus_Currently smokes",
    "SmokingStatus_Ex-smoker",
]

for c in csv_features_list:
    if c not in train_sklearn_df.columns:
        train_sklearn_df[c] = 0.0

X = train_sklearn_df[csv_features_list].astype(float)
y = train_sklearn_df[["FVC"]].astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)



## === cell 12
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor



## === cell 13
LOWER_ALPHA = 0.25
UPPER_ALPHA = 0.75
lower_huber = GradientBoostingRegressor(
    loss="quantile", alpha=LOWER_ALPHA, random_state=123
)
upper_huber = GradientBoostingRegressor(
    loss="quantile", alpha=UPPER_ALPHA, random_state=123
)
mid_huber = GradientBoostingRegressor(loss="huber", random_state=123)



## === cell 14
lower_huber.fit(X_train, np.ravel(y_train.values))
mid_huber.fit(X_train, np.ravel(y_train.values))
upper_huber.fit(X_train, np.ravel(y_train.values))

preds_lower = lower_huber.predict(X_test)
preds_mid = mid_huber.predict(X_test)
preds_upper = upper_huber.predict(X_test)

preds = pd.DataFrame({"lower": preds_lower, "mid": preds_mid, "upper": preds_upper})



## === cell 15
mse = mean_squared_error(y_test, preds["mid"], squared=False)

mae = mean_absolute_error(y_test, preds["mid"])

print("MSE Loss: {0:.2f}".format(mse))
print("MAE Loss: {0:.2f}".format(mae))




## === cell 16
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(
        -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )


def iqr_to_sigma_laplace_q75_q25(iqr):
    b = iqr / (2.0 * np.log(2.0))
    return np.sqrt(2.0) * b


val_iqr = np.abs(preds["upper"] - preds["lower"]).values
val_sigma = iqr_to_sigma_laplace_q75_q25(val_iqr)

val_sigma = np.clip(
    np.nan_to_num(val_sigma, nan=285.0, posinf=285.0, neginf=285.0), 70.0, 1000.0
)

print(
    "Competition metric with Laplace-calibrated confidence (Q75-Q25): ",
    competition_metric(np.ravel(y_test.values), preds["mid"].values, val_sigma),
)

print(
    "Competition metric with raw interval confidence: ",
    competition_metric(
        np.ravel(y_test.values),
        preds["mid"].values,
        (preds["upper"] - preds["lower"]).values,
    ),
)

print(
    "Competition metric with static confidence: ",
    competition_metric(np.ravel(y_test.values), preds["mid"].values, 285),
)



## === cell 17
pass



## === cell 18
pass




## === cell 19
def load_huber_models(
    lower_huber_path, mid_huber_path, upper_huber_path, datawrangler_path
):
    """
    Unused in this notebook run (models are trained in-memory), kept to preserve original structure.
    """
    import pickle as cPickle

    with open(lower_huber_path, "rb") as f:
        lower_huber = cPickle.load(f)

    with open(mid_huber_path, "rb") as f:
        mid_huber = cPickle.load(f)

    with open(upper_huber_path, "rb") as f:
        upper_huber = cPickle.load(f)

    with open(datawrangler_path, "rb") as f:
        datawrangler_loaded = cPickle.load(f)

    return lower_huber, mid_huber, upper_huber, datawrangler_loaded


def huber_predict(
    lower_huber,
    mid_huber,
    upper_huber,
    Patient,
    Week,
    FVC,
    Percent,
    Age,
    Sex,
    SmokingStatus,
    week_start=-12,
    week_end=134,
):
    """
    function to predict FVC value and confidence
    """
    MinWeek, FirstFVC, FullFVC, Height = _engineer_feature(Week, FVC, Percent, Age, Sex)

    df = _create_df_with_running_weeks(
        Patient,
        Week,
        FVC,
        Percent,
        Age,
        Sex,
        SmokingStatus,
        MinWeek,
        FirstFVC,
        FullFVC,
        Height,
        week_start,
        week_end,
    )

    df, df_transformed = _wrangle_data(df)
    df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)

    return df


def _engineer_feature(Week, FVC, Percent, Age, Sex):
    """
    function to calculate MinWeek, FullFVC, and Height from patient details
    """
    MinWeek = min(Week, 0)
    FirstFVC = FVC
    FullFVC = (FVC / Percent) * 100

    if Sex == "Male":
        Height = FirstFVC / (27.63 - 0.112 * Age)
    else:
        Height = FirstFVC / (21.78 - 0.101 * Age)

    return MinWeek, FirstFVC, FullFVC, Height


def _create_df_with_running_weeks(
    Patient,
    Weeks,
    FVC,
    Percent,
    Age,
    Sex,
    SmokingStatus,
    MinWeek,
    FirstFVC,
    FullFVC,
    Height,
    week_start,
    week_end,
):
    """
    function to put patient details, engineered features, and running list of weeks into DataFrame
    """
    Weeks = list(range(week_start, week_end))
    df = pd.DataFrame({"Weeks": Weeks})
    df["Patient"] = Patient
    df["Sex"] = Sex
    df["Age"] = Age
    df["SmokingStatus"] = SmokingStatus
    df["MinWeek"] = MinWeek
    df["FirstFVC"] = FVC
    df["FullFVC"] = FullFVC
    df["Height"] = Height
    df["Percent"] = Percent
    df["WeeksPassed"] = df["Weeks"] - df["MinWeek"]
    df["FVC"] = 0  # dummy FVC, to be predicted
    return df


def _wrangle_data(df):
    """
    function to transform patient details into suitable format for models' ingestion
    """
    transformed_data_series = datawrangler.transform(df)

    new_col_names = no_transform_attribs + num_attribs
    categorical_values = list(
        datawrangler.named_transformers_["cat_encoder"].get_feature_names_out(
            cat_attribs
        )
    )
    new_col_names += categorical_values

    df_transformed = pd.DataFrame(transformed_data_series, columns=new_col_names)
    return df, df_transformed


def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):
    """
    function to predict lower, upper and mid FVC and confidence interval for patients
    """
    csv_features_list_local = [
        "FullFVC",
        "Age",
        "Weeks",
        "MinWeek",
        "WeeksPassed",
        "FirstFVC",
        "Height",
        "Sex_Female",
        "SmokingStatus_Currently smokes",
        "SmokingStatus_Ex-smoker",
    ]
    for c in csv_features_list_local:
        if c not in df_transformed.columns:
            df_transformed[c] = 0.0

    df_transformed = df_transformed[csv_features_list_local].astype(float)

    preds_lower = lower_huber.predict(df_transformed)
    preds_mid = mid_huber.predict(df_transformed)
    preds_upper = upper_huber.predict(df_transformed)

    df["Lower"] = preds_lower
    df["Upper"] = preds_upper
    df["FVC"] = preds_mid

    iqr = np.abs(preds_upper - preds_lower)
    b = iqr / (2.0 * np.log(2.0))
    sigma = np.sqrt(2.0) * b

    sigma = np.nan_to_num(sigma, nan=285.0, posinf=285.0, neginf=285.0)
    sigma = np.clip(sigma, 70.0, 1000.0)

    df["Confidence"] = sigma
    return df




## === cell 20
Patient = "Albert"
Week = -4  # Number of weeks after CT Scan, can be negative
FVC = 3000
Percent = 78
Age = 69
Sex = "Female"
SmokingStatus = "Never smoked"

df_demo = huber_predict(
    lower_huber,
    mid_huber,
    upper_huber,
    Patient,
    Week,
    FVC,
    Percent,
    Age,
    Sex,
    SmokingStatus,
)
df_demo.head()



## === cell 21
base_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
df_test = pd.read_csv(base_path + "test.csv")
df_test.tail(20)



## === cell 22
pred_list = []
for _, row in df_test.iterrows():
    df_interim = huber_predict(
        lower_huber,
        mid_huber,
        upper_huber,
        row["Patient"],
        int(row["Weeks"]),
        float(row["FVC"]),
        float(row["Percent"]),
        float(row["Age"]),
        row["Sex"],
        row["SmokingStatus"],
    )
    pred_list.append(df_interim)

df_predict = pd.concat(pred_list, ignore_index=True)



## === cell 23
df_predict["Patient_Week"] = (
    df_predict["Patient"] + "_" + df_predict["Weeks"].astype(int).astype(str)
)

sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_predict_keyed = df_predict[["Patient_Week", "FVC", "Confidence"]].copy()
df_submission = sample_sub[["Patient_Week"]].merge(
    df_predict_keyed, on="Patient_Week", how="left"
)



## === cell 24
df_submission["FVC"] = (
    df_submission["FVC"].fillna(df_submission["FVC"].median()).astype(float)
)

df_submission["Confidence"] = df_submission["Confidence"].fillna(285.0).astype(float)

df_submission["Confidence"] = np.maximum(df_submission["Confidence"].values, 70.0)

df_submission.to_csv("/kaggle/working/submission.csv", index=False)
print(df_submission.head())
print("Wrote /kaggle/working/submission.csv with shape:", df_submission.shape)

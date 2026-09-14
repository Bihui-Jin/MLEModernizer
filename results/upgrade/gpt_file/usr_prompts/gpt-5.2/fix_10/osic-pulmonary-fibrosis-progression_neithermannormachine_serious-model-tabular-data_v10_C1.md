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

-6.8685

# 6. Current score

-11.16126

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.00278) has done: 'The immediate blocker is NaNs appearing in the engineered/test features (mainly from missing FirstFVC/FirstWeek/Height after merging test patients with train-derived baselines), which makes `LinearRegression.predict` crash and prevents any submission from being written. I fix this minimally by ensuring `MyFeatureEngineerer.transform()` correctly merges patient-level baseline features for *any* input and by filling remaining NaNs with training-set medians (score-neutral/stabilizing, not a modeling change). Then I ensure the transformed train/test matrices have aligned columns and that we reliably build `sub_df` and merge it into `sample_submission.csv` to output a valid `submission.csv`. Core model (linear regression), feature definitions, and confidence-selection logic remain unchanged.'
- What this solution (achieved -9.81597) has done: 'We make a small, metric-relevant adjustment to the prediction post-processing: clamp the predicted `FVC` to a plausible range derived from the training data (1st–99th percentiles). This tends to reduce large absolute errors (which, although capped at 1000, still hurt the Laplace term) without changing the model, features, or training loop. We also compute a patient-specific confidence (σ) from the model’s training residual dispersion, clipped at the competition minimum of 70, instead of using a single global constant—this better matches the evaluation’s uncertainty handling and typically improves the score. All changes keep the core logic (linear regression + existing features) intact and still produce a valid `submission.csv`.'
- What this solution (achieved -8.64583) has done: 'Your current score (-9.81597) is worse than the target (-6.8685), so we should cautiously improve performance with minimal, metric-aligned changes. The biggest low-risk win here is to make the confidence prediction reflect *time uncertainty*: keep your existing patient-specific residual sigma, but increase it smoothly as `|WeeksPassed|` grows (predictions farther from baseline should be less confident), which usually improves the Laplace log-likelihood without changing the model or features. I also fix a subtle data bug in the feature pipeline: `WeeksPassed` for train should be computed from the same patient-level baseline (`patient_base_`) used at test time, avoiding leakage/inconsistency caused by `FirstWeek` being computed within a modified frame. Finally, I keep your existing FVC clipping, but make it slightly less aggressive (0.5–99.5 percentile) to reduce bias while still preventing extreme errors.'
- What this solution (achieved -9.00247) has done: 'You’re currently well below the target (-8.6458 vs -6.8685; higher is better), so the smallest metric-aligned change is to tune only the confidence (sigma) post-processing while keeping the same linear regression + features. The Laplace metric often improves when sigma grows with time distance from baseline, but your current growth `|weeks|/50` is likely too aggressive and can over-penalize via `-log(sigma)`; I soften it and add a small constant floor so near-baseline weeks aren’t overconfident. I also compute `weeks_passed` directly from `sub_df["Weeks"] - sub_df["FirstWeek"]` (already present from `eng.transform`) to avoid any index alignment risk with `new_df`. Everything else (feature engineering, scaling, model, FVC clipping, submission merge) stays the same and still writes `submission.csv`.'
- What this solution (achieved -8.99821) has done: 'We need to move the score up toward the target, so I’m keeping your linear regression + features intact and only adjusting the post-processing that directly affects the Laplace metric. The biggest low-risk lever here is confidence (sigma): your current time scaling is likely still too large on average (hurting via `-log(sigma)`), so I soften it and make it smoothly patient-aware by anchoring growth to the distribution of `WeeksPassed` seen in training. I also add a tiny, metric-aligned safeguard to prevent unrealistically small patient sigmas (which can severely penalize delta) by blending patient sigma with the global sigma (shrinkage), without changing the model. Everything else (feature engineering, training, clipping, submission merge) remains the same and still produces `submission.csv`.'
- What this solution (achieved -9.62765) has done: 'I keep your linear regression model, features, and training flow unchanged, and only tune the confidence post-processing because that is the most metric-direct lever and your current score (-8.998) is still worse than the target (-6.8685). Specifically, I reduce the time-based sigma inflation (it’s likely too large on average, hurting via the `-log(sigma)` term) and replace it with a gentler, saturating growth so far-future weeks don’t get over-penalized. I also make the patient-sigma computation more robust by using a median absolute deviation (MAD)-based estimate (still derived from your model residuals; no leakage), which reduces instability from small sample sizes and outliers while preserving your patient-specific approach. Everything still runs end-to-end and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved -11.16126) has done: 'Your current score (-9.62765) is well below the target (-6.8685), so we should improve it with the smallest metric-aligned change: tune only the `Confidence` post-processing while keeping the same linear regression and features. The Laplace metric often improves when `sigma` is closer to the *typical absolute error* and not overly inflated (since `-log(sigma)` is a direct penalty), so I replace the MAD-only patient sigma with a simple, robust patient sigma that blends per-patient MAE with a global MAE (shrinkage), and I reduce the time-based inflation to a gentler, symmetric factor. I also ensure `Confidence` is clipped to a reasonable upper bound derived from training residuals (prevents over-penalizing via `-log(sigma)`), while still respecting the competition’s minimum clip at 70. Everything else—including feature engineering, the trained model, FVC clipping, and submission formatting—remains unchanged and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
pass



## === cell 2
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df




## === cell 3
def feature_engineer(data):
    """
    Feature engineering for train/test DataFrames.
    """
    df = data.copy()

    df["FirstWeek"] = df.groupby("Patient")["Weeks"].transform("min")

    first_fvc = (
        df.loc[df["Weeks"] == df["FirstWeek"]][["Patient", "FVC"]]
        .groupby("Patient")
        .first()
        .reset_index()
        .rename(columns={"FVC": "FirstFVC"})
    )
    df = df.merge(first_fvc, on="Patient", how="left")

    df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

    def calculate_height(row):
        if row["Sex"] == "Male":
            return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
        else:
            return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

    df["Height"] = df.apply(calculate_height, axis=1)
    return df


feature_engineer(train_df)



## === cell 4
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    Stores patient-level baseline-derived features at fit time,
    and can be used to transform a modified frame with extra weeks.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Can only use this estimator on Pandas DataFrame")

        df = feature_engineer(X)
        self.df_ = df

        self.patient_base_ = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient", as_index=False)
            .first()[["Patient", "FirstWeek", "FirstFVC", "Height"]]
        )
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Can only use this estimator on Pandas DataFrame")

        df = X.copy()

        for col in ["Patient", "Weeks", "Age", "Sex", "SmokingStatus", "Percent"]:
            if col not in df.columns:
                df[col] = np.nan

        df = df.merge(self.patient_base_, on="Patient", how="left")
        df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

        def calculate_height(row):
            if (
                pd.isna(row.get("FirstFVC"))
                or pd.isna(row.get("Age"))
                or pd.isna(row.get("Sex"))
            ):
                return np.nan
            if row["Sex"] == "Male":
                return row["FirstFVC"] / (27.63 - 0.112 * row["Age"])
            else:
                return row["FirstFVC"] / (21.78 - 0.101 * row["Age"])

        if "Height" not in df.columns or df["Height"].isna().any():
            df["Height"] = df.apply(calculate_height, axis=1)

        return df




## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom min-max scaler with fixed min/max passed as parameters.
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.min_val) / (self.max_val - self.min_val)




## === cell 6
def transformed_col_names(col_trans, input_features):
    """
    Return output feature names from a fitted ColumnTransformer.
    Works even when some transformers don't implement get_feature_names_out.
    """
    try:
        return list(col_trans.get_feature_names_out(input_features=input_features))
    except Exception:
        names = []
        for name, trans, cols in col_trans.transformers_:
            if name == "remainder" and trans == "drop":
                continue

            if trans == "passthrough":
                if isinstance(cols, slice):
                    cols = list(input_features[cols])
                elif isinstance(cols, (np.ndarray, list, tuple, pd.Index)):
                    cols = list(cols)
                else:
                    cols = [cols]
                names.extend([f"{name}__{c}" for c in cols])
                continue

            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(cols)
                except Exception:
                    fn = trans.get_feature_names_out()
                fn = [f"{name}__{f}" for f in fn]
                names.extend(fn)
            else:
                if isinstance(cols, slice):
                    cols = list(input_features[cols])
                elif isinstance(cols, (np.ndarray, list, tuple, pd.Index)):
                    cols = list(cols)
                else:
                    cols = [cols]
                names.extend([f"{name}__{c}" for c in cols])
        return names




## === cell 7
from sklearn_pandas import DataFrameMapper



## === cell 8
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

passthru_features = ["Patient", "FVC"]
onehot_features = ["Sex", "SmokingStatus"]
hundred_features = ["Percent", "Age"]
minmax_features = ["FirstFVC", "FirstWeek", "WeeksPassed", "Height"]

oh_enc = OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore")
hundred_minmax = ParamMinMaxScaler()
week_minmax = ParamMinMaxScaler(min_val=-12, max_val=133)
minmax = MinMaxScaler()

col_trans = ColumnTransformer(
    transformers=[
        ("original", "passthrough", passthru_features),
        ("week_minmax", week_minmax, ["Weeks"]),
        ("hundred_minmax", hundred_minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="drop",
    sparse_threshold=0,
)



## === cell 9
pass



## === cell 10
eng_train = MyFeatureEngineerer().fit(train_df)
train_fe = eng_train.transform(train_df)

num_cols = train_fe.select_dtypes(include=[np.number]).columns
train_fe[num_cols] = train_fe[num_cols].fillna(
    train_fe[num_cols].median(numeric_only=True)
)

new_arr = col_trans.fit_transform(train_fe)
feature_names = transformed_col_names(col_trans, input_features=train_fe.columns)

if len(feature_names) != new_arr.shape[1]:
    feature_names = [f"f_{i}" for i in range(new_arr.shape[1])]

train_df = pd.DataFrame(new_arr, columns=feature_names)

if "original__Patient" in train_df.columns:
    train_df["original__Patient"] = train_df["original__Patient"].astype(str)
if "original__FVC" in train_df.columns:
    train_df["original__FVC"] = pd.to_numeric(
        train_df["original__FVC"], errors="coerce"
    )

train_df




## === cell 11
def laplace_metric_np(y_true, y_pred_fvc, sigma):
    """
    Returns mean metric (higher is better) for the OSIC competition.
    """
    sigma_clipped = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred_fvc), 1000.0)
    sq2 = np.sqrt(2.0)
    metric = -(sq2 * delta) / sigma_clipped - np.log(sq2 * sigma_clipped)
    return float(np.mean(metric))


def temp_laplace_scorer(estimator, X, y_true, confidence):
    """
    sklearn scorer signature: (estimator, X, y) -> score
    """
    y_pred = estimator.predict(X)
    return laplace_metric_np(
        np.asarray(y_true, dtype=np.float32),
        np.asarray(y_pred, dtype=np.float32),
        sigma=float(confidence),
    )




## === cell 12
from sklearn.linear_model import LinearRegression


def make_model():
    return LinearRegression()




## === cell 13
train_df



## === cell 14
model = make_model()

drop_features = [
    "original__Patient",
    "original__FVC",
    "week_minmax__Weeks",
    "hundred_minmax__Percent",
]

X_train = train_df.drop(columns=drop_features, errors="ignore")
y_train = train_df["original__FVC"].astype(float)

model.fit(X_train, y_train)
X_train



## === cell 15
from sklearn.model_selection import RandomizedSearchCV

pass



## === cell 16
pass



## === cell 17
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import make_scorer

NFOLDS = 6
gkf = GroupKFold(n_splits=NFOLDS)

groups = train_df["original__Patient"].values

conf = np.arange(100, 401, 5)
conf_df = pd.DataFrame(index=conf, columns=["mean score", "std score"], dtype=float)
conf_df.index.name = "Confidence"

for c in conf:
    scorer = make_scorer(
        lambda yt, yp: laplace_metric_np(
            np.asarray(yt, dtype=np.float32),
            np.asarray(yp, dtype=np.float32),
            sigma=float(c),
        ),
        greater_is_better=True,
    )
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=gkf, groups=groups, scoring=scorer
    )
    conf_df.loc[c, "mean score"] = float(np.mean(cv_scores))
    conf_df.loc[c, "std score"] = float(np.std(cv_scores))

num_std = 2.3
conf_df["worst case"] = conf_df["mean score"] - num_std * conf_df["std score"]
conf_df



## === cell 18
best = conf_df.nlargest(10, columns=["worst case"], keep="all")
best_fmt = best.copy()
for col in best_fmt.columns:
    best_fmt[col] = best_fmt[col].map(lambda x: f"{x:,.6f}")
best_fmt



## === cell 19
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(X_train.columns.values, model.coef_)
plt.xticks(rotation=70)
plt.tight_layout()



## === cell 20
pred_train = model.predict(X_train)
pred_train[:10]



## === cell 21
import random

p = random.choice(train_df["original__Patient"].unique())
mask = train_df["original__Patient"] == p

temp_df = pd.DataFrame(
    {
        "Weeks": (train_fe.loc[train_fe["Patient"] == p, "Weeks"].values),
        "FVC": (train_fe.loc[train_fe["Patient"] == p, "FVC"].values),
        "FVC_pred": pd.Series(pred_train, index=train_df.index).loc[mask].values,
    }
)

temp_df = temp_df.sort_values("Weeks")
ax = temp_df.plot(
    x="Weeks", y=["FVC", "FVC_pred"], title=f"Patient {p}", figsize=(7, 4)
)
plt.tight_layout()



## === cell 22
from sklearn.pipeline import Pipeline

pass



## === cell 23
input_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
input_df



## === cell 24
eng = MyFeatureEngineerer().fit(
    pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
)

all_weeks = np.arange(-12, 134, dtype=int)
patients = input_df["Patient"].unique()

patient_weeks = pd.DataFrame(
    {
        "Patient": np.repeat(patients, len(all_weeks)),
        "Weeks": np.tile(all_weeks, len(patients)),
    }
)

input_df2 = input_df.drop(["FVC", "Weeks"], axis=1)
temp_df = patient_weeks.merge(input_df2, on="Patient", how="left")

new_df = eng.transform(temp_df)
new_df



## === cell 25
new_df = new_df.copy()
new_df["FVC"] = 0.0

num_cols_test = new_df.select_dtypes(include=[np.number]).columns
train_medians = train_fe[num_cols].median(numeric_only=True)
new_df[num_cols_test] = new_df[num_cols_test].fillna(
    train_medians.reindex(num_cols_test)
)

test_arr = col_trans.transform(new_df)
test_feature_names = transformed_col_names(col_trans, input_features=new_df.columns)

if len(test_feature_names) != test_arr.shape[1]:
    test_feature_names = [f"f_{i}" for i in range(test_arr.shape[1])]

test_df = pd.DataFrame(test_arr, columns=test_feature_names)

if "original__Patient" in test_df.columns:
    test_df["original__Patient"] = test_df["original__Patient"].astype(str)

test_df.head()



## === cell 26
X_test = test_df.drop(columns=drop_features, errors="ignore")

X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
X_test.head()



## === cell 27
pred = model.predict(X_test)

fvc_lo, fvc_hi = np.percentile(y_train.values.astype(float), [0.5, 99.5])
pred = np.clip(pred.astype(float), fvc_lo, fvc_hi)

pred[:10]



## === cell 28
sub_df = patient_weeks.copy()
sub_df["FVC"] = pred.astype(float)
sub_df.head()



## === cell 29
plt.figure(figsize=(17, 10))
for i, (patient, frame) in enumerate(sub_df.groupby("Patient")):
    ax = plt.subplot(3, 6, i + 1)
    frame.sort_values("Weeks")[["Weeks", "FVC"]].plot(
        x="Weeks", y="FVC", title=patient, ax=ax, legend=False
    )
plt.tight_layout()



## === cell 30
sub_df["Patient_Week"] = sub_df["Patient"] + "_" + sub_df["Weeks"].astype(str)

resid = y_train.values.astype(float) - pred_train.astype(float)
resid_df = pd.DataFrame(
    {"Patient": train_df["original__Patient"].astype(str).values, "resid": resid}
)

patient_mae = resid_df.groupby("Patient")["resid"].apply(
    lambda x: float(np.nanmean(np.abs(x.values.astype(float))))
)

global_mae = float(np.nanmean(np.abs(resid.astype(float))))
if not np.isfinite(global_mae) or global_mae <= 0:
    global_mae = 200.0

sigma_for_sub = sub_df["Patient"].map(patient_mae).astype(float).fillna(global_mae)

shrink_w = 0.65
sigma_for_sub = (1.0 - shrink_w) * sigma_for_sub + shrink_w * global_mae

weeks_passed = (
    new_df["Weeks"].astype(float) - new_df["FirstWeek"].astype(float)
).to_numpy()
weeks_passed = np.nan_to_num(weeks_passed, nan=0.0)

wp_train = train_fe["WeeksPassed"].astype(float).to_numpy()
wp_train = wp_train[np.isfinite(wp_train)]
wp_scale = float(np.percentile(np.abs(wp_train), 75)) if wp_train.size else 50.0
if not np.isfinite(wp_scale) or wp_scale <= 1e-6:
    wp_scale = 50.0

a = 0.04  # smaller than before
time_factor = 1.00 + a * (np.abs(weeks_passed) / (wp_scale + np.abs(weeks_passed)))

sigma_for_sub = sigma_for_sub.values.astype(float) * time_factor

sigma_cap = float(np.nanpercentile(np.abs(resid.astype(float)), 95) * 3.0)
if not np.isfinite(sigma_cap) or sigma_cap <= 70.0:
    sigma_cap = 500.0
sigma_for_sub = np.clip(sigma_for_sub.astype(float), 70.0, sigma_cap)

sub_df["Confidence"] = sigma_for_sub.astype(float)

sub_df[["Patient_Week", "FVC", "Confidence"]].head()



## === cell 31
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

out = sample_sub[["Patient_Week"]].merge(
    sub_df[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

out["FVC"] = out["FVC"].fillna(out["FVC"].median())
out["Confidence"] = out["Confidence"].fillna(np.maximum(global_mae, 70.0))

out.to_csv("submission.csv", index=False)
out.head()

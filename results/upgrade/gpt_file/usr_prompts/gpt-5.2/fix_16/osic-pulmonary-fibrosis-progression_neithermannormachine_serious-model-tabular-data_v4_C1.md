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

-6.8945

# 6. Current score

-9.88354

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.20925) has done: 'I fix the NaN issue that crashes `model.predict()` by adding a minimal `SimpleImputer` on the feature matrix (train and test) without changing the model or feature engineering logic. I also make the sample-submission merge robust by ensuring `sub_df` is always built with the required `Patient_Week`, `FVC`, and `Confidence` columns before selecting them. Finally, I keep the constant confidence approach but make it deterministic and ensure the script always writes `submission.csv` with the exact required columns and row order.'
- What this solution (achieved -9.84904) has done: 'You’re far below the target (-11.20925 vs -6.8945; higher is better), and the biggest low-risk gain without changing the model is to align the prediction generation with the competition’s “predict from the baseline visit” setup. I keep your exact linear regression + engineered features, but (1) train on “WeeksPassed” relative to each patient’s baseline week (closest to 0) rather than earliest week, and (2) generate test predictions by expanding weeks relative to each test patient’s provided baseline `Weeks` (not a fixed -12..133 for everyone). I also compute the constant `Confidence` using a CV approximation of the competition metric (still constant per your approach) instead of MAE, which usually moves the score upward because the metric rewards well-chosen sigma. All paths and submission writing stay the same and it still produce `submission.csv`.'
- What this solution (achieved -9.84904) has done: 'I make two minimal, score-relevant fixes without changing your model or feature set: (1) fit the `Weeks` min-max scaler on the training data (instead of using the hard-coded -12..133 range), which better matches the actual scale your linear model saw during training, and (2) generate predictions only for the weeks that actually appear in `sample_submission.csv` (the evaluation only uses those), avoiding extrapolation to many irrelevant weeks that can hurt alignment and merges. Everything else (feature engineering logic, linear regression, constant-confidence selection via CV metric, and submission writing) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -9.84904) has done: 'I keep your linear regression + engineered features exactly as-is, but make two minimal, score-relevant fixes to better match the competition’s evaluation framing. First, I remove the accidental “future information” leak in `FirstFVC` for training by defining `FirstFVC` strictly from the patient’s baseline visit (week closest to 0) rather than from each patient’s earliest week; this makes train-time features consistent with what’s available at test time and typically improves generalization toward your target. Second, I cap the predicted `FVC` to a reasonable physiological range (using training quantiles) to reduce extreme errors that hurt the Laplace score via the `Delta` term, without changing the model or loss. Everything else (week expansion via `sample_submission`, constant `Confidence` chosen by CV metric, and `submission.csv` writing) stays the same.'
- What this solution (achieved -9.84904) has done: 'I fix the root cause of the crash: `train_exp` becomes empty because you expand using `sample_submission` weeks but then inner-join to `train_df` on exact (Patient, Weeks), yielding zero rows; this makes `MinMaxScaler` fail. To keep the same core linear-regression approach and engineered features, I instead train on the original `train_df` (all available visits) while still using baseline-derived features (`FirstWeek/FirstFVC/WeeksPassed/Height`) exactly as your feature engineering intends. Then I fit the `ColumnTransformer`, imputer, and model on that non-empty training matrix, keep the same GroupKFold-based constant Confidence selection, and run inference for exactly the `sample_submission` patient-weeks before writing a valid `submission.csv` with the required columns and order.'
- What this solution (achieved -9.89305) has done: 'I keep your exact linear-regression pipeline and engineered features, but make two small, score-relevant adjustments that typically improve the Laplace log-likelihood without changing training semantics. First, I compute the out-of-fold residual distribution and set the constant `Confidence` to a robust (near-optimal) sigma estimate based on median absolute error (MAE→Laplace σ), then still refine it by your existing sigma-grid search (so it can only help, not hurt). Second, I replace the global FVC clipping with a patient-anchored clamp around each test patient’s baseline FVC (available in test.csv), which reduces extreme deltas for extrapolated weeks while preserving your linear predictions. The submission generation, row order, and required columns remain identical and it still write `submission.csv` end-to-end.'
- What this solution (achieved -9.89346) has done: 'We’re currently below the target (−9.893 vs −6.8945; higher is better), so the safest way to move upward without changing your model/feature core is to improve the *Confidence* calibration to match the Laplace metric better. I keep your linear regression, features, training, and prediction logic unchanged, but adjust constant `Confidence` selection to be more robust by (a) calibrating sigma using *trimmed* out-of-fold absolute errors (reduces the influence of rare outliers) and (b) broadening/refining the sigma search grid around that robust estimate. This directly targets the metric’s sigma term and is minimal-risk compared with altering the regressor. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved -9.88354) has done: 'We’re currently well below the target (−9.893 vs −6.8945; higher is better), so the smallest likely gain without touching your model/features is to better calibrate the constant `Confidence` to the competition’s Laplace score. I keep your exact linear-regression training and prediction flow, but (1) compute out-of-fold residuals and fit a single global σ by directly maximizing the Laplace metric on those residuals (instead of relying on a trimmed heuristic), and (2) optionally compute σ separately per “WeeksPassed distance bucket” and then collapse to a single constant via the same Laplace objective (still constant in the submission, but informed by heteroscedasticity). This only changes the confidence value while preserving evaluation semantics, and should move the score upward toward your target band. The script still run end-to-end and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved -9.88354) has done: 'I keep your exact feature engineering and linear regression training/inference pipeline unchanged, and focus on a small but metric-relevant improvement: better calibration of the *constant* `Confidence` value (σ) used in the Laplace log-likelihood. Specifically, instead of selecting σ on raw out-of-fold predictions (which can be slightly biased because the final model is trained on all data), I use a tiny second-stage calibration that refits the model on all training data and then selects σ by maximizing the Laplace metric on **in-sample residuals**, which typically shifts σ downward/upward to better match the final model’s error scale (often improving score without changing FVC predictions). To avoid overreacting, I combine OOF-optimized σ and in-sample-optimized σ via a simple average, then run the same metric-based refinement around that value. All I/O paths, submission format, and your FVC post-clipping remain exactly as-is.'
- What this solution (achieved -9.88354) has done: 'I keep your linear-regression model and feature engineering unchanged, and only adjust the constant Confidence calibration because your current score is far below the target and this is the smallest, most metric-aligned lever. Specifically, I (1) choose sigma by directly maximizing the Laplace metric on out-of-fold residuals using a fast scalar optimization (golden-section search) instead of an exhaustive 70..1200 grid, and (2) add a tiny per-patient adjustment check to ensure the chosen sigma isn’t distorted by patients with many rows (use per-patient mean absolute residuals for calibration, then blend with the row-wise optimum). This keeps “constant confidence in the submission” semantics intact while typically improving the public score without touching FVC predictions. The script still runs end-to-end on the same paths and writes a valid `submission.csv` with the exact required columns and order.'
- What this solution (achieved -9.88354) has done: 'I keep your linear regression model, feature engineering, and prediction generation unchanged, and only adjust the *constant Confidence calibration* because that’s the smallest lever directly optimized by the OSIC metric. Your current confidence selection uses in-sample residuals, which can overfit and push sigma away from what the final test error scale looks like; I instead calibrate sigma from out-of-fold residuals using a per-patient aggregation (so patients with many visits don’t dominate) and then do a narrow metric-based refinement around that value. This should move the public score upward toward the target without changing FVC predictions (only the Confidence column). The script still run end-to-end and write a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved -9.88354) has done: 'I keep your linear-regression model, feature engineering, and week-expansion logic exactly the same, and only adjust the constant `Confidence` calibration because it directly impacts the Laplace log-likelihood score without changing FVC predictions. Your current sigma selection uses residuals from all rows (and a patient-mean blend), which can still be skewed by patients with many visits and by systematic week-dependent error; I switch to selecting sigma by maximizing the metric on **per-patient residual aggregates** (one value per patient) to better match how errors generalize to unseen patients. I also compute sigma on a robustified residual (patient median absolute residual) and then do the same narrow refinement you already do, keeping everything deterministic. This is a minimal, score-relevant change intended to move your score upward toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
train_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
train_df.head()




## === cell 2
def feature_engineer(data):
    """
    Feature engineer any df, train or test.

    Uses each patient's baseline week (closest to 0), matching the competition framing
    where test provides the baseline visit.

    IMPORTANT (score-relevant, minimal, preserved):
    - FirstWeek = week closest to 0
    - FirstFVC = FVC at FirstWeek only (prevents leakage)
    - WeeksPassed = Weeks - FirstWeek
    - Height computed from FirstFVC and Age/Sex
    """
    df = data.copy()

    df["_abs_week"] = df["Weeks"].abs()
    base_weeks = (
        df.sort_values(["Patient", "_abs_week", "Weeks"])
        .groupby("Patient")["Weeks"]
        .first()
        .reset_index()
        .rename(columns={"Weeks": "FirstWeek"})
    )
    df = df.drop(columns=["_abs_week"]).merge(base_weeks, on="Patient", how="left")

    first_fvc = (
        df.loc[df["Weeks"] == df["FirstWeek"], ["Patient", "FVC"]]
        .groupby("Patient")
        .first()
        .reset_index()
        .rename(columns={"FVC": "FirstFVC"})
    )
    df = df.merge(first_fvc, on="Patient", how="left")

    df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]

    def calculate_height(row):
        if row["Sex"] == "Male":
            denom = 27.63 - 0.112 * row["Age"]
        else:
            denom = 21.78 - 0.101 * row["Age"]
        if denom == 0 or pd.isna(denom) or pd.isna(row["FirstFVC"]):
            return np.nan
        return row["FirstFVC"] / denom

    df["Height"] = df.apply(calculate_height, axis=1)
    return df


feature_engineer(train_df).head()



## === cell 3
from sklearn.base import BaseEstimator, TransformerMixin


class MyFeatureEngineerer(BaseEstimator, TransformerMixin):
    """
    Fit stores patient-level engineered info; transform can be applied after expanding weeks.
    """

    def __init__(self):
        self.df_ = None

    def fit(self, X, y=None):
        self.df_ = feature_engineer(X)
        return self

    def transform(self, X):
        """
        If X has been modified with additional weeks, merge stored patient-level columns back.
        """
        if self.df_ is None:
            raise RuntimeError("MyFeatureEngineerer must be fit before transform().")

        if len(X) != len(self.df_):
            base_cols = ["Patient", "FirstWeek", "FirstFVC", "Height"]
            keep = self.df_[base_cols].drop_duplicates("Patient")
            df = X.merge(keep, on="Patient", how="left")
            df["WeeksPassed"] = df["Weeks"] - df["FirstWeek"]
        else:
            df = self.df_.copy()
        return df




## === cell 4
def transformed_col_names(col_trans, input_features):
    """
    Robust helper to recover column names after ColumnTransformer.
    Works across sklearn versions by preferring get_feature_names_out().
    """
    new_colnames = []
    for name, t, cols in col_trans.transformers_:
        if name == "remainder" and t == "drop":
            continue

        if cols is None:
            cols_list = list(input_features)
        elif isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            cols_list = list(cols)
        else:
            cols_list = list(pd.Index(input_features)[cols])

        if t == "passthrough":
            new_colnames.extend(cols_list)
            continue
        if t == "drop":
            continue

        if hasattr(t, "get_feature_names_out"):
            try:
                feat = t.get_feature_names_out(cols_list)
                new_colnames.extend(list(feat))
                continue
            except Exception:
                pass

        if hasattr(t, "get_feature_names"):
            try:
                feat = t.get_feature_names(cols_list)
                new_colnames.extend(list(feat))
                continue
            except Exception:
                pass

        new_colnames.extend(cols_list)

    return new_colnames




## === cell 5
from sklearn.base import BaseEstimator, TransformerMixin


class ParamMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Custom minmax scaler with provided min/max.
    """

    def __init__(self, min_val=0, max_val=100):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        denom = self.max_val - self.min_val
        if denom == 0:
            return np.zeros_like(X, dtype=np.float64)
        return (X - self.min_val) / denom




## === cell 6
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

passthru_features = ["Patient", "FVC"]
onehot_features = ["Sex", "SmokingStatus"]
hundred_features = ["Percent", "Age"]
minmax_features = ["FirstFVC", "FirstWeek", "WeeksPassed", "Height"]

try:
    oh_enc = OneHotEncoder(
        sparse_output=False, drop="if_binary", handle_unknown="ignore"
    )
except TypeError:
    oh_enc = OneHotEncoder(sparse=False, drop="if_binary", handle_unknown="ignore")

hundred_minmax = ParamMinMaxScaler()
week_minmax = MinMaxScaler()
minmax = MinMaxScaler()

col_trans = ColumnTransformer(
    [
        ("original", "passthrough", passthru_features),
        ("week_minmax", week_minmax, ["Weeks"]),
        ("hundred_minmax", hundred_minmax, hundred_features),
        ("minmax", minmax, minmax_features),
        ("onehot", oh_enc, onehot_features),
    ],
    remainder="drop",
    sparse_threshold=0,
)



## === cell 7
sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
tmp = sample_sub["Patient_Week"].str.split("_", n=1, expand=True)
sub_patients = tmp[0].values
sub_weeks = tmp[1].astype(int).values
patient_weeks = pd.DataFrame({"Patient": sub_patients, "Weeks": sub_weeks})

eng_train = MyFeatureEngineerer()
eng_train.fit(train_df)
train_fe_full = eng_train.transform(train_df)

new_arr = col_trans.fit_transform(train_fe_full)
colnames = transformed_col_names(col_trans, input_features=train_fe_full.columns)

train_df_model = pd.DataFrame(new_arr, columns=colnames)
print("train_df_model shape:", train_df_model.shape)
train_df_model.head()



## === cell 8
pass




## === cell 9
def laplace_log_score(**kwargs):
    raise NotImplementedError("TensorFlow loss not used in this sklearn solution.")


def pinball_qloss(quantiles):
    raise NotImplementedError("TensorFlow loss not used in this sklearn solution.")


def weighted_loss(weights, loss_functions):
    raise NotImplementedError("TensorFlow loss not used in this sklearn solution.")


def mloss():
    raise NotImplementedError("TensorFlow loss not used in this sklearn solution.")




## === cell 10
from sklearn.linear_model import LinearRegression


def make_model():
    """
    Simple tabular model; core logic preserved: linear regression on engineered features.
    """
    model = LinearRegression()
    return model




## === cell 11
train_df_model.head()



## === cell 12
from sklearn.impute import SimpleImputer

model = make_model()

drop_features = ["Patient", "FVC", "Weeks", "Percent"]

X_train = train_df_model.drop(
    [c for c in drop_features if c in train_df_model.columns], axis=1
)
y_train = train_df_model["FVC"].astype(float)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)

model.fit(X_train_imp, y_train)

pd.DataFrame(X_train_imp, columns=X_train.columns).head()



## === cell 13
from sklearn.model_selection import RandomizedSearchCV  # kept (unused)



## === cell 14
pass



## === cell 15
from sklearn.model_selection import GroupKFold
from sklearn.base import clone


def osic_metric(y_true, y_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


def best_constant_sigma_from_residuals(abs_err):
    abs_err = np.asarray(abs_err, dtype=np.float64)
    abs_err = abs_err[np.isfinite(abs_err)]
    if abs_err.size == 0:
        return 70.0, float(osic_metric(np.array([0.0]), np.array([0.0]), 70.0).mean())

    def f(s):
        s = float(s)
        return float(osic_metric(abs_err, np.zeros_like(abs_err), s).mean())

    a, b = 70.0, 2000.0
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, fd = f(c), f(d)

    for _ in range(60):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)

    s_best = (a + b) / 2.0

    lo = max(70.0, np.floor(s_best - 20.0))
    hi = min(2000.0, np.ceil(s_best + 20.0))
    grid = np.arange(lo, hi + 1e-9, 1.0)
    vals = np.array([f(s) for s in grid], dtype=np.float64)
    s_ref = float(grid[int(np.argmax(vals))])
    return s_ref, float(np.max(vals))


NFOLDS = 6
gkf = GroupKFold(n_splits=NFOLDS)
groups = train_df_model["Patient"].values

oof_pred = np.zeros(len(y_train), dtype=float)
oof_weeks_passed = np.zeros(len(y_train), dtype=float)

weeks_passed_series = train_fe_full["WeeksPassed"].astype(float).values

for fold, (tr_idx, va_idx) in enumerate(
    gkf.split(X_train_imp, y_train, groups=groups), 1
):
    m = clone(model)
    m.fit(X_train_imp[tr_idx], y_train.iloc[tr_idx])
    oof_pred[va_idx] = m.predict(X_train_imp[va_idx])
    oof_weeks_passed[va_idx] = weeks_passed_series[va_idx]

abs_err_oof = np.abs(y_train.values - oof_pred)

oof_err_df = pd.DataFrame(
    {"Patient": groups, "abs_err": abs_err_oof.astype(np.float64)}
)
abs_by_patient_mean = oof_err_df.groupby("Patient")["abs_err"].mean().values
abs_by_patient_median = oof_err_df.groupby("Patient")["abs_err"].median().values

sigma_pat_mean, metric_pat_mean = best_constant_sigma_from_residuals(
    abs_by_patient_mean
)
sigma_pat_med, metric_pat_med = best_constant_sigma_from_residuals(
    abs_by_patient_median
)

sigma_row, metric_row = best_constant_sigma_from_residuals(abs_err_oof)

sigma_seed = float(0.55 * sigma_pat_med + 0.30 * sigma_pat_mean + 0.15 * sigma_row)

lo = max(70.0, sigma_seed - 250.0)
hi = min(2000.0, sigma_seed + 250.0)
grid_ref = np.arange(lo, hi + 1e-9, 1.0)
scores_ref = np.array(
    [osic_metric(abs_err_oof, np.zeros_like(abs_err_oof), s).mean() for s in grid_ref],
    dtype=np.float64,
)
sigma_global = float(grid_ref[int(np.argmax(scores_ref))])
best_cv_metric = float(np.max(scores_ref))

abs_w = np.abs(oof_weeks_passed)
q1, q2 = np.quantile(abs_w, [0.33, 0.66])
bins = np.array([-np.inf, q1, q2, np.inf])
bucket = np.digitize(abs_w, bins) - 1  # 0,1,2

bucket_sigmas = []
bucket_weights = []
for b in [0, 1, 2]:
    idx = np.where(bucket == b)[0]
    if idx.size < 20:
        continue
    sb, _ = best_constant_sigma_from_residuals(abs_err_oof[idx])
    bucket_sigmas.append(sb)
    bucket_weights.append(float(idx.size))

if len(bucket_sigmas) >= 2:
    sigma_bucket_avg = float(np.average(bucket_sigmas, weights=bucket_weights))
    score_global = osic_metric(y_train.values, oof_pred, sigma_global).mean()
    score_bucket = osic_metric(y_train.values, oof_pred, sigma_bucket_avg).mean()
    if score_bucket > score_global:
        sigma_global = sigma_bucket_avg
        best_cv_metric = float(score_bucket)

confidence = float(max(70.0, sigma_global))

print(
    "Sigma from OOF residuals (row-wise):",
    float(sigma_row),
    "OOF metric:",
    float(metric_row),
)
print(
    "Sigma from per-patient MEAN OOF residuals:",
    float(sigma_pat_mean),
    "Pat-OOF metric:",
    float(metric_pat_mean),
)
print(
    "Sigma from per-patient MEDIAN OOF residuals:",
    float(sigma_pat_med),
    "Pat-OOF metric:",
    float(metric_pat_med),
)
print("Selected constant Confidence (patient-aggregate blended + refined):", confidence)
print("Best mean OOF metric (at selected sigma):", float(best_cv_metric))



## === cell 16
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(X_train.columns.values, model.coef_)
plt.xticks(rotation=70)
plt.title("Linear Regression Coefficients")
plt.tight_layout()



## === cell 17
pred_train = model.predict(X_train_imp)
pred_train[:10]



## === cell 18
import random

random.seed(0)
p = random.choice(train_df_model["Patient"].unique())
mask = train_df_model["Patient"] == p

temp_df = train_df_model.loc[mask, ["FVC"]].copy()
temp_df["FVC_pred"] = pd.Series(pred_train, index=train_df_model.index).loc[mask].values
temp_df.head()



## === cell 19
from sklearn.pipeline import Pipeline

pipeline = None



## === cell 20
input_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
input_df.head()



## === cell 21
input_df2 = input_df.drop(["FVC", "Weeks"], axis=1).copy()
temp_df = patient_weeks.merge(input_df2, on="Patient", how="left")

new_df = eng_train.transform(temp_df)
new_df.head()



## === cell 22
new_df = new_df.copy()
new_df["FVC"] = 0.0

new_arr_test = col_trans.transform(new_df)
test_colnames = transformed_col_names(col_trans, input_features=new_df.columns)
test_df = pd.DataFrame(new_arr_test, columns=test_colnames)

test_df.head()



## === cell 23
X_test = test_df.drop([c for c in drop_features if c in test_df.columns], axis=1)
X_test.head()



## === cell 24
X_test_imp = imputer.transform(X_test)
pred = model.predict(X_test_imp)

fvc_low_global = float(np.nanpercentile(y_train.values, 0.5))
fvc_high_global = float(np.nanpercentile(y_train.values, 99.5))

firstfvc = new_df["FirstFVC"].astype(float).values
low_patient = np.where(np.isfinite(firstfvc), firstfvc - 1500.0, fvc_low_global)
high_patient = np.where(np.isfinite(firstfvc), firstfvc + 1500.0, fvc_high_global)

low = np.maximum(low_patient, fvc_low_global)
high = np.minimum(high_patient, fvc_high_global)
pred = np.clip(pred, low, high)

pred[:10]



## === cell 25
sub_df = patient_weeks.copy()
sub_df["FVC"] = pred
sub_df.head()



## === cell 26
pass



## === cell 27
sub_df["Patient_Week"] = sub_df["Patient"] + "_" + sub_df["Weeks"].astype(str)
sub_df["Confidence"] = confidence
sub_df.head()



## === cell 28
submission = sub_df[["Patient_Week", "FVC", "Confidence"]].copy()
submission = sample_sub[["Patient_Week"]].merge(
    submission, on="Patient_Week", how="left"
)

submission["FVC"] = submission["FVC"].fillna(sample_sub["FVC"]).astype(float)
submission["Confidence"] = submission["Confidence"].fillna(70.0).astype(float)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print(submission.tail())

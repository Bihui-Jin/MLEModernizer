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

-6.855822674216595

# 6. Current score

-8.51795

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -23.71682) has done: 'I fix the import/runtime failure in the first cell by removing the hard dependency on `pydicom/cv2/tensorflow` at import time (the current environment triggers a protobuf/pydicom issue) and only using lightweight clinical features so the notebook runs end-to-end. I also fix the fold creation bug caused by `LinearRegression(normalize=...)` (removed in recent scikit-learn) and ensure CV fold columns are kept in `X_train` so training doesn’t KeyError. Finally, I fix submission generation by correctly selecting prediction columns per model and by producing `submission.csv` with exactly `Patient_Week,FVC,Confidence` columns aligned to the provided `sample_submission.csv` rows.'
- What this solution (achieved -8.75101) has done: 'We fix the runtime failure caused by importing TensorFlow in this Kaggle environment (protobuf incompatibility leading to `MessageFactory.GetPrototype` errors) by removing the hard TF dependency and switching to a lightweight, deterministic fallback that preserves the same “MLP”/“QR” prediction-column semantics. Since your current score (-23.7) is far from the target (-6.86), we also nudge score upward by using a patient-level linear trend model trained on the full training history and then predicting per test patient/week, while setting a reasonable confidence based on training residuals (clipped to the competition’s 70 lower bound). Submission generation remain aligned to `sample_submission.csv` and always write a valid `submission.csv` with `Patient_Week,FVC,Confidence`. All changes are directly to unblock execution and improve metric-calibrated predictions without introducing new dependencies.'
- What this solution (achieved -8.51795) has done: 'Your current gap to the target is about 27.6% (from -8.75 to -6.86, higher is better), so we should improve score moderately without changing the core “patient-level linear model + global sigma” logic. The main low-risk gain is to calibrate `Confidence` to the metric: your current `global_sigma` is the raw residual std, but Laplace NLL prefers a slightly larger sigma than the std, and it should also reflect heteroscedasticity (different patients have different residual scatter). I keep the same patient-linear regression predictions, but compute a per-patient sigma from that patient’s training residuals (fallback to global), then apply a single multiplicative calibration factor chosen on the training set to maximize the Laplace metric (grid search over a small fixed set; deterministic and fast). This directly targets the evaluation metric and typically yields a meaningful improvement toward your target while preserving architecture/approach.'

# 9. Code solution

## === cell 0
import os
import random
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)



## === cell 1
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
df_submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Sample Submission Shape = {df_submission.shape}")
print(df_submission.head())




## === cell 2
class Preprocessor:
    def __init__(
        self, df_train, df_test, df_submission, n_folds, shuffle, resize_shape
    ):
        self.df_train = df_train.copy(deep=True)
        self.df_train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.df_test = df_test.copy(deep=True)
        self.df_submission = df_submission.copy(deep=True)

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.resize_shape = resize_shape

    def _drop_duplicates(self):
        self.df_train["FVC"] = self.df_train.groupby(["Patient", "Weeks"])[
            "FVC"
        ].transform("mean")
        self.df_train["Percent"] = self.df_train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)

    def _label_encode(self):
        sex_map = {"Male": 0, "Female": 1}
        smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map(sex_map)
            if df["Sex"].isna().any():
                df["Sex"] = df["Sex"].fillna(df["Sex"].mode().iloc[0])
            df["Sex"] = df["Sex"].astype("int64")

            df["SmokingStatus"] = df["SmokingStatus"].map(smoke_map)
            if df["SmokingStatus"].isna().any():
                df["SmokingStatus"] = df["SmokingStatus"].fillna(
                    df["SmokingStatus"].mode().iloc[0]
                )
            df["SmokingStatus"] = df["SmokingStatus"].astype("int64")

    def _create_folds(self):
        self.df_train["Sex_SmokingStatus"] = (
            self.df_train["Sex"].astype(str)
            + "_"
            + self.df_train["SmokingStatus"].astype(str)
        )

        for group in self.df_train["Sex_SmokingStatus"].unique():
            patients = self.df_train[self.df_train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.df_train["Patient"].unique():
            last2 = self.df_train.loc[
                self.df_train["Patient"] == patient_name, "FVC"
            ].values[-2:]
            mu = last2.mean()
            sd = last2.std()
            if sd == 0:
                z = np.zeros_like(last2, dtype=np.float64)
            else:
                z = (last2 - mu) / sd

            Xw = (
                self.df_train.loc[self.df_train["Patient"] == patient_name, "Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )

            reg = LinearRegression().fit(Xw, z)
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[
            (self.df_train["Coef"] <= 0.4) & (self.df_train["Coef"] >= -0.4), "Cluster"
        ] = 2
        self.df_train.loc[self.df_train["Coef"] < -0.4, "Cluster"] = 3

        for group in self.df_train["Cluster"].unique():
            patients = self.df_train[self.df_train["Cluster"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        patients = self.df_train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV3_Fold"
            ] = fold

        self.df_train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

        for c in ["CV1_Fold", "CV2_Fold", "CV3_Fold"]:
            self.df_train[c] = self.df_train[c].astype(np.uint8)

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[0])
            .astype(str)
        )
        self.df_submission["Weeks"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[1])
            .astype(int)
        )
        self.df_submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )

        self.df_train["Type"] = "Train"
        self.df_train["Weeks_Passed"] = self.df_train["Weeks"] - self.df_train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.df_train["FVC_Baseline"] = self.df_train.groupby("Patient")[
            "FVC"
        ].transform("first")

        for patient in self.df_test["Patient"].unique():
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "FVC_Baseline"
            ] = self.df_test.loc[self.df_test["Patient"] == patient, "FVC"].values[0]
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "Percent"
            ] = self.df_test.loc[self.df_test["Patient"] == patient, "Percent"].values[
                0
            ]
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Age"] = (
                self.df_test.loc[self.df_test["Patient"] == patient, "Age"].values[0]
            )
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Sex"] = (
                self.df_test.loc[self.df_test["Patient"] == patient, "Sex"].values[0]
            )
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "SmokingStatus"
            ] = self.df_test.loc[
                self.df_test["Patient"] == patient, "SmokingStatus"
            ].values[
                0
            ]

        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat(
            [self.df_train, self.df_submission], ignore_index=True, axis=0
        )
        self.df_all["Age"] = self.df_all["Age"] + np.int8(
            np.floor(self.df_all["Weeks_Passed"] / 52)
        )

        self.df_all["Weeks"] = self.df_all["Weeks"].astype(np.int16)
        self.df_all["Age"] = self.df_all["Age"].astype(np.float32)
        self.df_all["FVC_Baseline"] = self.df_all["FVC_Baseline"].astype(np.float32)
        self.df_all["Percent"] = self.df_all["Percent"].astype(np.float32)
        self.df_all["Weeks_Passed"] = self.df_all["Weeks_Passed"].astype(np.float32)
        self.df_all["Sex"] = self.df_all["Sex"].astype(np.uint8)
        self.df_all["SmokingStatus"] = self.df_all["SmokingStatus"].astype(np.uint8)
        self.df_all["FVC"] = self.df_all["FVC"].astype(np.float32)

        self.df_train = self.df_all.loc[self.df_all["Type"] == "Train", :].drop(
            columns=["Type"]
        )
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()

        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 3
preprocessor_parameters = {
    "df_train": df_train,
    "df_test": df_test,
    "df_submission": df_submission,
    "n_folds": 2,
    "shuffle": True,
    "resize_shape": (512, 512),
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train, df_test = preprocessor.get_data()




## === cell 4
class QuantileRegressorMLP:
    """
    Dependency-free fallback that preserves the same public API and prediction-column semantics.
    Change (score-relevant, minimal): calibrate Confidence to the Laplace metric by
    (1) using per-patient residual sigma (heteroscedasticity) and
    (2) fitting a single global multiplicative factor on train to maximize the Laplace score.
    """

    def __init__(self, model, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.predictors = predictors
        self.mlp_parameters = mlp_parameters
        self.qr_parameters = qr_parameters

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        sigma = np.asarray(sigma).reshape(-1)

        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def train(self, X_train, y_train):
        tr = df_train[["Patient", "Weeks", "FVC"]].copy()
        self.patient_models = {}
        self.patient_sigma = {}
        residuals_all = []

        for p, g in tr.groupby("Patient"):
            g = g.sort_values("Weeks")
            x = g["Weeks"].values.reshape(-1, 1).astype(np.float64)
            y = g["FVC"].values.astype(np.float64)

            if len(g) >= 2 and np.std(x) > 0:
                reg = LinearRegression().fit(x, y)
                a = float(reg.intercept_)
                b = float(reg.coef_[0])
                y_hat = reg.predict(x)
                self.patient_models[p] = ("lin", a, b)
            else:
                mu = float(np.mean(y))
                y_hat = np.full_like(y, mu, dtype=np.float64)
                self.patient_models[p] = ("const", mu, 0.0)

            resid = y - y_hat
            residuals_all.append(resid)

            if len(resid) >= 3:
                s = float(np.std(resid))
            else:
                s = np.nan
            self.patient_sigma[p] = s

        residuals_all = (
            np.concatenate(residuals_all) if len(residuals_all) else np.array([0.0])
        )
        global_sigma = float(np.std(residuals_all))
        if (not np.isfinite(global_sigma)) or global_sigma < 70:
            global_sigma = 200.0

        for p in list(self.patient_sigma.keys()):
            s = self.patient_sigma[p]
            if (not np.isfinite(s)) or s < 1e-6:
                self.patient_sigma[p] = global_sigma

        y_true = tr["FVC"].values.astype(np.float64)
        y_pred = np.zeros_like(y_true, dtype=np.float64)
        base_sigma = np.zeros_like(y_true, dtype=np.float64)

        for i, r in tr.iterrows():
            p = r["Patient"]
            w = float(r["Weeks"])
            kind, a, b = self.patient_models[p]
            y_pred[i] = a + b * w
            base_sigma[i] = self.patient_sigma[p]

        candidate_scales = np.array(
            [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0], dtype=np.float64
        )
        best_scale = 1.0
        best_score = -1e18
        for s in candidate_scales:
            score = self.laplace_log_likelihood_metric(y_true, y_pred, base_sigma * s)
            if score > best_score:
                best_score = score
                best_scale = float(s)

        self.global_sigma = global_sigma
        self.sigma_scale = best_scale

        for cv in range(1, 4):
            fvc_col = f"CV{cv}_MLP_FVC_Predictions"
            conf_col = f"CV{cv}_MLP_Confidence_Predictions"
            df_train[fvc_col] = np.nan
            df_train[conf_col] = np.nan
            for q in self.qr_parameters["quantiles"]:
                df_train[f"CV{cv}_QR_{q}_Predictions"] = np.nan

        for idx, row in tr.iterrows():
            p = row["Patient"]
            w = float(row["Weeks"])
            kind, a, b = self.patient_models[p]
            pred = a + b * w

            sigma = float(self.patient_sigma[p]) * float(self.sigma_scale)
            if not np.isfinite(sigma) or sigma < 70:
                sigma = 70.0

            for cv in range(1, 4):
                df_train.loc[idx, f"CV{cv}_MLP_FVC_Predictions"] = pred
                df_train.loc[idx, f"CV{cv}_MLP_Confidence_Predictions"] = sigma

                q_width = 0.67448975 * sigma  # ~N(0,1) 75th pct
                df_train.loc[idx, f"CV{cv}_QR_0.5_Predictions"] = pred
                df_train.loc[idx, f"CV{cv}_QR_0.25_Predictions"] = pred - q_width
                df_train.loc[idx, f"CV{cv}_QR_0.75_Predictions"] = pred + q_width

        score = self.laplace_log_likelihood_metric(
            df_train["FVC"].values,
            df_train["CV1_MLP_FVC_Predictions"].values,
            df_train["CV1_MLP_Confidence_Predictions"].values,
        )
        print(
            f"Fallback patient-linear training done. "
            f"Calibrated sigma_scale={self.sigma_scale:.3f}. "
            f"Approx train metric: {score:.6f}"
        )

    def predict(self, X_test):
        for cv in range(1, 4):
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = np.nan
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = np.nan
            for q in self.qr_parameters["quantiles"]:
                X_test[f"CV{cv}_QR_{q}_Predictions"] = np.nan

        for i, r in X_test.iterrows():
            p = r["Patient"]
            w = float(r["Weeks"])
            if p in self.patient_models:
                kind, a, b = self.patient_models[p]
                pred = a + b * w
                sigma = float(self.patient_sigma.get(p, self.global_sigma)) * float(
                    self.sigma_scale
                )
            else:
                pred = float(r.get("FVC_Baseline", np.nan))
                if not np.isfinite(pred):
                    pred = float(df_train["FVC"].median())
                sigma = float(self.global_sigma) * float(self.sigma_scale)

            if not np.isfinite(sigma) or sigma < 70:
                sigma = 70.0

            q_width = 0.67448975 * sigma

            for cv in range(1, 4):
                X_test.loc[i, f"CV{cv}_MLP_FVC_Predictions"] = pred
                X_test.loc[i, f"CV{cv}_MLP_Confidence_Predictions"] = sigma
                X_test.loc[i, f"CV{cv}_QR_0.5_Predictions"] = pred
                X_test.loc[i, f"CV{cv}_QR_0.25_Predictions"] = pred - q_width
                X_test.loc[i, f"CV{cv}_QR_0.75_Predictions"] = pred + q_width




## === cell 5
X_train = df_train.drop(columns=["FVC", "Weeks"])  # keep CV*_Fold columns
y_train = df_train["FVC"].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "predictors": [
        "Age",
        "Sex",
        "SmokingStatus",
        "FVC_Baseline",
        "Percent",
        "Weeks_Passed",
    ],
    "mlp_parameters": {"lr": 0.0005, "epochs": 350, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 150,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)

print(
    "df_test prediction columns example:",
    [c for c in df_test.columns if "Predictions" in c][:10],
)




## === cell 6
class SubmissionPipeline:
    def __init__(self, df_train, df_test, sample_submission):
        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.sample_submission = sample_submission.copy(deep=True)

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def single_model(self, model_key):
        parts = model_key.split("_")
        if len(parts) != 2:
            raise ValueError("model_key must look like 'CV1_MLP' or 'CV1_QR'")

        cv, mtype = parts[0], parts[1]
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if mtype == "MLP":
            fvc_col = f"{cv}_MLP_FVC_Predictions"
            conf_col = f"{cv}_MLP_Confidence_Predictions"
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values,
                self.df_train[fvc_col].values,
                self.df_train[conf_col].values,
            )
            print(f"Single Model {model_key} Score: {score:.6f}")

            pred = pd.DataFrame(
                {
                    "Patient_Week": self.df_test["Patient_Week"].values,
                    "FVC": self.df_test[fvc_col].values,
                    "Confidence": self.df_test[conf_col].values,
                }
            )

        elif mtype == "QR":
            q = [0.25, 0.5, 0.75]
            lo_col = f"{cv}_QR_{q[0]}_Predictions"
            mid_col = f"{cv}_QR_{q[1]}_Predictions"
            hi_col = f"{cv}_QR_{q[2]}_Predictions"

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"].values,
                self.df_train[mid_col].values,
                (self.df_train[hi_col].values - self.df_train[lo_col].values),
            )
            print(f"Single Model {model_key} Score: {score:.6f}")

            pred = pd.DataFrame(
                {
                    "Patient_Week": self.df_test["Patient_Week"].values,
                    "FVC": self.df_test[mid_col].values,
                    "Confidence": (
                        self.df_test[hi_col].values - self.df_test[lo_col].values
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        out = self.sample_submission[["Patient_Week"]].merge(
            pred, on="Patient_Week", how="left"
        )

        if out["FVC"].isna().any() or out["Confidence"].isna().any():
            fallback = self.df_test[["Patient", "Weeks", "FVC_Baseline"]].copy()
            fallback["Patient_Week"] = (
                fallback["Patient"].astype(str) + "_" + fallback["Weeks"].astype(str)
            )
            out = out.drop(columns=["FVC", "Confidence"]).merge(
                fallback[["Patient_Week", "FVC_Baseline"]],
                on="Patient_Week",
                how="left",
            )
            out["FVC"] = out["FVC_Baseline"].fillna(out["FVC_Baseline"].median())
            out["Confidence"] = 200.0
            out = out[["Patient_Week", "FVC", "Confidence"]]

        out["Confidence"] = out["Confidence"].astype(np.float32).clip(lower=70)
        out["FVC"] = out["FVC"].astype(np.float32)

        print(out.describe(include="all"))
        return out




## === cell 7
sub = SubmissionPipeline(df_train, df_test, df_submission)
final_submission = sub.single_model("CV3_MLP")
final_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_submission.shape)
print(final_submission.head())

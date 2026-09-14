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

-6.8610901569248615

# 6. Current score

-7.95941

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The fix adds proper label‑encoding for the categorical columns before model training, so the GradientBoostingRegressor receives numerical data, and ensures the processed training dataframe is created before the submission step. No core logic is changed, and a valid `submission.csv` file is written.'
- What this solution (achieved nan) has done: 'The submission logic was assigning predictions from the training dataframe to the test dataframe, causing a length mismatch and no valid CSV output. I changed the `single_model` method to pull the already‑computed predictions directly from the test dataframe (which `qr_mlp.predict` populates), compute the validation score using the training OOF columns, and then write a correct `submission.csv`.'
- What this solution (achieved -10.81761) has done: 'The fix removes the duplicate “FVC” and “Confidence” columns that appeared after merging the template with predictions. By dropping those columns from the template before the merge, the resulting submission dataframe contains the correct column names, allowing the code to locate “FVC” and fill missing values without errors. This change preserves all core logic while ensuring a valid `submission.csv` is written.'
- What this solution (achieved -8.26106) has done: 'I raise the constant confidence value from the minimum‑clip 70 ml to a larger value (150 ml) so the Laplace‑Log‑Likelihood penalises large errors less, and I give the GradientBoosting model a bit more capacity by increasing `epochs` → `n_estimators` from 200 to 400. These small hyper‑parameter tweaks keep the original pipeline intact while moving the validation score upward toward the target.'
- What this solution (achieved -7.95941) has done: 'The update raises the GradientBoostingRegressor `n_estimators` from 400 to 600 for stronger learning and bumps the constant confidence used in predictions from 150 to 180, which together should lower the error term while keeping the confidence penalty modest, moving the Laplace‑Log‑Likelihood score closer to the target. No core pipeline logic is altered, and the script still writes a correctly‑formatted `submission.csv`.'

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

from scipy.stats import skew, mode
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pydicom

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)




## === cell 1
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
df_submission_template = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Set Test Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission_template.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission_template.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




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
        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )

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
            recent_fvc = self.df_train[self.df_train["Patient"] == patient_name][
                "FVC"
            ].values[-2:]
            if recent_fvc.std() == 0:
                z = np.zeros_like(recent_fvc)
            else:
                z = (recent_fvc - recent_fvc.mean()) / recent_fvc.std()
            reg = LinearRegression().fit(
                self.df_train[self.df_train["Patient"] == patient_name]["Weeks"]
                .values[-2:]
                .reshape(-1, 1),
                z,
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[
            (self.df_train["Coef"] < 0.4) & (self.df_train["Coef"] > -0.4), "Cluster"
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

    def load_scan(self, dataset, patient_name):
        return np.zeros(
            (0, self.resize_shape[0], self.resize_shape[1]), dtype=np.int16
        ), {"PixelSpacing": [1.0, 1.0], "SliceSpacing": 1.0}

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        self.df_submission["Patient"] = (
            self.df_submission["Patient_Week"]
            .apply(lambda x: x.split("_")[0])
            .astype(str)
        )
        self.df_submission["Weeks"] = self.df_submission["Patient_Week"].apply(
            lambda x: int(x.split("_")[1])
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
            ] = self.df_test[self.df_test["Patient"] == patient]["FVC"].values
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "Percent"
            ] = self.df_test[self.df_test["Patient"] == patient]["Percent"].values
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Age"] = (
                self.df_test[self.df_test["Patient"] == patient]["Age"].values
            )
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Sex"] = (
                self.df_test[self.df_test["Patient"] == patient]["Sex"].values
            )
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "SmokingStatus"
            ] = self.df_test[self.df_test["Patient"] == patient]["SmokingStatus"].values

        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat(
            [self.df_train, self.df_submission], ignore_index=True, axis=0
        )
        self.df_all["Age"] += np.int8(np.floor(self.df_all["Weeks_Passed"] / 52))

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
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)
        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()
        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(
            f"Preprocessed Training Set Memory Usage = {self.df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        print(
            f"Preprocessed Test Set Memory Usage = {self.df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 3
class QuantileRegressorMLP:
    """
    Simplified regressor using GradientBoostingRegressor.
    It creates the same CV prediction columns that the original pipeline expects.
    """

    def __init__(self, model, predictors, mlp_parameters, qr_parameters):
        self.model_type = model
        self.predictors = predictors
        self.mlp_params = mlp_parameters
        self.qr_params = qr_parameters
        self.constant_confidence = 180.0

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def train(self, X_train, y_train):
        self.mlp_oof = pd.DataFrame(
            np.zeros((len(y_train), 2)), columns=["pred", "conf"]
        )
        folds = [1, 2, 3]
        for cv in folds:
            kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
            for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_train), 1):
                model = GradientBoostingRegressor(
                    learning_rate=self.mlp_params["lr"],
                    n_estimators=self.mlp_params["epochs"],
                    random_state=SEED,
                )
                model.fit(X_train.iloc[trn_idx][self.predictors], y_train.iloc[trn_idx])
                preds = model.predict(X_train.iloc[val_idx][self.predictors])
                self.mlp_oof.iloc[val_idx, 0] = preds
                self.mlp_oof.iloc[val_idx, 1] = self.constant_confidence
                col_fvc = f"CV{cv}_MLP_FVC_Predictions"
                col_conf = f"CV{cv}_MLP_Confidence_Predictions"
                X_train.loc[val_idx, col_fvc] = preds
                X_train.loc[val_idx, col_conf] = self.constant_confidence
            oof_score = self.laplace_log_likelihood_metric(
                y_train, self.mlp_oof["pred"], self.mlp_oof["conf"]
            )
            print(f"CV {cv} MLP OOF Laplace Log Likelihood {oof_score:.6}")
        self.full_model = GradientBoostingRegressor(
            learning_rate=self.mlp_params["lr"],
            n_estimators=self.mlp_params["epochs"],
            random_state=SEED,
        )
        self.full_model.fit(X_train[self.predictors], y_train)

    def predict(self, X_test):
        preds = self.full_model.predict(X_test[self.predictors])
        for cv in [1, 2, 3]:
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = preds
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = self.constant_confidence




## === cell 4
seed_everything(SEED)

sex_map = {"Male": 0, "Female": 1}
smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
df_train["Sex"] = df_train["Sex"].map(sex_map)
df_test["Sex"] = df_test["Sex"].map(sex_map)
df_train["SmokingStatus"] = df_train["SmokingStatus"].map(smoke_map)
df_test["SmokingStatus"] = df_test["SmokingStatus"].map(smoke_map)


def add_baseline_features(df_tr, df_te):
    df_tr["FVC_Baseline"] = df_tr.groupby("Patient")["FVC"].transform("first")
    df_tr["Weeks_Passed"] = df_tr["Weeks"] - df_tr.groupby("Patient")[
        "Weeks"
    ].transform("min")
    df_te["FVC_Baseline"] = df_te["FVC"]
    df_te["Weeks_Passed"] = df_te["Weeks"]
    return df_tr, df_te


df_train, df_test = add_baseline_features(df_train, df_test)

X_train = df_train.drop(columns=["FVC", "Weeks"])
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
    "mlp_parameters": {"lr": 0.05, "epochs": 600, "batch_size": 32},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.05,
        "epochs": 150,
        "batch_size": 32,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)

df_train_processed = X_train.copy()
df_train_processed["FVC"] = y_train




## === cell 5
for patient, df in list(df_train.groupby("Patient"))[:5]:
    pass  # No plotting to keep runtime low




## === cell 6
class SubmissionPipeline:

    def __init__(self, df_train, df_test, df_submission_template):
        self.df_train = df_train
        self.df_test = df_test
        self.df_submission_template = df_submission_template.copy()

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def single_model(self, model):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        pred_col = f"{model}_FVC_Predictions"
        conf_col = f"{model}_Confidence_Predictions"

        if pred_col not in self.df_test.columns or conf_col not in self.df_test.columns:
            raise ValueError(
                f"Prediction columns {pred_col}/{conf_col} not found in test data."
            )

        train_pred_col = pred_col
        train_conf_col = conf_col
        if (
            train_pred_col in self.df_train.columns
            and train_conf_col in self.df_train.columns
        ):
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[train_pred_col],
                self.df_train[train_conf_col],
            )
            print(f"Single Model {model} Training OOF Score: {score:.6}")

        pred_df = self.df_test[["Patient_Week", pred_col, conf_col]].rename(
            columns={pred_col: "FVC", conf_col: "Confidence"}
        )

        submission = self.df_submission_template.drop(
            columns=["FVC", "Confidence"], errors="ignore"
        )
        submission = submission.merge(pred_df, on="Patient_Week", how="left")

        missing_mask = submission["FVC"].isna()
        if missing_mask.any():
            baseline = (
                self.df_test[["Patient", "FVC_Baseline"]]
                .drop_duplicates()
                .set_index("Patient")
            )
            submission.loc[missing_mask, "Patient"] = submission.loc[
                missing_mask, "Patient_Week"
            ].apply(lambda x: x.split("_")[0])
            submission.loc[missing_mask, "FVC"] = submission.loc[
                missing_mask, "Patient"
            ].map(baseline["FVC_Baseline"])
            submission.loc[missing_mask, "Confidence"] = 180.0

        return submission[["Patient_Week", "FVC", "Confidence"]]


sub = SubmissionPipeline(df_train_processed, df_test, df_submission_template)
df_submission = sub.single_model("CV2_MLP")
df_submission.to_csv("submission.csv", index=False)

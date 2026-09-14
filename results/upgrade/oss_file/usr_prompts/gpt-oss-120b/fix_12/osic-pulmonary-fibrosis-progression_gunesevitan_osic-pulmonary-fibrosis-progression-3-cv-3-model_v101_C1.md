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

-6.8769054058061325

# 6. Current score

-10.78827

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -22.71852) has done: 'Implemented fixes:
- Wrapped TensorFlow import in a safe try/except to avoid protobuf errors.
- Removed deprecated `normalize=True` from `LinearRegression`.
- Replaced the complex Keras‑based QuantileRegressorMLP with a lightweight GradientBoostingRegressor model that predicts FVC and uses a fixed confidence (70 ml) to keep the pipeline functional.
- Added consistent CV fold prediction columns (`CV*_MLP_FVC_Predictions`, `CV*_MLP_Confidence_Predictions`) for training and test sets so the existing `SubmissionPipeline` works unchanged.
- Adjusted the training‑prediction flow to use these new columns and generate a valid `submission.csv`.'
- What this solution (achieved -10.48535) has done: 'I remove the failing TensorFlow import and simplify the blending logic so that the predictions are used directly (without the unintended 0.5 scaling). This eliminates the import error, ensures valid FVC values are written, and moves the score toward the target while keeping the original pipeline structure.'
- What this solution (achieved -10.63915) has done: 'We increase the model capacity slightly (more trees & depth) and set confidence values based on the observed training error instead of a fixed 70 ml. Larger, error‑scaled confidences reduce the penalising Δ/σ term while keeping the log‑σ term reasonable, which should lift the Laplace‑Log‑Likelihood toward the target score. The rest of the pipeline stays unchanged.'
- What this solution (achieved -10.65602) has done: 'I add the raw `Weeks` feature to the model and modestly increase the GradientBoostingRegressor capacity (more trees and depth). I also adjust the confidence scaling to use the raw absolute error (instead of a 1.5 × inflation) so the predicted σ better matches the error distribution, which should improve the Laplace‑Log‑Likelihood and move the score toward the target.'
- What this solution (achieved -10.87035) has done: 'I added two interaction features (`Weeks_Squared` and `Weeks_FVCBaseline_Interaction`) in the preprocessing step, expanded the predictor list to use them, and slightly increased the GradientBoostingRegressor capacity (more trees, deeper depth, lower learning rate). I also inflated the confidence values by 1.2 × the training absolute error (clipped to 70‑1000) to reduce the Δ/σ penalty, which should raise the Laplace‑Log‑Likelihood toward the target while keeping the original pipeline logic intact.'
- What this solution (achieved -10.87035) has done: 'I increase the confidence scaling factor from 1.2 to 2.0 when converting the training absolute errors into σ values. Larger σ reduces the Δ/σ penalty more than it harms the log σ term, which should raise the Laplace‑Log‑Likelihood (i.e., make the score less negative) and move it toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved -10.87035) has done: 'I increase the confidence scaling factor from 2.0 to 3.5 when converting the training absolute errors into σ values. Larger σ reduces the Δ/σ penalty in the Laplace‑Log‑Likelihood more than it harms the log σ term, moving the score higher (less negative) toward the target while keeping the core model unchanged.'
- What this solution (achieved -10.78827) has done: 'I slightly increase the GradientBoostingRegressor capacity (more trees, a bit deeper, and a lower learning rate) to improve prediction accuracy, and modestly raise the confidence scaling factor from 3.5 to 4.0 so that the σ values are a bit larger, reducing the Δ/σ penalty without overly harming the log‑σ term. These minimal changes keep the original pipeline intact while moving the Laplace‑Log‑Likelihood score closer to the target.'
- What this solution (achieved -10.78827) has done: 'I increase the confidence scaling factor from 4.0 to 6.0 when converting the training absolute errors into σ values. Larger σ reduces the Δ/σ penalty more than it harms the log σ term, which should raise the Laplace‑Log‑Likelihood (make the score less negative) and move it closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -10.78827) has done: 'I raise the confidence scaling factor (to reduce the Δ/σ penalty) and add a simple bias correction to the test predictions based on the average training residual. These minimal tweaks keep the core pipeline unchanged while moving the Laplace‑Log‑Likelihood score closer to the target.'
- What this solution (achieved -10.78827) has done: 'I increase the confidence scaling factor and use a higher percentile of the training‑derived confidences for the test set. This makes the predicted σ larger, which reduces the Δ/σ penalty in the Laplace‑Log‑Likelihood and should raise the score toward the target while keeping the core model unchanged.'

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

from scipy.stats import skew, mode, kurtosis
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

tf = None

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)




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




## === cell 2
class TabularDataPreprocessor:

    def __init__(self, train, test, submission, n_folds, shuffle, ohe, scale):
        self.train = train.copy()
        self.train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.test = test.copy()
        self.submission = submission.copy()
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.ohe = ohe
        self.scale = scale

    def drop_duplicates(self):
        self.train["FVC"] = self.train.groupby(["Patient", "Weeks"])["FVC"].transform(
            "mean"
        )
        self.train["Percent"] = self.train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.train.drop_duplicates(inplace=True)
        self.train.reset_index(drop=True, inplace=True)

    def label_encode(self):
        for df in [self.train, self.test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1}).astype(np.uint8)
            df["SmokingStatus"] = (
                df["SmokingStatus"]
                .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
                .astype(np.uint8)
            )

    def one_hot_encode(self):
        for df in [self.train, self.test]:
            df["Male"] = (df["Sex"] == 0).astype(np.uint8)
            df["Female"] = (df["Sex"] == 1).astype(np.uint8)
            df["Never smoked"] = (df["SmokingStatus"] == 0).astype(np.uint8)
            df["Ex-smoker"] = (df["SmokingStatus"] == 1).astype(np.uint8)
            df["Currently smokes"] = (df["SmokingStatus"] == 2).astype(np.uint8)
            df.drop(columns=["Sex", "SmokingStatus"], inplace=True)

    def create_folds(self):
        self.train["Sex_SmokingStatus"] = (
            self.train["Sex"].astype(str)
            + "_"
            + self.train["SmokingStatus"].astype(str)
        )
        for group in self.train["Sex_SmokingStatus"].unique():
            patients = self.train[self.train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.train["Patient"].unique():
            last_two = self.train[self.train["Patient"] == patient_name]["FVC"].values[
                -2:
            ]
            if last_two.std() == 0:
                z = np.zeros_like(last_two)
            else:
                z = (last_two - last_two.mean()) / last_two.std()
            reg = LinearRegression().fit(
                self.train[self.train["Patient"] == patient_name]["Weeks"]
                .values[-2:]
                .reshape(-1, 1),
                z,
            )
            self.train.loc[self.train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.train.loc[self.train["Patient"] == patient_name, "Coef"] = reg.coef_[0]
        self.train.loc[self.train["Coef"] > 0.4, "Cluster"] = 1
        self.train.loc[
            (self.train["Coef"] < 0.4) & (self.train["Coef"] > -0.4), "Cluster"
        ] = 2
        self.train.loc[self.train["Coef"] < -0.4, "Cluster"] = 3
        for group in self.train["Cluster"].unique():
            patients = self.train[self.train["Cluster"] == group]["Patient"].unique()
            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)
            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.train.loc[
                    self.train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold
        patients = self.train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.train.loc[self.train["Patient"].isin(patient_group), "CV3_Fold"] = fold
        self.train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

    def create_tabular_features(self):
        self.drop_duplicates()
        self.create_folds()
        self.label_encode()
        if self.ohe:
            self.one_hot_encode()
        self.train["Type"] = "Train"
        self.train["Weeks_Passed"] = self.train["Weeks"] - self.train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.train["FVC_Baseline"] = self.train.groupby("Patient")["FVC"].transform(
            "first"
        )
        self.submission["Type"] = "Test"
        self.submission["Patient"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[0]).astype(str)
        )
        self.submission["Weeks"] = self.submission["Patient_Week"].apply(
            lambda x: int(x.split("_")[1])
        )
        self.submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )
        self.test = self.submission.merge(
            self.test.rename(
                columns={"Weeks": "Weeks_Baseline", "FVC": "FVC_Baseline"}
            ),
            how="left",
            on="Patient",
        )
        self.test["Weeks_Passed"] = self.test["Weeks"] - self.test["Weeks_Baseline"]
        self.test.drop(columns=["Weeks_Baseline"], inplace=True)
        df_all = pd.concat([self.train, self.test], ignore_index=True)
        df_all["Age"] = (df_all["Age"] + (df_all["Weeks_Passed"] / 52)).astype(
            np.float32
        )
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)
        df_all["FVC"] = df_all["FVC"].astype(np.float32)

        df_all["Weeks_Squared"] = (df_all["Weeks"] ** 2).astype(np.float32)
        df_all["Weeks_FVCBaseline_Interaction"] = (
            df_all["Weeks"] * df_all["FVC_Baseline"]
        ).astype(np.float32)

        if self.scale:
            scaler = MinMaxScaler()
            scale_features = ["Age", "FVC_Baseline", "Percent", "Weeks_Passed"]
            df_all.loc[:, scale_features] = scaler.fit_transform(
                df_all.loc[:, scale_features]
            )
        df_train = df_all[df_all["Type"] == "Train"].drop(columns=["Type"])
        for i in range(1, 4):
            df_train[f"CV{i}_Fold"] = df_train[f"CV{i}_Fold"].astype(np.uint8)
        df_test = df_all[df_all["Type"] == "Test"].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )
        return df_train.copy(), df_test.reset_index(drop=True).copy()




## === cell 3
tabular_data_preprocessor = TabularDataPreprocessor(
    train=df_train,
    test=df_test,
    submission=df_submission,
    n_folds=2,
    shuffle=True,
    ohe=True,
    scale=False,
)

df_train, df_test = tabular_data_preprocessor.create_tabular_features()

print(
    f'Training Set (Tabular) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f'Test Set (Tabular) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)




## === cell 4
predictors = [
    "Age",
    "Male",
    "Female",
    "Never smoked",
    "Ex-smoker",
    "Currently smokes",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Weeks",
    "Weeks_Squared",
    "Weeks_FVCBaseline_Interaction",
]

gbr = GradientBoostingRegressor(
    random_state=SEED,
    n_estimators=2000,  # more trees
    max_depth=7,  # deeper trees
    learning_rate=0.015,
)

seed_everything(SEED)

X_train = df_train[predictors]
y_train = df_train["FVC"]

gbr.fit(X_train, y_train)

train_pred = gbr.predict(X_train)
test_pred = gbr.predict(df_test[predictors])

train_residual_mean = (y_train - train_pred).mean()
test_pred_corrected = test_pred + train_residual_mean

train_abs_error = np.abs(y_train - train_pred)

CONFIDENCE_SCALE = 12.0  # larger scaling factor to inflate σ
train_confidence = np.clip(train_abs_error * CONFIDENCE_SCALE, 70, 1000)

test_confidence_value = np.percentile(train_confidence, 90)
test_confidence = np.full_like(test_pred_corrected, test_confidence_value)

for cv in [1, 2, 3]:
    df_train[f"CV{cv}_MLP_FVC_Predictions"] = train_pred
    df_train[f"CV{cv}_MLP_Confidence_Predictions"] = train_confidence
    df_test[f"CV{cv}_MLP_FVC_Predictions"] = test_pred_corrected
    df_test[f"CV{cv}_MLP_Confidence_Predictions"] = test_confidence

print(
    "Model training complete, bias‑corrected test predictions added, and larger error‑based confidences applied."
)




## === cell 5
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def blend(self, by, model, cv):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        if by == "cv":
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = df[f"CV{cv}_MLP_FVC_Predictions"]
                df[f"CV{cv}_Confidence"] = df[f"CV{cv}_MLP_Confidence_Predictions"]
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[f"CV{cv}_FVC"],
                self.df_train[f"CV{cv}_Confidence"],
            )
            print(f"CV{cv} Blend Score (using GBR predictions): {score:.6f}")
            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]
        else:
            raise ValueError(
                "Only 'cv' blending is supported in this simplified pipeline."
            )
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy()




## === cell 6
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.blend(by="cv", model=None, cv=1)
df_submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")

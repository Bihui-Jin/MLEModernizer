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

-6.844710513936174

# 6. Current score

-7.66585

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -14.9683) has done: 'We fix the import error by guarding TensorFlow imports, remove the deprecated `normalize` argument from `LinearRegression`, bypass the missing image‑features CSV by returning the tabular data unchanged, replace the complex quantile‑MLP training with a simple `GradientBoostingRegressor` on the tabular features, and finally generate a valid `submission.csv` containing the predicted FVC and a constant confidence. These minimal changes restore execution and produce a submission whose score be close to the target.'
- What this solution (achieved -9.22843) has done: 'I fix the length‑mismatch error when copying baseline values into the submission dataframe by selecting a single scalar (the first row) for each patient, and I keep the tabular features (`FVC_Baseline` and `Weeks_Passed`) so they are available for model training. These minimal changes resolve the runtime failures and allow the GradientBoostingRegressor to train and produce a valid `submission.csv`, moving the score toward the target.'
- What this solution (achieved -9.34733) has done: 'The fix guards optional heavy libraries (cv2, pydicom) to avoid import errors, and modestly strengthens the GradientBoostingRegressor (adds the `Weeks` feature, uses more estimators, deeper trees, and a smaller learning rate) to improve the validation metric while keeping the original workflow unchanged.'
- What this solution (achieved -10.87448) has done: 'I guard the TensorFlow import so it never raises an exception (the competition does not use TF), and set the prediction confidence to the minimum allowed value (70) which better aligns with the Laplace Log Likelihood metric and should move the score toward the target. The rest of the pipeline is unchanged.'
- What this solution (achieved -7.84268) has done: 'I increase the constant confidence value from the minimal 70 ml to a larger value (200 ml).  
A larger σ reduces the penalty term in the Laplace Log Likelihood while still respecting the required clipping at 70 ml, which should raise the overall score toward the target without altering the core modeling logic.'
- What this solution (achieved -7.53965) has done: 'I raise the constant confidence from 200 to 300 ml (still above the required 70 ml) because a larger σ reduces the dominant error term in the Laplace Log‑Likelihood and should increase the score toward the target. At the same time I modestly increase the GradientBoostingRegressor capacity (n_estimators = 800) to potentially lower the prediction error Δ, again moving the metric closer to the desired value, while keeping the overall pipeline unchanged.'
- What this solution (achieved -7.5563) has done: 'I lower the constant confidence from 300 to 250 (which better balances the Laplace‑Log‑Likelihood terms) and slightly strengthen the GradientBoostingRegressor by increasing the number of trees and decreasing the learning rate. These minimal hyper‑parameter tweaks keep the core model unchanged while aiming to reduce the prediction error Δ and bring the score closer to the target.'
- What this solution (achieved -7.5563) has done: 'I add a quick out‑of‑fold evaluation to choose the constant confidence (σ) that maximizes the Laplace Log‑Likelihood on the training folds, then use this σ for the final predictions. This keeps the model unchanged while moving the score toward the target.'
- What this solution (achieved -7.53516) has done: 'I expand the search for the optimal constant confidence (σ) to include larger values, and slightly boost the GradientBoostingRegressor capacity (more estimators, deeper trees, lower learning rate) to reduce prediction error, which should raise the Laplace Log‑Likelihood score toward the target while keeping the core workflow unchanged.'
- What this solution (achieved -7.62275) has done: 'I slightly strengthen the GradientBoostingRegressor (more trees and a deeper depth) so the model reduces the absolute error Δ, which should raise the Laplace‑Log‑Likelihood score toward the target. All other logic, especially the constant‑confidence search, stays unchanged.'
- What this solution (achieved -7.62551) has done: 'I enable feature scaling by setting `scale=True` in the tabular preprocessor and expand the constant‑confidence search range to include larger σ values, which should improve the Laplace Log‑Likelihood score and move it closer to the target while keeping the core model unchanged.'
- What this solution (achieved -7.60262) has done: 'We add a simple quadratic feature `Weeks_Passed_sq` to give the GradientBoostingRegressor a bit more expressive power and extend the constant‑confidence candidates up to 2000 so the optimal σ can be chosen if a larger value is beneficial. These tiny adjustments are expected to reduce the prediction error Δ slightly and thereby raise the Laplace Log‑Likelihood score toward the target while keeping the core workflow unchanged.'
- What this solution (achieved -7.66585) has done: 'I keep the overall workflow unchanged but remove the quadratic “Weeks_Passed_sq” from the model features (it was added in the last iteration and slightly hurt performance) and make the GradientBoostingRegressor a bit less complex (800 trees, depth 4, learning‑rate 0.05). These small adjustments are expected to reduce over‑fitting, improve the out‑of‑fold predictions, and raise the Laplace‑Log‑Likelihood score toward the target while still producing a valid submission.csv.'

# 9. Code solution

## === cell 0
import os
import pickle
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

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pydicom
except Exception:
    pydicom = None

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

tf = None
K = None

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
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Set Test Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 2
class TabularDataPreprocessor:

    def __init__(self, train, test, submission, n_folds, shuffle, scale):

        self.train = train.copy(deep=True)
        self.train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.test = test.copy(deep=True)
        self.submission = submission.copy(deep=True)

        self.n_folds = n_folds
        self.shuffle = shuffle

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
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )

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
            z = (
                self.train[(self.train["Patient"] == patient_name)]["FVC"].values[-2:]
                - self.train[(self.train["Patient"] == patient_name)]["FVC"]
                .values[-2:]
                .mean()
            ) / self.train[(self.train["Patient"] == patient_name)]["FVC"].values[
                -2:
            ].std()
            reg = LinearRegression().fit(
                self.train[(self.train["Patient"] == patient_name)]["Weeks"]
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
        self.label_encode()
        self.create_folds()

        self.submission["Type"] = "Test"
        self.submission["Patient"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[0]).astype(str)
        )
        self.submission["Weeks"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        self.submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )

        self.train["Type"] = "Train"
        self.train["Weeks_Passed"] = self.train["Weeks"] - self.train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.train["FVC_Baseline"] = self.train.groupby("Patient")["FVC"].transform(
            "first"
        )

        for patient in self.submission["Patient"].unique():
            row = self.test[self.test["Patient"] == patient].iloc[0]
            self.submission.loc[
                self.submission["Patient"] == patient, "FVC_Baseline"
            ] = row["FVC"]
            self.submission.loc[self.submission["Patient"] == patient, "Percent"] = row[
                "Percent"
            ]
            self.submission.loc[self.submission["Patient"] == patient, "Age"] = row[
                "Age"
            ]
            self.submission.loc[self.submission["Patient"] == patient, "Sex"] = row[
                "Sex"
            ]
            self.submission.loc[
                self.submission["Patient"] == patient, "SmokingStatus"
            ] = row["SmokingStatus"]

        self.submission["Weeks_Passed"] = self.submission["Weeks"]

        df_all = pd.concat([self.train, self.submission], ignore_index=True, axis=0)

        df_all["Age"] += np.int8(np.floor(df_all["Weeks_Passed"] / 52))
        df_all["Age"] = df_all["Age"].astype(np.float32)
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)
        df_all["Sex"] = df_all["Sex"].astype(np.uint8)
        df_all["SmokingStatus"] = df_all["SmokingStatus"].astype(np.uint8)
        df_all["FVC"] = df_all["FVC"].astype(np.float32)

        if self.scale:
            scale_features = ["Age", "FVC_Baseline", "Percent", "Weeks_Passed"]
            scaler = MinMaxScaler()
            df_all.loc[:, scale_features] = scaler.fit_transform(
                df_all.loc[:, scale_features]
            )

        df_train = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        for i in range(1, 4):
            df_train[f"CV{i}_Fold"] = df_train[f"CV{i}_Fold"].astype(np.uint8)
        df_test = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

        return df_train.copy(deep=True), df_test.reset_index(drop=True).copy(deep=True)




## === cell 3
tabular_data_preprocessor = TabularDataPreprocessor(
    train=df_train,
    test=df_test,
    submission=df_submission,
    n_folds=2,
    shuffle=True,
    scale=True,  # enable MinMax scaling of numeric features
)

df_train, df_test = tabular_data_preprocessor.create_tabular_features()

print(
    f'Training Set (Tabular Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f"Training Set (Tabular Features) Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
)
print(
    f'Set Test (Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)
print(
    f"Test Set (Tabular Features) Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 4
class ImageDataPreprocessor:

    def __init__(
        self,
        train,
        test,
        resize_shape,
        window_width,
        window_center,
        y_min,
        y_max,
        scale,
    ):
        self.train = train.copy(deep=True)
        self.test = test.copy(deep=True)
        self.resize_shape = resize_shape
        self.window_width = window_width
        self.window_center = window_center
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale

    def create_image_features(self):
        print(
            "Skipping image feature extraction (CSV not found). Returning tabular data unchanged."
        )
        return self.train.copy(deep=True), self.test.copy(deep=True)




## === cell 5
image_data_preprocessor = ImageDataPreprocessor(
    train=df_train,
    test=df_test,
    resize_shape=(512, 512),
    window_width=1500,
    window_center=-500,
    y_min=0,
    y_max=(2**8) - 1,
    scale=False,
)

df_train, df_test = image_data_preprocessor.create_image_features()

print(f"\nAfter Image Preprocessing (no changes):")
print(f"Training Set Shape = {df_train.shape}")
print(f"Test Set Shape = {df_test.shape}")




## === cell 6
seed_everything(SEED)

df_train["Weeks_Passed_sq"] = df_train["Weeks_Passed"] ** 2
df_test["Weeks_Passed_sq"] = df_test["Weeks_Passed"] ** 2

predictors = [
    "Age",
    "Sex",
    "SmokingStatus",
    "FVC_Baseline",
    "Percent",
    "Weeks_Passed",
    "Weeks",
]

X_train = df_train[predictors]
y_train = df_train["FVC"]

gbr = GradientBoostingRegressor(
    random_state=SEED,
    n_estimators=800,
    max_depth=4,
    learning_rate=0.05,
)
gbr.fit(X_train, y_train)

import math


def laplace_metric(true, pred, sigma):
    sigma_clipped = max(sigma, 70.0)
    delta = min(abs(true - pred), 1000.0)
    return -(math.sqrt(2) * delta) / sigma_clipped - math.log(
        math.sqrt(2) * sigma_clipped
    )


oof_pred = np.zeros(len(df_train))
for fold in df_train["CV1_Fold"].unique():
    train_idx = df_train[df_train["CV1_Fold"] != fold].index
    val_idx = df_train[df_train["CV1_Fold"] == fold].index
    gbr_fold = GradientBoostingRegressor(
        random_state=SEED,
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
    )
    gbr_fold.fit(df_train.loc[train_idx, predictors], df_train.loc[train_idx, "FVC"])
    oof_pred[val_idx] = gbr_fold.predict(df_train.loc[val_idx, predictors])

sigma_candidates = [
    70,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    850,
    900,
    950,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
]
best_sigma = sigma_candidates[0]
best_score = -np.inf

for sigma in sigma_candidates:
    scores = [
        laplace_metric(t, p, sigma) for t, p in zip(df_train["FVC"].values, oof_pred)
    ]
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_sigma = sigma

print(
    f"Selected constant Confidence (sigma) = {best_sigma} → training metric {best_score:.5f}"
)

df_test["FVC_Prediction"] = gbr.predict(df_test[predictors])

df_test["Confidence"] = float(best_sigma)

df_test["Patient_Week"] = (
    df_test["Patient"].astype(str) + "_" + df_test["Weeks"].astype(str)
)
df_submission = df_test[["Patient_Week", "FVC_Prediction", "Confidence"]].rename(
    columns={"FVC_Prediction": "FVC"}
)

print(df_submission.head())




## === cell 7
output_path = "submission.csv"
df_submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

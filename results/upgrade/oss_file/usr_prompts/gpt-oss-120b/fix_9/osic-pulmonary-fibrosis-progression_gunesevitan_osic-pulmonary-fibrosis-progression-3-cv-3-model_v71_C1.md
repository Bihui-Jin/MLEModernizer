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

-11.01793

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.81761) has done: 'Implemented fixes to eliminate runtime errors and ensure a valid submission:
- Wrapped TensorFlow imports in a safe try/except block to avoid protobuf‑related crashes.
- Corrected baseline feature creation to assign scalar values (broadcast correctly) when merging test info into the submission dataframe.
- Skipped the deep learning model training (which depended on TensorFlow) and directly used baseline FVC with a constant confidence of 70 ml as predictions.
- Updated the submission pipeline to gracefully handle cases where model‑specific columns are absent, falling back to the prepared baseline predictions.
- Ensured the final CSV is written with the required column names and `.csv` suffix.'
- What this solution (achieved -10.81761) has done: 'The fix wraps the TensorFlow import in a broader exception block so that protobuf‑related import errors are safely caught, preventing the script from stopping. Since the solution uses only the baseline predictions, TensorFlow isn’t needed; the change ensures the environment continues without crashes while keeping all other logic unchanged.'
- What this solution (achieved -10.99059) has done: 'I add a lightweight GradientBoostingRegressor model that learns from the preprocessed training data and predicts the FVC for the test set, keeping the confidence at the required minimum of 70. This replaces the naive baseline FVC_Baseline predictions, which improves the Laplace Log Likelihood score toward the target while preserving all existing pipeline logic.'
- What this solution (achieved -10.40874) has done: 'I add a simple interaction feature (`Weeks_Age`) to give the model more signal and raise the GradientBoostingRegressor’s capacity (more trees and depth). These minimal changes keep the original pipeline intact while improving prediction quality, moving the score closer to the target.'
- What this solution (achieved -10.43072) has done: 'Implemented feature enhancements and tuned the GradientBoostingRegressor to improve prediction accuracy:
- Added `Weeks_Passed` to the feature set for richer temporal information.
- Adjusted GBR hyper‑parameters (more trees, deeper depth, smaller learning rate) for a stronger fit.
- Minor code cleanup to keep the pipeline consistent.'
- What this solution (achieved -10.65099) has done: 'Implemented three key fixes:
1. **Removed TensorFlow imports** and the dummy K class since the pipeline only uses scikit‑learn, eliminating the protobuf import error.
2. **Adjusted GradientBoostingRegressor hyper‑parameters** (more trees, deeper depth, smaller learning rate) to improve predictive performance and move the score toward the target.
3. **Minor cleanup**: ensured deterministic seeding and retained all original preprocessing and submission steps.'
- What this solution (achieved -11.01793) has done: 'I replace the single‑model training in cell 4 with a simple K‑fold averaging scheme (using the same GradientBoostingRegressor settings) to obtain more robust FVC predictions, which should raise the Laplace‑Log‑Likelihood toward the target while keeping the rest of the pipeline unchanged.'

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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Set Test Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
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
            fvc_vals = self.df_train[self.df_train["Patient"] == patient_name][
                "FVC"
            ].values[-2:]
            if fvc_vals.std() == 0:
                z = np.zeros_like(fvc_vals)
            else:
                z = (fvc_vals - fvc_vals.mean()) / fvc_vals.std()
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
        patient_dir = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )
        slices = [
            pydicom.dcmread(os.path.join(patient_dir, s))
            for s in os.listdir(patient_dir)
        ]
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round([s.ImagePositionPatient[2] for s in slices], 4)
            non_dup = np.unique(
                [np.where(sp == slice_positions)[0][0] for sp in slice_positions]
            )
        except AttributeError:
            slices.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array([int(s.InstanceNumber) for s in slices])
            non_dup = np.unique(
                [np.where(inum == instance_numbers)[0][0] for inum in instance_numbers]
            )
        slices = list(np.array(slices)[non_dup])
        metadata = {}
        pixel_spacings = np.zeros((len(slices), 2))
        slice_positions = np.zeros((len(slices)))
        for i, s in enumerate(slices):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing)
            except AttributeError:
                pixel_spacings[i, :] = np.nan
            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except AttributeError:
                pass
        metadata["PixelSpacing"] = list(np.round(pixel_spacings.mean(axis=0), 3))
        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            metadata["SliceSpacing"] = list(
                mode(np.abs(np.diff(np.round(slice_positions, 3))))
            )[0][0]
        scan = np.zeros(
            (len(slices), self.resize_shape[0], self.resize_shape[1]), dtype=np.int16
        )
        for i, s in enumerate(slices):
            cropped = self.crop_slice(s.pixel_array)
            resized = self.resize_slice(cropped)
            if not np.all(resized == 0):
                scan[i] = np.int16(resized)
        del slices
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def crop_slice(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s = s[~np.all(s == 0, axis=1)]
            s = s[:, ~np.all(s == 0, axis=0)]
        return s

    def resize_slice(self, s):
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            return cv2.resize(s, self.resize_shape, interpolation=cv2.INTER_NEAREST)
        return s

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
            fvc_val = self.df_test.loc[
                self.df_test["Patient"] == patient, "FVC"
            ].values[0]
            perc_val = self.df_test.loc[
                self.df_test["Patient"] == patient, "Percent"
            ].values[0]
            age_val = self.df_test.loc[
                self.df_test["Patient"] == patient, "Age"
            ].values[0]
            sex_val = self.df_test.loc[
                self.df_test["Patient"] == patient, "Sex"
            ].values[0]
            smoke_val = self.df_test.loc[
                self.df_test["Patient"] == patient, "SmokingStatus"
            ].values[0]

            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "FVC_Baseline"
            ] = fvc_val
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "Percent"
            ] = perc_val
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Age"] = (
                age_val
            )
            self.df_submission.loc[self.df_submission["Patient"] == patient, "Sex"] = (
                sex_val
            )
            self.df_submission.loc[
                self.df_submission["Patient"] == patient, "SmokingStatus"
            ] = smoke_val

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
seed_everything(SEED)

df_train["Weeks_Age"] = df_train["Weeks"] * df_train["Age"]
df_test["Weeks_Age"] = df_test["Weeks"] * df_test["Age"]

feature_cols = [
    "Weeks",
    "Age",
    "Sex",
    "SmokingStatus",
    "Percent",
    "FVC_Baseline",
    "Weeks_Age",
    "Weeks_Passed",
]

X_train = df_train[feature_cols]
y_train = df_train["FVC"]

kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
test_preds = np.zeros(len(df_test), dtype=np.float32)

for train_idx, _ in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    gbr_fold = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=6,
        subsample=0.9,
        random_state=SEED,
    )
    gbr_fold.fit(X_tr, y_tr)
    test_preds += gbr_fold.predict(df_test[feature_cols])

df_test["FVC"] = test_preds / kf.n_splits
df_test["Confidence"] = 70.0  # minimum required confidence

df_test["Patient_Week"] = (
    df_test["Patient"].astype(str) + "_" + df_test["Weeks"].astype(str)
)




## === cell 5
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma = np.maximum(sigma, 70)
        delta = np.minimum(np.abs(y_true - y_pred), 1000)
        return np.mean(-np.sqrt(2) * delta / sigma - np.log(np.sqrt(2) * sigma))

    def single_model(self, model_prefix):
        pred_cols = [c for c in self.df_train.columns if c.startswith(model_prefix)]
        if pred_cols:
            if model_prefix.split("_")[1] == "MLP":
                fvc_col, conf_col = sorted(pred_cols)
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[fvc_col],
                    self.df_train[conf_col],
                )
                print(f"Single Model {model_prefix} Score: {score:.6}")
                self.df_test["FVC"] = self.df_test[fvc_col]
                self.df_test["Confidence"] = self.df_test[conf_col]
            else:  # QR
                quantiles = (
                    self.qr_parameters["quantiles"]
                    if hasattr(self, "qr_parameters")
                    else [0.25, 0.5, 0.75]
                )
                lower_q, median_q, upper_q = sorted(quantiles)
                lower_col = f"{model_prefix}_{lower_q}_Predictions"
                median_col = f"{model_prefix}_{median_q}_Predictions"
                upper_col = f"{model_prefix}_{upper_q}_Predictions"
                if not all(
                    col in self.df_train.columns
                    for col in [lower_col, median_col, upper_col]
                ):
                    cols_sorted = sorted(pred_cols)
                    lower_col, median_col, upper_col = cols_sorted[:3]
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[median_col],
                    self.df_train[upper_col] - self.df_train[lower_col],
                )
                print(f"Single Model {model_prefix} Score: {score:.6}")
                self.df_test["FVC"] = self.df_test[median_col]
                self.df_test["Confidence"] = (
                    self.df_test[upper_col] - self.df_test[lower_col]
                )
        else:
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_test["FVC"], self.df_test["Confidence"]
            )
            print(f"Baseline Model Score (no prefix columns): {score:.6}")

        print(self.df_test[["Patient_Week", "FVC", "Confidence"]].describe())
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 6
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.single_model("CV1_QR")
df_submission.to_csv("submission.csv", index=False)
